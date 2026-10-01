# Streaming LangChain Orchestrator

A FastAPI service with one endpoint, `POST /ask`. It uses LangChain to decide whether a question is about math or something general, sends it to the matching chain, and streams the answer back token by token as Server-Sent Events.

The focus is on the parts that matter in production: time to first token, exact math, safe handling of LLM output, clean cancellation, and keeping secrets out of the code.

Example run (timings from a local mock provider, they vary by model):

```
$ python scripts/ask.py "What is 15% of 80?"
[route] math via llm in 412 ms | expression: 0.15 * 80
[tool] calculator(0.15 * 80) -> 12
15% of 80 is 12. Multiply 80 by 0.15 ...
[done] first token 720 ms, total 1221 ms, 10 chunks, speculation: cancelled
```

## Quick start

Requires Python 3.11+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

cp .env.example .env          # then put your key in OPENAI_API_KEY
uvicorn app.main:app --reload
```

In a second terminal:

```bash
python scripts/ask.py "Who wrote Dune?"
python scripts/ask.py "what is 2^10"

# or raw SSE with curl (-N turns off curl's buffering)
curl -N -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is 15% of 80?"}'
```

Interactive API docs are at <http://127.0.0.1:8000/docs>. Tests need no API key: `pytest -q`.

With Docker:

```bash
docker build -t orchestrator .
docker run --rm -p 8000:8000 --env-file .env orchestrator
```

## API

`POST /ask` with body `{"query": "..."}` (1 to 4000 characters, no extra fields).

The response is `text/event-stream`. Every event has a name and a JSON `data` payload:

| event   | when                          | data                                                                 |
|---------|-------------------------------|----------------------------------------------------------------------|
| `route` | once, first                   | `route` (math or general), `source` (rules, llm or fallback), `expression`, `ms` |
| `tool`  | math questions with a number  | `name`, `expression`, and `result` or `error`                        |
| `token` | many times                    | `text`, one chunk of the LLM output                                  |
| `done`  | once, last                    | `route`, `chunks`, `first_token_ms`, `total_ms`, `speculation`       |
| `error` | instead of `done` on failure  | `message`, `request_id`                                              |

Raw stream for `what is 6*7` (token text varies by model):

```
event: route
data: {"route": "math", "source": "rules", "expression": "6*7", "ms": 0}

event: tool
data: {"name": "calculator", "expression": "6*7", "result": "42"}

event: token
data: {"text": "6 × 7 = "}

event: token
data: {"text": "42"}
...
event: done
data: {"route": "math", "chunks": 9, "first_token_ms": 308, "total_ms": 810, "speculation": null}
```

The response also carries `X-Request-ID`, which matches the server log line for that request. `GET /health` is a liveness probe.

## How it works

```
                  POST /ask {"query": ...}
                             |
              +--------------+---------------+
              | 1. rules fast path           |  "2+2", "what is 12*(3+4)?"
              |    query parses as math?     |---- yes ----> math (0 ms)
              +--------------+---------------+
                             | no
        +--------------------+---------------------+
        | 2. LLM router          | general chain    |   both start at the same time
        |    structured output   | streams into a   |   ("speculative routing")
        |    {route, expression} | buffer           |
        +--------------------+---------------------+
                             |
             +---------------+----------------+
             | math                           | general
             v                                v
   stop the buffered draft            flush the buffer, keep streaming
   calculator tool (exact result)
   RunnableBranch -> math chain       (or RunnableBranch -> general chain
   explains the result                 when speculation is off)
             |                                |
             +---------------+----------------+
                             v
            SSE: route, tool, token, token, ..., done
```

Everything that talks to a model is a LangChain Runnable built with LCEL:

| piece            | LangChain construct                                                     | file             |
|------------------|-------------------------------------------------------------------------|------------------|
| router           | `ChatPromptTemplate \| llm.with_structured_output(RouteDecision)`        | `app/chains.py`  |
| general chain    | `ChatPromptTemplate \| llm \| StrOutputParser()`                         | `app/chains.py`  |
| math chain       | same shape, with the calculator result injected into the system prompt   | `app/chains.py`  |
| dispatch         | `RunnableBranch((route == "math", math_chain), general_chain)`           | `app/chains.py`  |
| math tool        | `@tool("calculator")` with a Pydantic args schema                        | `app/calculator.py` |
| model factory    | `init_chat_model("provider:model")`                                      | `app/chains.py`  |

`app/orchestrator.py` runs these with asyncio and turns the results into events. `app/main.py` is only the HTTP layer.

### Where each acceptance criterion is met

| criterion                                   | where                                                                                               |
|---------------------------------------------|-----------------------------------------------------------------------------------------------------|
| LangChain routing and orchestration         | `build_router_chain`, `build_answer_branch`, the chains and the `calculator` tool in `app/chains.py` and `app/calculator.py`, wired in `Orchestrator.stream` |
| Chunked streaming response over SSE         | `ask()` in `app/main.py` (`EventSourceResponse`, one event per token), tested over a real socket in `tests/test_api.py` |
| No API keys in code                         | `app/config.py` (environment and `.env` only), `.env.example`, `.gitignore`, key scan in `tests/test_security.py` |

## Design decisions

**1. Two stage router: rules first, then an LLM.**
If the whole query is a valid arithmetic expression (after dropping phrases like "what is"), the service skips the router LLM call and saves a full round trip. The rule is deliberately narrow. Anything that does not fully parse goes to the LLM router, and so does anything that looks like a date, a phone number or a score (`9/11`, `2024-12-25`, `555-1234`), since a miss only costs latency but a wrong hit would give a wrong answer. The LLM router uses structured output, so its answer is a validated Pydantic object, not text I have to parse. Its schema has only two fields (`route`, `expression`) because every output token of the router is time the user waits before seeing anything.

**2. Speculative routing to cut time to first token.**
Waiting for the router and then starting the answer puts two model latencies in a row. Most traffic is general, so the general chain starts at the same moment as the router and buffers its tokens in a background task. If the router says general, the buffer is flushed right away. If it says math, the draft is stopped, which also closes its HTTP connection so no more tokens are billed. Against a local mock provider with 400 ms router latency and 300 ms time to first token, a general question went from 759 ms to 438 ms first token. The cost is paid on LLM routed math questions only: the general prompt's input tokens plus whatever the draft streamed before it was stopped. Set `SPECULATIVE_ROUTING=false` to turn it off, and the `done` event reports whether speculation was a `hit` or `cancelled`.

**3. The LLM writes the expression, a calculator computes it, the LLM explains it.**
LLMs are unreliable at arithmetic. The router already returns an expression like `0.15 * 80`, so the math path evaluates it with a deterministic tool and gives the exact result to the math chain with the instruction to use it and not recompute it. The result is also sent as its own `tool` event, so a client has the exact number before the explanation starts streaming. Conceptual math ("why is the derivative of x² equal to 2x?") has no expression and goes to the math chain without a tool result.

**4. A safe evaluator instead of `eval()` or LLMMathChain.**
The expression is written by an LLM that is steered by user text, so it is untrusted input. LangChain's old LLMMathChain evaluated model output and was assigned CVE-2023-29374 (code execution through prompt injection). `app/calculator.py` parses the expression into a Python AST and only allows numbers, arithmetic operators, a fixed list of math functions and three constants. Names, attributes, calls to anything else, keywords, strings and comparisons are rejected before anything runs. It also bounds the operations that can hang a CPU, like `9**9**9`, `factorial(100000)` or `round(5, -10**9)`, and each of those is rejected in microseconds.

**5. FastAPI's native SSE support.**
`EventSourceResponse` (a `StreamingResponse` subclass) plus `ServerSentEvent` handles the wire format. Each event's data is JSON, so a token that contains a newline cannot break the SSE framing (there is a test for exactly this). It also sends a keep-alive comment every 15 seconds and sets `Cache-Control: no-cache` and `X-Accel-Buffering: no`, so proxies like nginx do not hold the stream back.

**6. Streams are always cleaned up.**
Each chain runs in its own task (`PrefetchedStream`). When the client disconnects, the request task is cancelled and a `finally` block stops every model stream it started. Stopping is graceful first: the reader quits at the next chunk and calls `aclose()` on the LangChain stream, which unwinds every layer down to the provider's HTTP response. I found that cancelling a task in the middle of a LangChain step can leave the innermost model generator open until garbage collection, because LangChain runs each step in a helper task. So a hard cancel is only the fallback for a stream that stays silent past a 0.5 s grace period. I checked this against the real `ChatOpenAI` client and a mock provider: the upstream stream closes when the client goes away, and when a speculative draft is dropped. A stream that goes silent for `STREAM_IDLE_TIMEOUT_S` (this includes the wait for the first token) is also stopped and reported as an `error` event.

**7. Built once, fail fast.**
Models are created once in the FastAPI lifespan, so the provider's HTTP connection pool is reused instead of opening a new TLS connection per request. If a required key is missing, startup fails with a message that names the variable. The first user request never finds out.

**8. Provider agnostic.**
Models are configured as `provider:model` strings through `init_chat_model`. The default is `openai:gpt-5.6-luna` for both the router and answers, because it is OpenAI's tier meant for latency sensitive work. `ROUTER_MODEL` and `ANSWER_MODEL` can differ, for example a small router and a larger answer model. Switching to Anthropic is `pip install langchain-anthropic` and `ANSWER_MODEL=anthropic:<model>`.

**9. Errors after the first byte are events.**
Once streaming starts the status code is already 200, so failures are sent as an `error` event. The message is generic on purpose, since provider errors can contain account details. The full exception goes to the server log with the request ID. If the router itself fails, the request falls back to the general chain instead of failing.

## Security

- No keys in code. Keys are read from the environment, optionally seeded from a local `.env` through `python-dotenv`. Real environment variables win over `.env`.
- `.env` is in `.gitignore` and `.dockerignore`. `.env.example` lists variable names with empty values.
- The provider SDK reads its key from the environment itself, so the key never passes through this code, and the config check only ever reports variable names.
- `tests/test_security.py` scans every file in the repo for the shapes of real OpenAI, Anthropic, AWS, Google, Groq and LangSmith keys, so CI fails if one is committed.
- LLM output is treated as untrusted. It is never executed, only parsed by the whitelist evaluator.
- Input is validated: 1 to 4000 characters, extra JSON fields rejected.
- Optional service auth: if `SERVICE_API_KEY` is set, callers must send it in `X-API-Key`. It is compared with `secrets.compare_digest` to avoid timing leaks.
- The Docker image runs as a non-root user and gets keys at runtime, never at build time.

## Configuration

All settings come from environment variables (see `.env.example`).

| variable                | default               | purpose                                          |
|-------------------------|-----------------------|--------------------------------------------------|
| `OPENAI_API_KEY`        | none (required)       | key for the default provider                     |
| `ROUTER_MODEL`          | `openai:gpt-5.6-luna` | model that classifies the query                  |
| `ANSWER_MODEL`          | `openai:gpt-5.6-luna` | model that writes the answer                     |
| `REASONING_EFFORT`      | unset                 | passed to models that support it                 |
| `SPECULATIVE_ROUTING`   | `true`                | start the general answer during routing          |
| `LLM_TIMEOUT_S`         | `30`                  | per request timeout to the provider              |
| `LLM_MAX_RETRIES`       | `2`                   | provider retries (before the first token)        |
| `STREAM_IDLE_TIMEOUT_S` | `30`                  | max wait for the first chunk and between chunks  |
| `SERVICE_API_KEY`       | unset                 | require `X-API-Key` from callers                 |
| `LANGSMITH_TRACING`     | unset                 | trace every chain run in LangSmith               |

## Tests

```bash
pytest -q        # 105 tests, about 4 seconds, no network, no API key
ruff check .
```

The tests use a scripted LangChain chat model with configurable latency, plugged into the real chains. They cover:

- calculator results, and rejection of code execution and resource exhaustion attempts
- the rules fast path, including what it must not catch (dates, phone numbers, overflow)
- the `RunnableBranch` dispatch still streams token by token
- every route: general, rules math, LLM math, conceptual math, bad expressions, router failure
- speculative routing: time to first token drops from router + model to max(router, model), the draft is stopped on math, and a stalled draft never delays the math answer
- model failure and stalled streams become `error` events without leaking details
- SSE headers, framing, newlines inside tokens, validation errors and service auth
- a real uvicorn server: tokens arrive over time with chunked transfer encoding, and closing the connection cancels the model stream
- 20 concurrent requests stay independent
- no keys in the repo, missing keys fail fast

CI runs lint and tests on Python 3.11 and 3.12 (`.github/workflows/ci.yml`).

## Project layout

```
app/
  main.py          FastAPI app, lifespan, /ask SSE endpoint, optional auth
  orchestrator.py  async routing, speculative execution, cancellation, events
  chains.py        LangChain router, general chain, math chain, model factory
  calculator.py    whitelist AST evaluator exposed as a LangChain tool
  schemas.py       request model and the router's structured output schema
  config.py        settings from environment variables, fail-fast key check
scripts/ask.py     terminal client that prints the stream as it arrives
tests/             pytest suite (no network needed)
```

## Limitations and next steps

- Symbolic math (solving equations, derivatives) is explained by the math chain but not checked by a tool. SymPy could be added as a second tool.
- No conversation memory. Each request is independent.
- For real production traffic I would add per-client rate limiting, a concurrency cap per worker to protect provider rate limits, and OpenTelemetry tracing.

## License

MIT, see `LICENSE.txt`.
