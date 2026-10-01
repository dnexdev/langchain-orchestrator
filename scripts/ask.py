"""Tiny terminal client that prints the SSE stream as it arrives.

    python scripts/ask.py "What is 15% of 80?"
    python scripts/ask.py --url http://localhost:8000 "Who wrote Dune?"

Tokens are printed the moment each chunk lands, and the route, tool and
timing events are shown in grey so you can see the orchestration.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import httpx  # already installed as a dependency of the OpenAI SDK

GREY, RESET = ("\033[90m", "\033[0m") if sys.stdout.isatty() else ("", "")


def meta(text: str) -> None:
    print(f"{GREY}{text}{RESET}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("query")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()

    headers = {"Accept": "text/event-stream"}
    if key := os.getenv("SERVICE_API_KEY"):
        headers["X-API-Key"] = key

    event = "message"
    with httpx.stream(
        "POST", f"{args.url}/ask", json={"query": args.query}, headers=headers, timeout=60
    ) as response:
        if response.status_code != 200:
            print(f"HTTP {response.status_code}: {response.read().decode()}", file=sys.stderr)
            return 1
        meta(f"request {response.headers.get('x-request-id')}")
        for line in response.iter_lines():
            if line.startswith("event:"):
                event = line[6:].strip()
                continue
            if not line.startswith("data:"):
                continue
            data = json.loads(line[5:])
            if event == "token":
                print(data["text"], end="", flush=True)
            elif event == "route":
                expr = f" | expression: {data['expression']}" if data.get("expression") else ""
                meta(f"[route] {data['route']} via {data['source']} in {data['ms']} ms{expr}")
            elif event == "tool":
                outcome = data.get("result", f"error: {data.get('error')}")
                meta(f"[tool] {data['name']}({data['expression']}) -> {outcome}")
            elif event == "done":
                print()
                meta(
                    f"[done] first token {data['first_token_ms']} ms, total {data['total_ms']} ms, "
                    f"{data['chunks']} chunks, speculation: {data['speculation']}"
                )
            elif event == "error":
                print()
                meta(f"[error] {data['message']} (request {data['request_id']})")
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
