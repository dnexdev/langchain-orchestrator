"""Async orchestration: route the query, run the right chain, stream events.

Flow for one request:

    query
      |-- fast_route()  (pure arithmetic like "12*(3+4)")  -> math, 0 ms
      |-- otherwise LLM router (structured output)          -> math | general
      |        while it runs, the general chain is already streaming into a
      |        buffer ("speculative routing", see below)
      |
      |-- math:    calculator tool -> math chain streams an explanation
      |-- general: keep the speculative stream, or start the general chain
      v
    events: route, tool, token..., done   (or error)

Speculative routing
-------------------
The router is a full LLM round trip, typically a few hundred ms, and the
answer model then needs its own time to first token. Doing them one after
the other puts both delays in front of the user. Most traffic is general, so
we start the general chain at the same moment as the router and buffer its
tokens. If the router says "general", the buffered tokens are flushed
immediately and time to first token becomes max(router, answer TTFT) instead
of router + answer TTFT. If it says "math", the draft is stopped, which also
closes its HTTP stream so we stop paying for tokens. The cost, on math
questions only, is the general prompt's input tokens plus whatever the draft
streamed before it was stopped. Turn it off with SPECULATIVE_ROUTING=false.

This module does not know about HTTP. It yields plain Event objects, and
main.py turns them into Server-Sent Events.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool

from .chains import build_answer_branch, fast_route, tool_context
from .schemas import RouteDecision

log = logging.getLogger("orchestrator")


@dataclass(frozen=True)
class Event:
    name: str  # route | tool | token | done | error
    data: dict[str, Any] = field(default_factory=dict)


class StreamIdleTimeout(TimeoutError):
    """The upstream model went silent for too long in the middle of a stream."""


_END = object()


@dataclass(frozen=True)
class _Failure:
    exc: BaseException


class PrefetchedStream:
    """Consume an async iterator in a background task, buffering its items.

    This is what lets a chain start streaming before we know whether we
    need it. Reading happens in its own task, so tokens pile up in the queue
    while the caller is still awaiting the router.

    Stopping is graceful first: the pump quits at the next chunk boundary
    and then closes the LangChain stream with aclose(), which unwinds every
    layer down to the provider's HTTP response. Cancelling a task in the
    middle of a LangChain step can leave the innermost model generator
    suspended until garbage collection (LangChain runs steps in helper
    tasks), so a hard cancel is only the fallback for a stream that stays
    silent past the grace period.
    """

    def __init__(
        self, source: AsyncIterator[str], *, idle_timeout_s: float, name: str, grace_s: float = 0.5
    ) -> None:
        self._queue: asyncio.Queue[object] = asyncio.Queue()
        self._idle_timeout_s = idle_timeout_s
        self._grace_s = grace_s
        self._stop_requested = False
        self._task = asyncio.create_task(self._pump(source), name=name)

    async def _pump(self, source: AsyncIterator[str]) -> None:
        try:
            async for item in source:
                if self._stop_requested:
                    break
                self._queue.put_nowait(item)
        except Exception as exc:  # handed to the consumer, re-raised there
            self._queue.put_nowait(_Failure(exc))
        finally:
            # Close the stream now instead of whenever garbage collection
            # runs. Cleanup errors must never mask the real outcome.
            aclose = getattr(source, "aclose", None)
            if aclose is not None:
                with contextlib.suppress(Exception):
                    await aclose()
            self._queue.put_nowait(_END)

    def __aiter__(self) -> PrefetchedStream:
        return self

    async def __anext__(self) -> str:
        try:
            async with asyncio.timeout(self._idle_timeout_s):
                item = await self._queue.get()
        except TimeoutError:
            self.stop()
            raise StreamIdleTimeout(f"no output from the model for {self._idle_timeout_s:.0f}s") from None
        if item is _END:
            raise StopAsyncIteration
        if isinstance(item, _Failure):
            raise item.exc
        return item  # type: ignore[return-value]

    def stop(self) -> None:
        """Ask the pump to stop, without waiting. Safe to call more than once.

        It stops at the next chunk (tens of ms for a streaming model). If no
        chunk comes within the grace period, the task is cancelled once.
        """
        if self._stop_requested or self._task.done():
            return
        self._stop_requested = True
        fallback = asyncio.get_running_loop().call_later(self._grace_s, self._task.cancel)
        self._task.add_done_callback(lambda _: fallback.cancel())

    async def cancel(self) -> None:
        """Stop and wait until the stream is closed.

        asyncio.wait() does not raise the child's CancelledError, so it
        cannot be confused with our own task being cancelled (for example
        because the client disconnected). If our wait is interrupted, the
        stop request above still completes on its own.
        """
        self.stop()
        await asyncio.wait({self._task})


def _ms(start: float) -> int:
    return round((time.perf_counter() - start) * 1000)


class Orchestrator:
    def __init__(
        self,
        *,
        router: Runnable[dict, RouteDecision],
        general_chain: Runnable[dict, str],
        math_chain: Runnable[dict, str],
        calculator: BaseTool,
        speculative_routing: bool = True,
        stream_idle_timeout_s: float = 20.0,
    ) -> None:
        self.router = router
        self.general_chain = general_chain
        # LangChain RunnableBranch that picks the answer chain from the route.
        self.answers = build_answer_branch(general_chain, math_chain)
        self.calculator = calculator
        self.speculative_routing = speculative_routing
        self.stream_idle_timeout_s = stream_idle_timeout_s

    def _start(self, chain: Runnable[dict, str], inputs: dict, name: str) -> PrefetchedStream:
        return PrefetchedStream(chain.astream(inputs), idle_timeout_s=self.stream_idle_timeout_s, name=name)

    async def _llm_route(self, query: str) -> tuple[RouteDecision, str]:
        """Ask the router chain. Returns the decision and where it came from."""
        try:
            return await self.router.ainvoke({"query": query}), "llm"
        except Exception:
            # A broken router should degrade the answer, not fail the request.
            log.warning("router failed, falling back to general", exc_info=True)
            return RouteDecision(route="general", expression=None), "fallback"

    async def stream(self, query: str, request_id: str = "-") -> AsyncIterator[Event]:
        t0 = time.perf_counter()
        inputs = {"query": query}
        draft: PrefetchedStream | None = None
        answer: PrefetchedStream | None = None
        speculation: str | None = None  # "hit", "cancelled" or None (not used)

        try:
            # The rules check is instant, so it runs first. Speculation only
            # pays off when we have to wait for the LLM router.
            rule = fast_route(query)
            if rule is not None:
                decision, source = rule, "rules"
            else:
                if self.speculative_routing:
                    draft = self._start(self.general_chain, inputs, f"draft-{request_id}")
                decision, source = await self._llm_route(query)
            yield Event(
                "route",
                {
                    "route": decision.route,
                    "source": source,
                    "expression": decision.expression,
                    "ms": _ms(t0),
                },
            )

            if decision.route == "math":
                expression, result = decision.expression, None
                if expression:
                    try:
                        result = await self.calculator.ainvoke({"expression": expression})
                        yield Event(
                            "tool",
                            {"name": self.calculator.name, "expression": expression, "result": result},
                        )
                    except Exception as exc:  # noqa: BLE001  invalid expression from the LLM
                        expression = None
                        yield Event(
                            "tool",
                            {
                                "name": self.calculator.name,
                                "expression": decision.expression,
                                "error": str(exc),
                            },
                        )
                if draft is not None:
                    # Do not wait for it: the math answer should not queue
                    # behind the draft's shutdown. The finally block below
                    # waits for it at the end of the request.
                    draft.stop()
                    speculation = "cancelled"
                answer = self._start(
                    self.answers,
                    {"route": "math", "query": query, "tool_context": tool_context(expression, result)},
                    f"math-{request_id}",
                )
            else:
                if draft is not None:
                    # The draft ran the general chain directly, since it had
                    # to start before the route was known.
                    answer, draft, speculation = draft, None, "hit"
                else:
                    answer = self._start(
                        self.answers, {"route": "general", "query": query}, f"general-{request_id}"
                    )

            first_token_ms: int | None = None
            chunks = 0
            async for text in answer:
                if not text:
                    continue
                if first_token_ms is None:
                    first_token_ms = _ms(t0)
                chunks += 1
                yield Event("token", {"text": text})

            yield Event(
                "done",
                {
                    "route": decision.route,
                    "chunks": chunks,
                    "first_token_ms": first_token_ms,
                    "total_ms": _ms(t0),
                    "speculation": speculation,
                },
            )
            log.info(
                "request_id=%s route=%s source=%s first_token_ms=%s total_ms=%s chunks=%d",
                request_id,
                decision.route,
                source,
                first_token_ms,
                _ms(t0),
                chunks,
            )

        except Exception as exc:
            # Headers (200) are already sent, so errors travel as an event.
            # The message is generic on purpose: provider errors can contain
            # account details we do not want to hand to clients.
            log.exception("request_id=%s stream failed", request_id)
            message = (
                "The model stopped responding."
                if isinstance(exc, StreamIdleTimeout)
                else "The language model request failed."
            )
            yield Event("error", {"message": message, "request_id": request_id})

        finally:
            # Runs on success, on error and when the client disconnects
            # (the task is cancelled). Never leave a model stream running.
            # Request every stop first, then wait. If the wait is interrupted
            # (client gone), each stream still shuts itself down.
            pending = [s for s in (draft, answer) if s is not None]
            for stream in pending:
                stream.stop()
            for stream in pending:
                await stream.cancel()
