"""Test doubles. No test needs a real API key or network access."""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import RunnableLambda
from pydantic import Field

from app.orchestrator import Event
from app.schemas import RouteDecision


class ScriptedChatModel(BaseChatModel):
    """Streams a fixed reply word by word, with configurable latency.

    Records every prompt it receives and whether a stream was cut off
    before the end, so tests can check cancellation.
    """

    reply: str = "Paris is the capital of France."
    first_token_delay: float = 0.0
    token_delay: float = 0.0
    error: str | None = None
    prompts: list[str] = Field(default_factory=list)
    finished: int = 0
    cancelled: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kw: Any
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.reply))])

    async def _astream(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kw: Any
    ) -> AsyncIterator[ChatGenerationChunk]:
        self.prompts.append("\n".join(str(m.content) for m in messages))
        try:
            await asyncio.sleep(self.first_token_delay)
            if self.error:
                raise RuntimeError(self.error)
            for token in re.split(r"(\s)", self.reply):
                yield ChatGenerationChunk(message=AIMessageChunk(content=token))
                await asyncio.sleep(self.token_delay)
            self.finished += 1
        except (asyncio.CancelledError, GeneratorExit):
            self.cancelled += 1
            raise


def make_router(route: str, expression: str | None = None, delay: float = 0.0, fail: bool = False):
    """A stand-in for the LLM router chain with a fixed answer and latency."""
    calls: list[str] = []

    async def _route(inputs: dict) -> RouteDecision:
        calls.append(inputs["query"])
        await asyncio.sleep(delay)
        if fail:
            raise ValueError("router returned invalid JSON")
        return RouteDecision(route=route, expression=expression)

    runnable = RunnableLambda(_route, name="fake_router")
    runnable.calls = calls  # type: ignore[attr-defined]
    return runnable


async def collect(stream: AsyncIterator[Event]) -> list[Event]:
    return [event async for event in stream]


def text_of(events: list[Event]) -> str:
    return "".join(e.data["text"] for e in events if e.name == "token")


async def eventually(check, within_s: float = 2.0) -> bool:
    """Poll until check() is true. Stream cleanup finishes on the next few
    event loop turns, not synchronously, so tests wait for it."""
    try:
        async with asyncio.timeout(within_s):
            while not check():  # noqa: ASYNC110  polling a plain counter is fine in a test
                await asyncio.sleep(0.01)
    except TimeoutError:
        return False
    return True
