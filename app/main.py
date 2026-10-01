"""FastAPI entry point: POST /ask streams the answer as Server-Sent Events.

Run it with:
    uvicorn app.main:app --reload
"""

from __future__ import annotations

import logging
import os
import secrets
import uuid
from collections.abc import AsyncIterable
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response, status
from fastapi.sse import EventSourceResponse, ServerSentEvent

from .calculator import calculator
from .chains import build_chat_model, build_general_chain, build_math_chain, build_router_chain
from .config import Settings
from .orchestrator import Orchestrator
from .schemas import AskRequest

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)


def build_orchestrator(settings: Settings) -> Orchestrator:
    settings.require_provider_keys()
    router_llm = build_chat_model(settings.router_model, settings)
    answer_llm = (
        router_llm
        if settings.answer_model == settings.router_model
        else build_chat_model(settings.answer_model, settings)
    )
    return Orchestrator(
        router=build_router_chain(router_llm),
        general_chain=build_general_chain(answer_llm),
        math_chain=build_math_chain(answer_llm),
        calculator=calculator,
        speculative_routing=settings.speculative_routing,
        stream_idle_timeout_s=settings.stream_idle_timeout_s,
    )


def create_app(settings: Settings | None = None, orchestrator: Orchestrator | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Build the models once per process instead of once per request, so
        # the provider's HTTP connection pool (and its TLS sessions) is reused.
        # Missing keys fail here, at startup, not on the first user request.
        app.state.orchestrator = orchestrator or build_orchestrator(settings)
        yield

    app = FastAPI(
        title="Streaming LangChain Orchestrator",
        description="Routes a query to a math or general LangChain chain and streams the answer over SSE.",
        version="1.0.0",
        lifespan=lifespan,
    )

    def request_id(response: Response) -> str:
        # Set as a dependency so the header is attached before streaming starts.
        rid = uuid.uuid4().hex[:12]
        response.headers["X-Request-ID"] = rid
        return rid

    def require_service_key(x_api_key: str | None = Header(default=None)) -> None:
        # Optional auth for callers of this service. Disabled unless
        # SERVICE_API_KEY is set. compare_digest avoids timing leaks.
        expected = settings.service_api_key
        if expected is None:
            return
        if x_api_key is None or not secrets.compare_digest(x_api_key.encode(), expected.encode()):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or missing X-API-Key")

    @app.post(
        "/ask",
        response_class=EventSourceResponse,
        dependencies=[Depends(require_service_key)],
        summary="Ask a question and stream the answer",
    )
    async def ask(
        body: AskRequest,
        request: Request,
        rid: str = Depends(request_id),
    ) -> AsyncIterable[ServerSentEvent]:
        """Stream events: `route`, optional `tool`, many `token`, then `done` or `error`.

        Each event's `data` is JSON, so tokens containing newlines cannot
        break the SSE framing.
        """
        orchestrator: Orchestrator = request.app.state.orchestrator
        async for event in orchestrator.stream(body.query, request_id=rid):
            yield ServerSentEvent(event=event.name, data=event.data)

    @app.get("/health", summary="Liveness probe")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
