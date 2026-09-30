"""
This is a FastAPI entry point.
POST /ask streams the answer as Server-Sent Events.

Run command is:
uvicorn app,main:app --reload
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
from .chains import (
    build_chat_model,
    build_general_chain,
    build_math_chain,
    build_router_chain,
)
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
