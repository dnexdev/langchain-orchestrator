"""Shared fixtures."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from app.calculator import calculator
from app.chains import build_general_chain, build_math_chain
from app.orchestrator import Orchestrator
from tests.fakes import ScriptedChatModel, make_router


@pytest.fixture
def build() -> Callable[..., tuple[Orchestrator, ScriptedChatModel, ScriptedChatModel]]:
    """Build an orchestrator from real chains wired to scripted models."""

    def _build(
        router: Any = None,
        general: ScriptedChatModel | None = None,
        math: ScriptedChatModel | None = None,
        speculative: bool = True,
        idle_timeout: float = 5.0,
    ) -> tuple[Orchestrator, ScriptedChatModel, ScriptedChatModel]:
        general = general or ScriptedChatModel()
        math = math or ScriptedChatModel(reply="The answer is 42.")
        orch = Orchestrator(
            router=router or make_router("general"),
            general_chain=build_general_chain(general),
            math_chain=build_math_chain(math),
            calculator=calculator,
            speculative_routing=speculative,
            stream_idle_timeout_s=idle_timeout,
        )
        return orch, general, math

    return _build
