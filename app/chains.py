"""LangChain building blocks: the router and the two answer chains.

Everything here is a LangChain Runnable (prompt | model | parser), so each
piece can be streamed, traced (LangSmith picks it up from env vars) and
tested on its own with a fake model.
"""

from __future__ import annotations

import re
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableBranch

from .calculator import is_plain_expression, normalize
from .config import Settings
from .schemas import RouteDecision

# ---------------------------------------------------------------- models


def build_chat_model(spec: str, settings: Settings) -> BaseChatModel:
    """Create a chat model from a "provider:model" string.

    The provider SDK reads its API key from the environment itself
    (OPENAI_API_KEY and so on), so the key never passes through our code.
    """
    kwargs: dict[str, Any] = {
        "timeout": settings.llm_timeout_s,
        "max_retries": settings.llm_max_retries,
    }
    if settings.reasoning_effort:
        kwargs["reasoning_effort"] = settings.reasoning_effort
    return init_chat_model(spec, **kwargs)


# ---------------------------------------------------------------- router

ROUTER_SYSTEM = """You route user questions to one of two backends.

math: the question is about mathematics. This includes arithmetic, \
percentages, unit-free word problems with numbers, powers, roots, logs, \
trigonometry, and questions about math concepts or methods.
general: everything else.

If the route is math and the answer is a single number that can be computed, \
also return one arithmetic expression that computes it. Otherwise return null \
for the expression. Do not answer the question."""

ROUTER_PROMPT = ChatPromptTemplate.from_messages([("system", ROUTER_SYSTEM), ("human", "{query}")])


def build_router_chain(llm: BaseChatModel) -> Runnable[dict, RouteDecision]:
    """LLM router: prompt | model constrained to the RouteDecision schema."""
    return (ROUTER_PROMPT | llm.with_structured_output(RouteDecision)).with_config(run_name="router")


# Phrases people put around a bare expression, e.g. "what is 2+2?".
_PREFIX = re.compile(r"^\s*(?:what\s+is|what's|whats|calculate|compute|evaluate|solve)\s+", re.I)
_SUFFIX = re.compile(r"[\s?=!.]+$")
# Numbers joined only by "-" or "/" (9/11, 2024-12-25, 555-1234, 3-1) are
# more often dates, phone numbers, scores or names than arithmetic.
_AMBIGUOUS = re.compile(r"^\d+(?:\s*[-/]\s*\d+)+$")


def fast_route(query: str) -> RouteDecision | None:
    """Zero-latency routing for queries that are just an arithmetic expression.

    If the query (minus a polite prefix) parses as a valid expression, we
    skip the router LLM round trip entirely. The rule is deliberately narrow:
    anything that does not fully parse, and anything that looks like a date,
    a phone number or a score, returns None and goes to the LLM router. A
    miss only costs latency, a wrong hit would give a wrong answer.
    """
    candidate = _SUFFIX.sub("", _PREFIX.sub("", query))
    if not candidate or not any(ch.isdigit() for ch in candidate) or _AMBIGUOUS.match(candidate):
        return None
    if is_plain_expression(candidate):
        return RouteDecision(route="math", expression=normalize(candidate))
    return None


# ---------------------------------------------------------------- answers

GENERAL_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer clearly and directly. "
            "Prefer short answers unless the user asks for depth.",
        ),
        ("human", "{query}"),
    ]
)

MATH_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a precise math assistant.\n{tool_context}\n"
            "Lead with the answer, then explain briefly. Keep it short.",
        ),
        ("human", "{query}"),
    ]
)


def tool_context(expression: str | None, result: str | None) -> str:
    """Text injected into the math prompt so the LLM explains, not computes."""
    if expression is not None and result is not None:
        return (
            f"A calculator evaluated `{expression}` = {result}. This value is exact. "
            "Use it as the final answer and do not recompute it."
        )
    return "No calculator result is available for this question. If it needs arithmetic, show the steps."


def build_general_chain(llm: BaseChatModel) -> Runnable[dict, str]:
    return (GENERAL_PROMPT | llm | StrOutputParser()).with_config(run_name="general_chain")


def build_math_chain(llm: BaseChatModel) -> Runnable[dict, str]:
    return (MATH_PROMPT | llm | StrOutputParser()).with_config(run_name="math_chain")


def build_answer_branch(
    general_chain: Runnable[dict, str], math_chain: Runnable[dict, str]
) -> Runnable[dict, str]:
    """LangChain dispatch from a routing decision to the chain that answers.

    Input: {"route": "math" | "general", "query": ..., "tool_context": ...}.
    RunnableBranch streams whatever the selected chain streams, so tokens
    still arrive one by one.
    """
    return RunnableBranch(
        (lambda inputs: inputs["route"] == "math", math_chain),
        general_chain,
    ).with_config(run_name="answer_branch")
