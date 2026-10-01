import pytest
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableBranch

from app.chains import ROUTER_PROMPT, build_answer_branch, fast_route, tool_context
from app.config import provider_of
from app.schemas import RouteDecision
from tests.fakes import ScriptedChatModel


@pytest.mark.parametrize(
    ("query", "expression"),
    [
        ("2+2", "2+2"),
        ("what is 12 * (3 + 4)?", "12 * (3 + 4)"),
        ("Calculate 2^10", "2**10"),
        ("sqrt(144) =", "sqrt(144)"),
        ("  evaluate log(1000, 10)  ", "log(1000, 10)"),
    ],
)
def test_fast_path_catches_pure_arithmetic(query: str, expression: str) -> None:
    assert fast_route(query) == RouteDecision(route="math", expression=expression)


@pytest.mark.parametrize(
    "query",
    [
        "What is the capital of France?",
        "what is love",
        "42",
        "pi",
        "what is 15% of 80?",  # math, but needs the LLM to translate it
        "solve 2x + 3 = 7",
        "explain the Pythagorean theorem",
        "Write a haiku about 3 cats",
        "9/11",  # an event, not 0.818
        "2024-12-25",  # a date
        "555-1234",  # a phone number
        "what is 10**400 / 3",  # parses, but overflows: the LLM router decides
    ],
)
def test_fast_path_defers_everything_else_to_the_llm(query: str) -> None:
    assert fast_route(query) is None


def test_router_prompt_carries_the_query() -> None:
    messages = ROUTER_PROMPT.invoke({"query": "How tall is Everest?"}).to_messages()
    assert isinstance(messages[0], SystemMessage)
    assert "math" in messages[0].content and "general" in messages[0].content
    assert messages[-1].content == "How tall is Everest?"


def test_route_schema_is_strict_mode_compatible() -> None:
    # OpenAI strict JSON schema needs every property listed as required.
    schema = RouteDecision.model_json_schema()
    assert set(schema["required"]) == {"route", "expression"}
    assert schema["properties"]["route"]["enum"] == ["math", "general"]


def test_tool_context_tells_the_llm_not_to_recompute() -> None:
    assert "= 42" in tool_context("6*7", "42")
    assert "do not recompute" in tool_context("6*7", "42")
    assert "No calculator result" in tool_context(None, None)


async def test_answer_branch_dispatches_and_still_streams() -> None:
    from app.chains import build_general_chain, build_math_chain

    general = ScriptedChatModel(reply="general answer here")
    math = ScriptedChatModel(reply="math answer here")
    branch = build_answer_branch(build_general_chain(general), build_math_chain(math))
    assert isinstance(branch.bound if hasattr(branch, "bound") else branch, RunnableBranch)

    chunks = [c async for c in branch.astream({"route": "math", "query": "q", "tool_context": ""})]
    assert "".join(chunks) == "math answer here" and len(chunks) > 1
    chunks = [c async for c in branch.astream({"route": "general", "query": "q"})]
    assert "".join(chunks) == "general answer here" and len(chunks) > 1


def test_provider_inference() -> None:
    assert provider_of("openai:gpt-5.6-luna") == "openai"
    assert provider_of("anthropic:claude-haiku-4-5") == "anthropic"
    assert provider_of("gpt-5.6-terra") == "openai"
    assert provider_of("claude-sonnet-5-5") == "anthropic"
    assert provider_of("some-local-model") is None
