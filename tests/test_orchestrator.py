import asyncio

import pytest

from tests.fakes import ScriptedChatModel, collect, eventually, make_router, text_of


def names(events):
    return [e.name for e in events]


async def test_general_route_streams_tokens_in_order(build) -> None:
    orch, general, math = build(router=make_router("general"))
    events = await collect(orch.stream("What is the capital of France?"))

    assert names(events)[0] == "route"
    assert names(events)[-1] == "done"
    assert set(names(events)[1:-1]) == {"token"}
    assert len([e for e in events if e.name == "token"]) > 1  # really chunked
    assert text_of(events) == "Paris is the capital of France."
    assert events[0].data["route"] == "general"
    assert events[-1].data["speculation"] == "hit"
    assert math.prompts == []


async def test_pure_arithmetic_skips_the_router_llm(build) -> None:
    router = make_router("general")
    orch, general, math = build(router=router)
    events = await collect(orch.stream("what is 6 * 7?"))

    assert router.calls == []  # no LLM round trip for routing
    assert general.prompts == []  # no speculative draft either
    route, tool = events[0], events[1]
    assert route.data == {"route": "math", "source": "rules", "expression": "6 * 7", "ms": route.data["ms"]}
    assert tool.name == "tool"
    assert tool.data == {"name": "calculator", "expression": "6 * 7", "result": "42"}
    # The math LLM explains the exact result instead of computing it.
    assert "`6 * 7` = 42" in math.prompts[0]
    assert text_of(events) == "The answer is 42."


async def test_llm_routed_math_cancels_the_speculative_draft(build) -> None:
    general = ScriptedChatModel(token_delay=0.05, reply="word " * 50)
    orch, general, math = build(
        router=make_router("math", expression="0.15 * 80", delay=0.05), general=general
    )
    events = await collect(orch.stream("What is 15% of 80?"))

    assert events[0].data["source"] == "llm"
    assert events[1].data["result"] == "12"
    assert events[-1].data["speculation"] == "cancelled"
    assert await eventually(lambda: general.cancelled == 1)
    assert general.finished == 0
    assert text_of(events) == "The answer is 42."


async def test_math_answer_does_not_wait_for_a_stalled_draft(build) -> None:
    # The draft has not produced a token yet when the router says math.
    # Its shutdown (up to the grace period) must not delay the math answer.
    general = ScriptedChatModel(first_token_delay=5)
    orch, general, math = build(router=make_router("math", expression="6*7", delay=0.05), general=general)
    events = await collect(orch.stream("six times seven, please"))
    first_token = next(e for e in events if e.name == "token")
    assert events[-1].data["first_token_ms"] < 300
    assert first_token.data["text"] == "The"
    assert await eventually(lambda: general.cancelled == 1)


async def test_conceptual_math_uses_math_chain_without_tool(build) -> None:
    orch, general, math = build(router=make_router("math", expression=None))
    events = await collect(orch.stream("Why is the derivative of x^2 equal to 2x?"))

    assert "tool" not in names(events)
    assert "No calculator result" in math.prompts[0]


async def test_bad_expression_from_llm_is_reported_not_executed(build) -> None:
    orch, general, math = build(router=make_router("math", expression="__import__('os').system('id')"))
    events = await collect(orch.stream("do some math"))

    tool = next(e for e in events if e.name == "tool")
    assert "error" in tool.data and "result" not in tool.data
    assert "No calculator result" in math.prompts[0]
    assert names(events)[-1] == "done"


async def test_router_failure_degrades_to_general(build) -> None:
    orch, general, math = build(router=make_router("math", fail=True))
    events = await collect(orch.stream("Tell me a joke"))

    assert events[0].data["route"] == "general"
    assert events[0].data["source"] == "fallback"
    assert text_of(events) == general.reply


async def test_model_failure_becomes_a_generic_error_event(build) -> None:
    secret_looking = "401 invalid key sk-proj-abc123 for org-xyz"
    general = ScriptedChatModel(error=secret_looking)
    orch, *_ = build(general=general)
    events = await collect(orch.stream("hello there"))

    assert names(events) == ["route", "error"]
    assert events[-1].data["message"] == "The language model request failed."
    assert "sk-proj" not in str(events[-1].data)


async def test_stalled_model_times_out(build) -> None:
    general = ScriptedChatModel(first_token_delay=5)
    orch, *_ = build(general=general, idle_timeout=0.2)
    events = await collect(orch.stream("hello there"))

    assert events[-1].name == "error"
    assert events[-1].data["message"] == "The model stopped responding."
    assert await eventually(lambda: general.cancelled == 1)


async def test_speculative_routing_hides_router_latency(build) -> None:
    # Router takes 300 ms, the answer model needs 300 ms to its first token.
    async def ttft(speculative: bool) -> int:
        orch, *_ = build(
            router=make_router("general", delay=0.3),
            general=ScriptedChatModel(first_token_delay=0.3),
            speculative=speculative,
        )
        events = await collect(orch.stream("Tell me about Waterloo"))
        return events[-1].data["first_token_ms"]

    sequential, speculative = await ttft(False), await ttft(True)
    assert sequential >= 580  # router + model
    assert speculative < 450  # max(router, model)


async def test_client_disconnect_cancels_the_model_stream(build) -> None:
    general = ScriptedChatModel(token_delay=0.05, reply="word " * 100)
    orch, general, _ = build(general=general)

    stream = orch.stream("long answer please")
    async for event in stream:
        if event.name == "token":
            break
    await stream.aclose()  # what happens when the SSE consumer goes away

    assert await eventually(lambda: general.cancelled == 1)
    assert general.finished == 0


@pytest.mark.parametrize("speculative", [True, False])
async def test_concurrent_requests_do_not_interfere(build, speculative: bool) -> None:
    orch, general, math = build(
        router=make_router("general", delay=0.05),
        general=ScriptedChatModel(token_delay=0.01),
        speculative=speculative,
    )
    results = await asyncio.gather(*(collect(orch.stream(f"question {i}")) for i in range(20)))
    assert all(text_of(events) == general.reply for events in results)
    assert general.finished == 20
