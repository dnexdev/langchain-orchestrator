"""HTTP level tests: SSE framing, validation, auth, and real chunked streaming."""

from __future__ import annotations

import json
import socket
import threading
import time
from collections.abc import Iterator

import httpx
import pytest
import uvicorn
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from tests.fakes import ScriptedChatModel, make_router


def parse_sse(lines: Iterator[str]) -> Iterator[tuple[str, dict]]:
    """Minimal SSE parser: yields (event, data) for each blank-line terminated block."""
    event, data = "message", []
    for line in lines:
        if line == "":
            if data:
                yield event, json.loads("\n".join(data))
            event, data = "message", []
        elif line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].strip())


@pytest.fixture
def client(build):
    orch, *_ = build(router=make_router("general"))
    with TestClient(create_app(settings=Settings(), orchestrator=orch)) as c:
        yield c


def test_ask_returns_event_stream(client: TestClient) -> None:
    with client.stream("POST", "/ask", json={"query": "What is the capital of France?"}) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        assert r.headers["cache-control"] == "no-cache"
        assert r.headers["x-accel-buffering"] == "no"  # nginx must not buffer
        assert len(r.headers["x-request-id"]) == 12
        events = list(parse_sse(r.iter_lines()))

    kinds = [e for e, _ in events]
    assert kinds[0] == "route" and kinds[-1] == "done"
    assert kinds.count("token") > 1
    assert "".join(d["text"] for e, d in events if e == "token") == "Paris is the capital of France."


def test_newlines_in_tokens_do_not_break_framing(build) -> None:
    orch, *_ = build(general=ScriptedChatModel(reply="line one\n\nline two\ndata: fake"))
    with TestClient(create_app(settings=Settings(), orchestrator=orch)) as c:
        with c.stream("POST", "/ask", json={"query": "two lines please"}) as r:
            events = list(parse_sse(r.iter_lines()))
    assert "".join(d["text"] for e, d in events if e == "token") == "line one\n\nline two\ndata: fake"


def test_math_route_over_http(client: TestClient) -> None:
    with client.stream("POST", "/ask", json={"query": "2^10"}) as r:
        events = list(parse_sse(r.iter_lines()))
    assert events[0] == (
        "route",
        {"route": "math", "source": "rules", "expression": "2**10", "ms": events[0][1]["ms"]},
    )
    assert events[1] == ("tool", {"name": "calculator", "expression": "2**10", "result": "1024"})


@pytest.mark.parametrize(
    "body",
    [{}, {"query": ""}, {"query": "   "}, {"query": 42}, {"query": "x" * 4001}, {"query": "hi", "extra": 1}],
)
def test_bad_payloads_are_rejected(client: TestClient, body: dict) -> None:
    assert client.post("/ask", json=body).status_code == 422


def test_health(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_optional_service_key(build) -> None:
    orch, *_ = build()
    app = create_app(settings=Settings(service_api_key="let-me-in"), orchestrator=orch)
    with TestClient(app) as c:
        assert c.post("/ask", json={"query": "hi"}).status_code == 401
        assert c.post("/ask", json={"query": "hi"}, headers={"X-API-Key": "nope"}).status_code == 401
        assert c.post("/ask", json={"query": "hi"}, headers={"X-API-Key": "let-me-in"}).status_code == 200


# ---------------------------------------------------------------- live server
# TestClient buffers, so it cannot prove that bytes leave the server before
# the answer is finished. These tests run a real uvicorn server instead.


@pytest.fixture
def live_server():
    servers: list[tuple[uvicorn.Server, threading.Thread]] = []

    def start(orchestrator) -> str:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        app = create_app(settings=Settings(), orchestrator=orchestrator)
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        deadline = time.time() + 5
        while not server.started:
            assert time.time() < deadline, "server did not start"
            time.sleep(0.01)
        servers.append((server, thread))
        return f"http://127.0.0.1:{port}"

    yield start
    for server, thread in servers:
        server.should_exit = True
        thread.join(timeout=5)


def test_tokens_arrive_incrementally_over_real_http(build, live_server) -> None:
    orch, *_ = build(general=ScriptedChatModel(token_delay=0.05, reply="one two three four five six"))
    url = live_server(orch)

    arrivals: list[tuple[str, float]] = []
    with httpx.Client(trust_env=False, timeout=10) as http:
        with http.stream("POST", f"{url}/ask", json={"query": "count to six"}) as r:
            assert r.headers["transfer-encoding"] == "chunked"
            for event, _ in parse_sse(r.iter_lines()):
                arrivals.append((event, time.perf_counter()))

    first_token = next(t for e, t in arrivals if e == "token")
    done = next(t for e, t in arrivals if e == "done")
    # 11 chunks at 50 ms each: the first token must land long before the end.
    assert done - first_token > 0.3


def test_disconnect_stops_the_upstream_model(build, live_server) -> None:
    general = ScriptedChatModel(token_delay=0.05, reply="word " * 200)
    orch, general, _ = build(general=general)
    url = live_server(orch)

    with httpx.Client(trust_env=False, timeout=10) as http:
        with http.stream("POST", f"{url}/ask", json={"query": "talk for a while"}) as r:
            for event, _ in parse_sse(r.iter_lines()):
                if event == "token":
                    break  # leaving the block closes the connection

    deadline = time.time() + 3
    while general.cancelled == 0 and time.time() < deadline:
        time.sleep(0.02)
    assert general.cancelled == 1
    assert general.finished == 0
