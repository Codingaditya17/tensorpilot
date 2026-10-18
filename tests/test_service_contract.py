import asyncio
import json
import threading

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from tensorpilot.client import Client, TensorPilotError
from tensorpilot.server import ENGINE, create_app
from test_serving import FakeBackend


def test_auth_probes_and_request_ids():
    async def scenario():
        async with TestClient(TestServer(create_app(FakeBackend, api_key="test-secret", warmup=False))) as http:
            for path in ("/livez", "/readyz"):
                async with http.get(path) as response:
                    assert response.status == 200
            for path in ("/metrics", "/v1/model", "/health"):
                async with http.get(path) as response:
                    assert response.status == 401
                    body = await response.json()
                    assert body["error"]["code"] == "unauthorized"
                    assert body["request_id"] == response.headers["X-Request-ID"]
            async with http.get("/v1/model", headers={"Authorization": "Bearer test-secret"}) as response:
                assert response.status == 200
                assert (await response.json())["execution_concurrency"] == 1
            async with Client(str(http.make_url("/")), api_key="test-secret") as client:
                events = [event async for event in client.stream("hello", max_tokens=2)]
                ids = {event["request_id"] for event in events}
                assert len(ids) == 1
                assert len(next(iter(ids))) == 32
                assert events[-1]["type"] == "done"
    asyncio.run(scenario())


def test_validation_errors_and_sdk_errors_are_structured():
    async def scenario():
        async with TestClient(TestServer(create_app(FakeBackend, warmup=False))) as http:
            for body in ([], {}, {"prompt": "hi", "unsupported": True}, {"prompt": "hi", "timeout_s": -1}):
                async with http.post("/v1/generate", json=body) as response:
                    assert response.status == 400
                    assert (await response.json())["error"]["code"] == "invalid_request"
            async with http.post("/v1/generate", data="{" , headers={"Content-Type": "application/json"}) as response:
                assert response.status == 400
            async with http.get("/missing") as response:
                assert response.status == 404
                assert (await response.json())["error"]["code"] == "not_found"
            async with Client(str(http.make_url("/"))) as client:
                with pytest.raises(TensorPilotError) as failure:
                    _ = [event async for event in client.stream("", max_tokens=1)]
                assert failure.value.status == 400
                assert failure.value.code == "invalid_request"
                assert failure.value.request_id
                with pytest.raises(TensorPilotError) as failure:
                    _ = [event async for event in client.stream("fail", max_tokens=1)]
                assert failure.value.code == "invalid_prompt"
                assert "injected failure" not in str(failure.value)
    asyncio.run(scenario())


def test_worker_metrics_and_readiness():
    async def scenario():
        app = create_app(FakeBackend, warmup=False)
        async with TestClient(TestServer(app)) as http:
            async with Client(str(http.make_url("/"))) as client:
                _ = [event async for event in client.stream("hello", max_tokens=3)]
            await asyncio.wait_for(app[ENGINE].queue.join(), 2)
            async with http.get("/metrics") as response:
                assert response.status == 200
                text = await response.text()
                assert 'tensorpilot_jobs_total{outcome="completed"} 1.0' in text
                assert "tensorpilot_output_tokens_total 3.0" in text
                assert "tensorpilot_queue_depth 0.0" in text
                assert "tensorpilot_active_requests 0.0" in text
                assert "tensorpilot_ttft_seconds_count 1.0" in text
                assert "hello" not in text
            app[ENGINE].closing = True
            async with http.get("/readyz") as response:
                assert response.status == 503
            async with http.get("/livez") as response:
                assert response.status == 200
            async with http.post("/v1/generate", json={"prompt": "hi"}) as response:
                assert response.status == 503
                assert (await response.json())["error"]["code"] == "unavailable"
    asyncio.run(scenario())


def test_http_overload_has_retry_after_and_request_id():
    class GatedBackend(FakeBackend):
        def stream(self, prompt, max_tokens):
            iterator = super().stream(prompt, max_tokens)
            yield next(iterator)
            gate.wait(3)
            yield from iterator

    gate = threading.Event()

    async def scenario():
        app = create_app(GatedBackend, queue_size=1, warmup=False)
        async with TestClient(TestServer(app)) as http:
            try:
                first = await http.post("/v1/generate", json={"prompt": "active", "max_tokens": 2})
                await asyncio.wait_for(first.content.readline(), 2)
                second = await http.post("/v1/generate", json={"prompt": "waiting", "max_tokens": 1})
                async with http.post("/v1/generate", json={"prompt": "overflow"}) as response:
                    assert response.status == 503
                    assert response.headers["Retry-After"] == "1"
                    body = await response.json()
                    assert body["error"]["code"] == "overloaded"
                    assert body["request_id"] == response.headers["X-Request-ID"]
            finally:
                gate.set()
            await first.read()
            await second.read()
            async with http.get("/metrics") as response:
                assert "tensorpilot_rejected_requests_total 1.0" in await response.text()
    asyncio.run(scenario())


def test_sdk_detects_truncated_stream():
    async def scenario():
        async def truncated(request):
            return web.Response(text=json.dumps({"type": "token", "text": "x", "token_id": 1, "output": True}) + "\n")
        app = web.Application()
        app.router.add_post("/v1/generate", truncated)
        async with TestClient(TestServer(app)) as http:
            async with Client(str(http.make_url("/"))) as client:
                with pytest.raises(TensorPilotError) as failure:
                    _ = [event async for event in client.stream("hi")]
                assert failure.value.code == "incomplete_stream"
    asyncio.run(scenario())


def test_engine_shutdown_is_idempotent():
    async def scenario():
        from tensorpilot.engine import Engine
        engine = Engine(FakeBackend)
        await engine.start(warmup=False)
        await asyncio.wait_for(asyncio.gather(engine.close(), engine.close()), 2)
    asyncio.run(scenario())
