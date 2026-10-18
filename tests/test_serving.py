"""CPU-only behavior tests. These do not measure GPU performance."""
import asyncio
import json
import threading
import time
from types import SimpleNamespace

import pytest
from aiohttp.test_utils import TestClient, TestServer

from tensorpilot.backend import Token
from tensorpilot.engine import Engine, Overloaded
from tensorpilot.load import percentile, run
from tensorpilot.server import create_app


class FakeBackend:
    def __init__(self):
        self.owner = threading.get_ident()

    def stream(self, prompt, max_tokens):
        assert threading.get_ident() == self.owner
        if prompt == "fail":
            raise ValueError("injected failure")
        for i in range(max_tokens):
            time.sleep(.002)
            yield Token("x", i, True)

    def memory(self):
        assert threading.get_ident() == self.owner
        return {"mlx_active_bytes": 0, "mlx_peak_bytes_process": 0}


async def terminal(job):
    while True:
        event = await asyncio.wait_for(job.events.get(), 5)
        if event["type"] in ("done", "error"):
            return event


def test_fifo_overload_and_failure_recovery():
    async def scenario():
        engine = Engine(FakeBackend, queue_size=2)
        await engine.start(warmup=False)
        try:
            first = engine.submit("hello", 3, 5)
            second = engine.submit("fail", 1, 5)
            with pytest.raises(Overloaded):
                engine.submit("overflow", 1, 5)
            done = await terminal(first)
            assert done["generated_tokens"] == 3
            assert len(done["token_times_s"]) == 3
            assert done["ttft_s"] >= done["queue_wait_s"] >= 0
            assert (await terminal(second))["type"] == "error"
            assert (await terminal(engine.submit("recovered", 1, 5)))["type"] == "done"
        finally:
            await engine.close()
    asyncio.run(scenario())


def test_cancel_and_queue_deadline():
    async def scenario():
        engine = Engine(FakeBackend, queue_size=3)
        await engine.start(warmup=False)
        try:
            first = engine.submit("blocking", 10, 5)
            expired = engine.submit("expire", 1, .001)
            cancelled = engine.submit("cancel", 1, 5)
            cancelled.cancelled = True
            assert (await terminal(first))["type"] == "done"
            assert (await terminal(expired))["error_type"] == "TimeoutError"
            assert (await terminal(cancelled))["error_type"] == "TimeoutError"
        finally:
            await engine.close()
    asyncio.run(scenario())


def test_slow_consumer_is_bounded_and_worker_recovers():
    async def scenario():
        engine = Engine(FakeBackend)
        await engine.start(warmup=False)
        try:
            stalled = engine.submit("slow reader", 100, .1)
            await asyncio.wait_for(engine.queue.join(), 3)
            assert stalled.events.qsize() <= 16
            assert (await terminal(stalled))["type"] == "error"
            assert (await terminal(engine.submit("next", 1, 5)))["type"] == "done"
        finally:
            await engine.close()
    asyncio.run(scenario())


def test_http_stream_validation_and_load_report():
    async def scenario():
        async with TestClient(TestServer(create_app(FakeBackend, warmup=False))) as client:
            async with client.post("/generate", json={"prompt": "hello", "max_tokens": 2}) as response:
                assert response.status == 200
                events = [json.loads(line) for line in (await response.text()).splitlines()]
                assert [e["type"] for e in events] == ["token", "token", "done"]
                assert events[-1]["generated_tokens"] == 2
            async with client.post("/generate", json={"prompt": "hello", "max_tokens": True}) as response:
                assert response.status == 400
            args = SimpleNamespace(url=str(client.make_url("/")).rstrip("/"), requests=4,
                                   concurrency=2, max_tokens=2, timeout=5, rate=0,
                                   output="unused.json")
            result = await run(args)
            assert result["summary"]["successful_requests"] == 4
            assert result["summary"]["completed_output_tokens"] == 8
            assert result["summary"]["completed_output_tokens_per_s"] == 8 / result["summary"]["wall_s"]
            assert len(result["requests"]) == 4
    asyncio.run(scenario())


def test_percentiles():
    assert percentile([], .95) is None
    assert percentile([1], .99) == 1
    assert percentile([1, 3], .5) == 2


def test_shutdown_cancels_active_and_queued_jobs():
    async def scenario():
        engine = Engine(FakeBackend)
        await engine.start(warmup=False)
        first = engine.submit("active", 1000, 5)
        await asyncio.wait_for(first.events.get(), 2)
        second = engine.submit("queued", 1, 5)
        await asyncio.wait_for(engine.close(), 2)
        assert (await terminal(first))["type"] == "error"
        assert (await terminal(second))["type"] == "error"
        with pytest.raises(Overloaded):
            engine.submit("closed", 1, 5)
    asyncio.run(scenario())
