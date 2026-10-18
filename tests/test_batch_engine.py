"""Batch lifecycle tests use a CPU stub with explicit per-sequence state."""
import asyncio
import threading
import time

from aiohttp.test_utils import TestClient, TestServer
import pytest

from tensorpilot.backend import Token
from tensorpilot.batch_engine import BatchEngine
from tensorpilot.client import Client
from tensorpilot.metrics import Metrics
from tensorpilot.server import create_app
from test_serving import FakeBackend, terminal


class Adapter:
    def __init__(self, backend, *, capacity, prefill_step_size):
        self.owner = threading.get_ident()
        self.pending = {}
        self.next_uid = 0
        self.capacity = capacity
        self.closed = False

    def owner_only(self):
        assert threading.get_ident() == self.owner

    def insert(self, prompt, budget):
        self.owner_only()
        if prompt == "invalid":
            raise ValueError("injected invalid prompt")
        assert len(self.pending) < self.capacity
        uid, self.next_uid = self.next_uid, self.next_uid + 1
        self.pending[uid] = [budget, 0]
        return uid

    def step(self):
        self.owner_only()
        time.sleep(.002)
        result = []
        for uid, (budget, generated) in list(self.pending.items()):
            generated += 1
            finish = "length" if generated == budget else None
            result.append((uid, Token("x", generated, True, 3, finish)))
            if finish:
                del self.pending[uid]
            else:
                self.pending[uid][1] = generated
        return result

    def remove(self, uid):
        self.owner_only()
        return self.pending.pop(uid, None) is not None

    def close(self):
        self.owner_only()
        self.pending.clear()
        self.closed = True


def engine(**kwargs):
    return BatchEngine(FakeBackend, adapter_factory=Adapter, **kwargs)


def test_two_sequences_share_steps_and_finish_independently():
    async def scenario():
        service = engine()
        await service.start(warmup=False)
        try:
            first, second = service.submit("a", 2, 5), service.submit("b", 4, 5)
            done_a, done_b = await asyncio.gather(terminal(first), terminal(second))
            assert done_a["generated_tokens"] == 2
            assert done_b["generated_tokens"] == 4
            assert done_a["max_decode_batch_width"] == done_b["max_decode_batch_width"] == 2
            assert done_a["execution_mode"] == "batch"
            await asyncio.wait_for(service.queue.join(), 1)
            assert service.active_count == 0
        finally:
            await service.close()
    asyncio.run(scenario())


def test_slow_sequence_does_not_block_healthy_peer():
    async def scenario():
        metrics = Metrics()
        service = engine(metrics=metrics)
        await service.start(warmup=False)
        try:
            slow, healthy = service.submit("unread", 64, 5), service.submit("reader", 32, 5)
            assert (await terminal(healthy))["generated_tokens"] == 32
            assert (await terminal(slow))["code"] == "slow_consumer"
            await asyncio.wait_for(service.queue.join(), 1)
            assert service.ready
            assert metrics.outcomes.labels(outcome="failed")._value.get() == 1
        finally:
            await service.close()
    asyncio.run(scenario())


def test_cancellation_and_deadline_are_per_sequence():
    async def scenario():
        service = engine()
        await service.start(warmup=False)
        try:
            cancelled = service.submit("cancel", 100, 5)
            peer = service.submit("peer", 5, 5)
            await asyncio.wait_for(cancelled.events.get(), 1)
            service.cancel(cancelled)
            assert (await terminal(cancelled))["code"] == "cancelled"
            assert (await terminal(peer))["generated_tokens"] == 5
            expired, peer = service.submit("expire", 100, .01), service.submit("peer", 10, 5)
            assert (await terminal(expired))["code"] == "deadline_exceeded"
            assert (await terminal(peer))["generated_tokens"] == 10
            assert service.ready
        finally:
            await service.close()
    asyncio.run(scenario())


def test_invalid_prompt_does_not_poison_batch_or_queue_accounting():
    async def scenario():
        service = engine()
        await service.start(warmup=False)
        try:
            invalid, valid = service.submit("invalid", 1, 5), service.submit("valid", 1, 5)
            assert (await terminal(invalid))["code"] == "invalid_prompt"
            assert (await terminal(valid))["type"] == "done"
            await asyncio.wait_for(service.queue.join(), 1)
            assert service.ready
        finally:
            await service.close()
    asyncio.run(scenario())


def test_shutdown_finishes_all_active_and_waiting_jobs_once():
    async def scenario():
        service = engine(max_batch_size=2)
        await service.start(warmup=False)
        jobs = [service.submit(str(index), 100, 5) for index in range(4)]
        await asyncio.wait_for(jobs[0].events.get(), 1)
        await asyncio.wait_for(service.close(), 2)
        for job in jobs:
            assert (await terminal(job))["type"] == "error"
        await asyncio.wait_for(service.queue.join(), 1)
        assert service.adapter.closed
    asyncio.run(scenario())


def test_failed_shared_step_degrades_all_sequences():
    class Broken(Adapter):
        def step(self):
            raise RuntimeError("injected shared failure")

    async def scenario():
        service = BatchEngine(FakeBackend, adapter_factory=Broken)
        await service.start(warmup=False)
        jobs = [service.submit(str(index), 2, 5) for index in range(3)]
        for job in jobs:
            assert (await terminal(job))["code"] == "backend_unavailable"
        assert not service.ready
        await service.close()
        await asyncio.wait_for(service.queue.join(), 1)
    asyncio.run(scenario())


def test_http_batch_mode_uses_same_streaming_contract():
    async def scenario():
        app = create_app(FakeBackend, execution="batch", batch_adapter_factory=Adapter, warmup=False)
        async with TestClient(TestServer(app)) as http:
            async with Client(str(http.make_url("/"))) as client:
                async def generate():
                    return [event async for event in client.stream("hello", max_tokens=3)]
                results = await asyncio.gather(generate(), generate())
                assert all(events[-1]["generated_tokens"] == 3 for events in results)
                assert any(events[-1]["max_decode_batch_width"] == 2 for events in results)
            async with http.get("/health") as response:
                body = await response.json()
                assert body["execution_mode"] == "batch"
                assert body["execution_concurrency"] == 2
    asyncio.run(scenario())


def test_new_sequence_joins_an_existing_decode():
    async def scenario():
        service = engine()
        await service.start(warmup=False)
        try:
            first = service.submit("first", 12, 5)
            await asyncio.wait_for(first.events.get(), 1)
            later = service.submit("later", 3, 5)
            early_done, late_done = await asyncio.gather(terminal(first), terminal(later))
            assert early_done["generated_tokens"] == 12
            assert late_done["generated_tokens"] == 3
            assert late_done["max_decode_batch_width"] == 2
            await asyncio.wait_for(service.queue.join(), 1)
        finally:
            await service.close()
    asyncio.run(scenario())


def test_remove_failure_degrades_shared_context():
    class Broken(Adapter):
        def remove(self, uid):
            raise RuntimeError("injected remove failure")

    async def scenario():
        service = BatchEngine(FakeBackend, adapter_factory=Broken)
        await service.start(warmup=False)
        try:
            first, peer = service.submit("first", 50, 5), service.submit("peer", 50, 5)
            await asyncio.wait_for(first.events.get(), 1)
            service.cancel(first)
            for job in (first, peer):
                assert (await terminal(job))["code"] == "backend_unavailable"
            assert not service.ready
        finally:
            await service.close()
        await asyncio.wait_for(service.queue.join(), 1)
    asyncio.run(scenario())


def test_watchdog_fails_all_active_sequences():
    class Stalled(Adapter):
        def step(self):
            time.sleep(.08)
            return super().step()

    async def scenario():
        service = BatchEngine(FakeBackend, adapter_factory=Stalled, stall_timeout_s=.02)
        await service.start(warmup=False)
        try:
            jobs = [service.submit(str(index), 2, 5) for index in range(2)]
            for job in jobs:
                assert (await terminal(job))["code"] == "backend_unavailable"
            assert not service.ready
        finally:
            await service.close()
        await asyncio.wait_for(service.queue.join(), 1)
    asyncio.run(scenario())


@pytest.mark.parametrize("options", [{"max_batch_size": 0}, {"max_batch_size": 9}, {"prefill_step_size": 0}])
def test_invalid_batch_configuration(options):
    with pytest.raises(ValueError):
        engine(**options)
