"""Backend fault/cancellation tests use CPU stubs, not simulated GPU metrics."""
import asyncio
import threading

import pytest

from tensorpilot.backend import Token
from tensorpilot.engine import Engine, Overloaded
from tensorpilot.metrics import Metrics
from test_serving import FakeBackend, terminal


class GatedBackend(FakeBackend):
    def __init__(self, gate):
        super().__init__()
        self.gate = gate

    def stream(self, prompt, max_tokens):
        yield Token("first", 0, True)
        self.gate.wait(3)
        for index in range(1, max_tokens):
            yield Token("next", index, True)


def test_waiting_deadline_expires_while_model_step_is_blocked():
    gate = threading.Event()

    async def scenario():
        engine = Engine(lambda: GatedBackend(gate), queue_size=1)
        await engine.start(warmup=False)
        try:
            first = engine.submit("active", 2, 5)
            await asyncio.wait_for(first.events.get(), 1)
            expired = engine.submit("queued", 1, .02)
            failure = await terminal(expired)
            assert failure["code"] == "deadline_exceeded"
            assert expired.state == "timed_out"
            assert engine.active is first
            assert engine.queue.qsize() == 0
            replacement = engine.submit("replacement", 1, 5)
            gate.set()
            assert (await terminal(first))["type"] == "done"
            assert (await terminal(replacement))["type"] == "done"
        finally:
            gate.set()
            await engine.close()
    asyncio.run(scenario())


def test_waiting_cancellation_frees_capacity_and_is_idempotent():
    gate = threading.Event()

    async def scenario():
        metrics = Metrics()
        engine = Engine(lambda: GatedBackend(gate), queue_size=1, metrics=metrics)
        await engine.start(warmup=False)
        try:
            first = engine.submit("active", 2, 5)
            await asyncio.wait_for(first.events.get(), 1)
            waiting = engine.submit("cancel", 1, 5)
            engine.cancel(waiting)
            engine.cancel(waiting)
            assert engine.queue.empty()
            assert (await terminal(waiting))["code"] == "cancelled"
            replacement = engine.submit("replacement", 1, 5)
            gate.set()
            await terminal(first)
            await terminal(replacement)
            await asyncio.wait_for(engine.queue.join(), 1)
            assert metrics.outcomes.labels(outcome="cancelled")._value.get() == 1
        finally:
            gate.set()
            await engine.close()
    asyncio.run(scenario())


def test_active_cancellation_closes_backend_and_recovers():
    gate = threading.Event()

    async def scenario():
        engine = Engine(lambda: GatedBackend(gate))
        await engine.start(warmup=False)
        try:
            active = engine.submit("active", 100, 5)
            await asyncio.wait_for(active.events.get(), 1)
            engine.cancel(active)
            gate.set()
            assert (await terminal(active))["code"] == "cancelled"
            assert (await terminal(engine.submit("recovered", 1, 5)))["type"] == "done"
        finally:
            gate.set()
            await engine.close()
    asyncio.run(scenario())


def test_cleanup_failure_never_reports_success_and_fails_readiness():
    class BadIterator:
        def __init__(self):
            self.count = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self.count:
                raise StopIteration
            self.count += 1
            return Token("x", 0, True)

        def close(self):
            raise RuntimeError("injected cleanup failure")

    class BrokenBackend(FakeBackend):
        def stream(self, prompt, max_tokens):
            return BadIterator()

    async def scenario():
        engine = Engine(BrokenBackend)
        await engine.start(warmup=False)
        try:
            first = engine.submit("fails cleanup", 1, 5)
            second = engine.submit("waiting", 1, 5)
            assert (await terminal(first))["code"] == "backend_unavailable"
            assert (await terminal(second))["code"] == "backend_unavailable"
            assert not engine.ready
            with pytest.raises(Overloaded):
                engine.submit("rejected", 1, 5)
            await asyncio.wait_for(engine.queue.join(), 1)
        finally:
            await engine.close()
    asyncio.run(scenario())


def test_worker_supervisor_fails_pending_requests_if_worker_stops():
    async def scenario():
        engine = Engine(FakeBackend)
        await engine.start(warmup=False)
        job = engine.submit("waiting", 1, 5)
        engine.worker.cancel()
        await asyncio.gather(engine.worker, return_exceptions=True)
        assert (await terminal(job))["code"] == "backend_unavailable"
        assert not engine.ready
        await engine.close()
    asyncio.run(scenario())


def test_engine_dispatches_shortest_budget_without_preempting_model():
    class RecordingBackend(FakeBackend):
        def __init__(self):
            super().__init__()
            self.started = []

        def stream(self, prompt, max_tokens):
            self.started.append(prompt)
            yield from super().stream(prompt, max_tokens)

    async def scenario():
        engine = Engine(RecordingBackend, scheduler="shortest-budget", max_queue_wait_s=10)
        await engine.start(warmup=False)
        try:
            jobs = [engine.submit(name, budget, 5) for name, budget in (("long", 8), ("short", 1), ("medium", 3))]
            results = await asyncio.gather(*(terminal(job) for job in jobs))
            assert all(result["scheduler"] == "shortest-budget" for result in results)
            assert engine.backend.started == ["short", "medium", "long"]
        finally:
            await engine.close()
    asyncio.run(scenario())


def test_stall_watchdog_fails_active_and_waiting_jobs_before_model_returns():
    gate = threading.Event()

    async def scenario():
        metrics = Metrics()
        engine = Engine(lambda: GatedBackend(gate), metrics=metrics, stall_timeout_s=.03)
        await engine.start(warmup=False)
        try:
            active = engine.submit("blocked", 2, 5)
            await asyncio.wait_for(active.events.get(), 1)
            pending = engine.submit("waiting", 1, 5)
            assert (await terminal(active))["code"] == "backend_unavailable"
            assert (await terminal(pending))["code"] == "backend_unavailable"
            assert engine.unhealthy_reason == "worker_stalled"
            assert not engine.ready
            assert engine.active is active  # Model call is still running.
            assert metrics.watchdog_trips._value.get() == 1
            assert metrics.outcomes.labels(outcome="failed")._value.get() == 2
        finally:
            gate.set()
            await engine.close()
    asyncio.run(scenario())


def test_slow_consumer_is_not_misclassified_as_a_stalled_backend():
    async def scenario():
        engine = Engine(FakeBackend, stall_timeout_s=.03)
        await engine.start(warmup=False)
        try:
            job = engine.submit("slow reader", 100, .12)
            await asyncio.wait_for(engine.queue.join(), 1)
            assert (await terminal(job))["code"] == "deadline_exceeded"
            assert engine.ready
            assert not engine.unhealthy
        finally:
            await engine.close()
    asyncio.run(scenario())
