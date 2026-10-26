"""Backend fault/cancellation tests use CPU stubs, not simulated GPU metrics."""
import asyncio
import threading
import time

import pytest

from tensorpilot.admission import ResourceAdmission

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


def test_resource_admission_estimates_reserves_and_releases_deterministically():
    admission = ResourceAdmission(5_000, bytes_per_token=10, sequence_overhead_bytes=100)
    estimate = admission.estimate("four", 6)
    assert estimate.prompt_tokens_upper_bound == 4
    assert estimate.kv_bytes == 200
    assert admission.reserve(estimate)
    assert admission.snapshot()["estimated_kv_reserved_bytes"] == 200
    admission.release(estimate)
    assert admission.snapshot()["pressure_ratio"] == 0


def test_memory_pressure_rejection_does_not_leak_reservations():
    async def scenario():
        admission = ResourceAdmission(240, bytes_per_token=10, sequence_overhead_bytes=100)
        metrics = Metrics()
        engine = Engine(FakeBackend, admission=admission, metrics=metrics)
        await engine.start(warmup=False)
        try:
            accepted = engine.submit("abcd", 10, 5)  # 240 bytes, entire estimate budget.
            with pytest.raises(Overloaded, match="KV-cache"):
                engine.submit("x", 1, 5)
            assert admission.reservations == 1
            assert metrics.admission_decisions.labels(decision="memory_pressure")._value.get() == 1
            assert (await terminal(accepted))["type"] == "done"
            await asyncio.wait_for(engine.queue.join(), 1)
            assert admission.reservations == 0
            assert admission.reserved_bytes == 0
            assert (await terminal(engine.submit("again", 1, 5)))["type"] == "done"
        finally:
            await engine.close()
    asyncio.run(scenario())


def test_queue_full_rolls_back_memory_reservation():
    async def scenario():
        engine = Engine(FakeBackend, queue_size=1)
        await engine.start(warmup=False)
        try:
            queued = engine.submit("queued", 20, 5)
            before = engine.admission.reserved_bytes
            with pytest.raises(Overloaded, match="queue full"):
                engine.submit("overflow", 20, 5)
            assert engine.admission.reserved_bytes == before
            assert (await terminal(queued))["type"] == "done"
        finally:
            await engine.close()
    asyncio.run(scenario())


def test_an_explicit_cancel_reason_overrides_the_clock_in_both_directions():
    """The caller's reason wins whichever side of the deadline the clock says it is on."""
    from tensorpilot.engine import Job

    async def scenario():
        engine = Engine(lambda: FakeBackend(), queue_size=4, metrics=Metrics())
        now = time.perf_counter()
        before_deadline = Job("a", 4, deadline=now + 60)     # the clock says: not yet expired
        after_deadline = Job("b", 4, deadline=now - 60)      # the clock says: long expired
        engine.cancel(before_deadline, "deadline_exceeded")  # but the caller's own timeout fired
        engine.cancel(after_deadline, "cancelled")           # but the caller says the client left
        return (engine._classify(before_deadline, TimeoutError()), engine._classify(after_deadline, TimeoutError()),
                [before_deadline.cancel_code, after_deadline.cancel_code])

    early, late, codes = asyncio.run(scenario())
    assert early == ("deadline_exceeded", "timed_out") and late == ("cancelled", "cancelled")
    assert codes == ["deadline_exceeded", "cancelled"]


def test_a_queued_job_cancelled_with_a_reason_is_finished_with_that_outcome():
    async def scenario():
        gate = threading.Event()
        engine = Engine(lambda: GatedBackend(gate), queue_size=4, metrics=Metrics())
        await engine.start(warmup=False)
        try:
            blocker = engine.submit("hold the worker", 4, 30)
            await asyncio.sleep(0.05)  # the blocker occupies the model thread, so the next job stays queued
            queued = engine.submit("a", 4, 30)
            engine.cancel(queued, "deadline_exceeded")
            return queued.state, (await queued.events.get())["code"]
        finally:
            gate.set()
            engine.cancel(blocker)
            await engine.close()

    assert asyncio.run(scenario()) == ("timed_out", "deadline_exceeded")


def test_without_a_reason_the_clock_decides_and_an_invalid_reason_is_refused():
    async def scenario():
        gate = threading.Event()
        engine = Engine(lambda: GatedBackend(gate), queue_size=4, metrics=Metrics())
        await engine.start(warmup=False)
        try:
            blocker = engine.submit("hold the worker", 4, 30)
            await asyncio.sleep(0.05)
            job = engine.submit("a", 4, 30)
            with pytest.raises(ValueError, match="reason must be"):
                engine.cancel(job, "because")
            engine.cancel(job)
            return job.state
        finally:
            gate.set()
            engine.cancel(blocker)
            await engine.close()

    assert asyncio.run(scenario()) == "cancelled"
