"""CUDA continuous batching behind the real server app: concurrency, cancellation, deadlines, overload,
failure handling and accounting, over loopback HTTP on CPU with a tiny model. No GPU claim."""
import asyncio
import json
import random

import aiohttp
import pytest
from aiohttp.test_utils import TestServer

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from tiny_model import make_tiny_model_dir  # noqa: E402
from tensorpilot.cuda_backend import CUDABackend  # noqa: E402
from tensorpilot.cuda_batch import CUDABatchAdapterFactory  # noqa: E402
from tensorpilot.server import ENGINE, create_app  # noqa: E402

MiB = 1024**2
WORDS = [f"w{i}" for i in range(60)] + ["hello", "world", "the", "a"]
KEY = "batch-test-key"


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    return make_tiny_model_dir(tmp_path_factory.mktemp("batch-engine-model"))


def prompt_of(length, seed):
    rng = random.Random(seed)
    return " ".join(rng.choice(WORDS) for _ in range(length))


def tiny_backend(model_dir, mode="none"):
    """A random tiny model can emit its configured EOS id by chance, depending on the host CPU's numerics; take it out of play."""
    backend = CUDABackend(str(model_dir), device="cpu", allow_cpu=True, dtype="float32", quantization=mode)
    backend.eos_token_ids = frozenset()
    return backend


def reference(model_dir, prompt, max_tokens, mode="none"):
    backend = tiny_backend(model_dir, mode)
    return [t.token_id for t in backend.stream(prompt, max_tokens) if t.output]


class Server:
    def __init__(self, model_dir, *, batch_size=4, queue_size=16, hook=None, mode="none", **options):
        def factory():
            return tiny_backend(model_dir, mode)

        self.app = create_app(factory, execution="batch", batch_size=batch_size, queue_size=queue_size,
                              prefill_step_size=8, warmup=False, api_key=KEY,
                              batch_adapter_factory=CUDABatchAdapterFactory(kv_budget_bytes=64 * MiB),
                              max_queue_wait_s=1.0, **options)
        self.server = TestServer(self.app)
        self.hook = hook

    async def __aenter__(self):
        await self.server.start_server()
        self.base = str(self.server.make_url("")).rstrip("/")
        self.session = aiohttp.ClientSession(headers={"Authorization": f"Bearer {KEY}"})
        if self.hook:
            self.hook(self.app[ENGINE])
        return self

    async def __aexit__(self, *exc):
        await self.session.close()
        await self.server.close()

    async def generate(self, prompt, max_tokens, *, timeout_s=30, stop_after=None):
        """Returns (status, events). With stop_after, drops the connection after that many token events."""
        body = {"prompt": prompt, "max_tokens": max_tokens, "timeout_s": timeout_s}
        events = []
        async with self.session.post(self.base + "/v1/generate", json=body) as response:
            if response.status != 200:
                return response.status, [await response.json()]
            async for line in response.content:
                event = json.loads(line)
                events.append(event)
                if stop_after is not None and sum(e["type"] == "token" for e in events) >= stop_after:
                    break
                if event["type"] in ("done", "error"):
                    break
        return response.status, events

    async def health(self):
        async with self.session.get(self.base + "/health") as response:
            return await response.json()

    async def settle(self, *, rows=0, timeout=10):
        deadline = asyncio.get_running_loop().time() + timeout
        while asyncio.get_running_loop().time() < deadline:
            health = await self.health()
            if (health["kv_pool"]["decode_rows_active"] == rows and health["active_sequences"] == rows
                    and health["queue_depth"] == 0):
                return health
            await asyncio.sleep(.02)
        raise AssertionError(f"batch did not settle: {await self.health()}")


def tokens_of(events):
    return [e["token_id"] for e in events if e["type"] == "token" and e["output"]]


def terminal(events):
    assert events and events[-1]["type"] in ("done", "error"), "stream ended without a terminal event"
    assert sum(e["type"] in ("done", "error") for e in events) == 1, "exactly one terminal event"
    return events[-1]


@pytest.mark.parametrize("mode", ["none", "nf4"])
def test_concurrent_streams_are_correct_and_actually_batched(model_dir, mode):
    plan = [(prompt_of(n, n), mt) for n, mt in [(3, 20), (9, 16), (14, 12), (5, 24), (20, 10), (7, 18),
                                                 (2, 22), (11, 14), (6, 20), (16, 8), (4, 26), (8, 12)]]

    async def scenario():
        async with Server(model_dir, batch_size=4, mode=mode) as server:
            results = await asyncio.gather(*(server.generate(p, mt) for p, mt in plan))
            health = await server.settle()
            metrics = await (await server.session.get(server.base + "/metrics")).text()
        return results, health, metrics

    results, health, metrics = asyncio.run(scenario())
    for (prompt, max_tokens), (status, events) in zip(plan, results):
        assert status == 200
        done = terminal(events)
        assert done["type"] == "done" and done["execution_mode"] == "batch"
        assert tokens_of(events) == reference(model_dir, prompt, max_tokens, mode)
    widths = [terminal(events)["max_decode_batch_width"] for _, events in results]
    assert max(widths) > 1, "requests must share decode steps, otherwise this is serial execution"
    assert health["kv_pool"]["kind"] == "fixed_slot_kv_pool_not_paged"
    # Evidence must say which precision served the requests; a mislabeled run would corrupt a comparison.
    if mode == "nf4":
        assert health["quantization"]["format"] == "nf4" and health["quantization"]["quantized_linear_layers"] > 0
    else:
        assert health["quantization"] is None
    assert health["weights_resident_bytes"] == health["kv_pool"]["weights_bytes"]
    assert health["execution_concurrency"] == 4 and health["kv_pool"]["capacity_rows"] == 4
    assert health["admission"]["reservations"] == 0
    assert "tensorpilot_kv_pool_rows_capacity 4.0" in metrics
    assert health["runtime_profile"]["decode_steps"] > 0 and health["runtime_profile"]["prefill_steps"] > 0


@pytest.mark.parametrize("mode", ["none", "nf4"])
def test_a_client_disconnect_mid_decode_frees_its_slot_and_spares_the_others(model_dir, mode):
    long = (prompt_of(6, 1), 60)
    others = [(prompt_of(5, 2), 14), (prompt_of(9, 3), 18)]

    async def scenario():
        async with Server(model_dir, batch_size=4, mode=mode) as server:
            dropped = asyncio.create_task(server.generate(*long, stop_after=4))
            rest = [asyncio.create_task(server.generate(p, mt)) for p, mt in others]
            await dropped
            results = await asyncio.gather(*rest)
            health = await server.settle()
            outcomes = await (await server.session.get(server.base + "/metrics")).text()
        return results, health, outcomes

    results, health, outcomes = asyncio.run(scenario())
    for (prompt, max_tokens), (status, events) in zip(others, results):
        assert tokens_of(events) == reference(model_dir, prompt, max_tokens, mode)
    assert health["admission"]["reservations"] == 0 and health["admission"]["estimated_kv_reserved_bytes"] == 0
    assert 'tensorpilot_jobs_total{outcome="cancelled"} 1.0' in outcomes
    assert health["kv_pool"]["sequences_removed"] >= 1


def test_a_request_that_misses_its_deadline_never_completes_and_its_slot_is_freed(model_dir):
    """The HTTP handler's own timeout can close the stream before the engine's error event is read; either
    way the engine must record the timeout, free the slot and release the reservation."""
    async def scenario():
        async with Server(model_dir, batch_size=2) as server:
            neighbour = asyncio.create_task(server.generate(prompt_of(5, 2), 12))
            status, events = await server.generate(prompt_of(4, 1), 250, timeout_s=0.05)
            other = await neighbour
            health = await server.settle()
            metrics = await (await server.session.get(server.base + "/metrics")).text()
        return events, other, health, metrics

    events, other, health, metrics = asyncio.run(scenario())
    assert not any(event["type"] == "done" for event in events)
    if events and events[-1]["type"] == "error":
        assert events[-1]["code"] == "deadline_exceeded"
    assert len(tokens_of(events)) < 250
    assert terminal(other[1])["type"] == "done", "a neighbour must not be harmed by another request's deadline"
    assert 'tensorpilot_jobs_total{outcome="timed_out"} 1.0' in metrics
    assert health["admission"]["reservations"] == 0


def test_overload_rejects_at_admission_and_every_accepted_request_still_completes(model_dir):
    async def scenario():
        async with Server(model_dir, batch_size=2, queue_size=2) as server:
            results = await asyncio.gather(*(server.generate(prompt_of(5, i), 10) for i in range(14)))
            health = await server.settle()
        return results, health

    results, health = asyncio.run(scenario())
    statuses = [status for status, _ in results]
    assert 503 in statuses and 200 in statuses, statuses
    for status, events in results:
        if status == 200:
            assert terminal(events)["type"] == "done"
        else:
            assert events[0]["error"]["code"] == "overloaded"
    assert health["admission"]["reservations"] == 0


def test_a_step_failure_degrades_the_worker_and_every_request_gets_one_terminal_error(model_dir):
    state = {"calls": 0}

    def arm(engine):
        original = engine.adapter._forward

        def failing(*args, **kwargs):
            state["calls"] += 1
            if state["calls"] == 6:
                raise RuntimeError("injected GPU fault")
            return original(*args, **kwargs)

        engine.adapter._forward = failing

    async def scenario():
        async with Server(model_dir, batch_size=4, hook=arm) as server:
            results = await asyncio.gather(*(server.generate(prompt_of(5, i), 40) for i in range(6)))
            async with server.session.get(server.base + "/readyz") as response:
                ready = response.status
            late_status, late = await server.generate("hello world", 4)
            health = await server.health()
        return results, ready, late_status, late, health

    results, ready, late_status, late, health = asyncio.run(scenario())
    failed = 0
    for status, events in results:
        last = terminal(events)
        if last["type"] == "error":
            failed += 1
            assert last["code"] == "backend_unavailable"
    assert failed >= 1
    assert ready == 503 and health["ready"] is False and health["unhealthy_reason"] == "batch_step_failed"
    assert late_status == 503 and late[0]["error"]["code"] in ("unavailable", "overloaded")


def test_authentication_still_protects_the_batch_worker(model_dir):
    async def scenario():
        async with Server(model_dir) as server:
            async with aiohttp.ClientSession() as anonymous:
                async with anonymous.post(server.base + "/v1/generate", json={"prompt": "hi"}) as response:
                    return response.status, (await response.json())["error"]["code"]

    assert asyncio.run(scenario()) == (401, "unauthorized")


@pytest.mark.parametrize("mode", ["none", "nf4"])
def test_sustained_churn_returns_the_pool_to_idle_without_reservation_leaks(model_dir, mode):
    rng = random.Random(11)

    async def scenario():
        async with Server(model_dir, batch_size=4, queue_size=64, mode=mode) as server:
            jobs = []
            for i in range(40):
                stop = 3 if i % 7 == 0 else None
                jobs.append(asyncio.create_task(server.generate(prompt_of(rng.randrange(1, 18), i),
                                                                rng.randrange(1, 20), stop_after=stop)))
                await asyncio.sleep(rng.random() * 0.01)
            results = await asyncio.gather(*jobs)
            return results, await server.settle()

    results, health = asyncio.run(scenario())
    assert all(status == 200 for status, _ in results)
    assert health["kv_pool"]["decode_rows_active"] == 0 and health["kv_pool"]["prefill_waiting"] == 0
    assert health["admission"]["reservations"] == 0
    assert health["kv_pool"]["pool_bytes_allocated"] == 5 * health["kv_pool"]["row_tokens"] * \
        health["kv_pool"]["bytes_per_token"], "the pool must never change size"


def test_a_handler_timeout_is_a_deadline_even_when_the_timer_fires_before_the_jobs_own_deadline(model_dir, monkeypatch):
    """Timer granularity can fire the handler's timeout a fraction of a millisecond before the job's deadline.
    The handler knows why it is cancelling, so the outcome must not depend on that clock comparison."""
    real_timeout = asyncio.timeout
    monkeypatch.setattr(asyncio, "timeout", lambda delay: real_timeout(None if delay is None else delay * 0.2))

    def slow_down(engine):
        original = engine.adapter._forward

        def slow(*args, **kwargs):
            import time
            time.sleep(0.01)  # the request must outlast the shortened handler timeout
            return original(*args, **kwargs)

        engine.adapter._forward = slow

    async def scenario():
        async with Server(model_dir, batch_size=2, hook=slow_down) as server:
            status, events = await server.generate(prompt_of(4, 1), 250, timeout_s=0.5)
            health = await server.settle()
            metrics = await (await server.session.get(server.base + "/metrics")).text()
        return events, health, metrics

    events, health, metrics = asyncio.run(scenario())
    assert not any(event["type"] == "done" for event in events)
    assert 'tensorpilot_jobs_total{outcome="timed_out"} 1.0' in metrics
    assert 'outcome="cancelled"' not in metrics
    assert health["admission"]["reservations"] == 0


def test_a_client_disconnect_is_still_recorded_as_a_cancellation_not_a_timeout(model_dir):
    async def scenario():
        async with Server(model_dir, batch_size=2) as server:
            await server.generate(prompt_of(4, 1), 200, stop_after=2)
            await server.settle()
            return await (await server.session.get(server.base + "/metrics")).text()

    metrics = asyncio.run(scenario())
    assert 'tensorpilot_jobs_total{outcome="cancelled"} 1.0' in metrics and 'outcome="timed_out"' not in metrics
