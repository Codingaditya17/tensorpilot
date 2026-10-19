"""Adaptive policy tests are deterministic and do not initialize a GPU."""
import asyncio

import pytest
from aiohttp.test_utils import TestClient, TestServer

from tensorpilot.adaptive import AdaptiveBatchController
from tensorpilot.batch_backend import BatchStep
from tensorpilot.batch_engine import BatchEngine
from test_batch_engine import Adapter
from test_serving import FakeBackend, terminal
from tensorpilot.server import ENGINE, create_app


def test_width_tracks_demand_and_is_bounded():
    policy = AdaptiveBatchController(8)
    assert policy.choose(queue_depth=0, active_sequences=0, oldest_wait_s=0,
                         estimated_memory_pressure=0).width == 1
    assert policy.choose(queue_depth=3, active_sequences=1, oldest_wait_s=.01,
                         estimated_memory_pressure=.2).width == 4
    assert policy.choose(queue_depth=100, active_sequences=8, oldest_wait_s=.01,
                         estimated_memory_pressure=.2).width == 8


def test_latency_target_opens_width_and_memory_pressure_closes_it():
    policy = AdaptiveBatchController(8, latency_target_s=.1,
                                     memory_soft_limit=.7, memory_hard_limit=.9)
    latency = policy.choose(queue_depth=1, active_sequences=1, oldest_wait_s=.11,
                            estimated_memory_pressure=.2)
    assert (latency.width, latency.reason) == (8, "latency-target")
    soft = policy.choose(queue_depth=7, active_sequences=1, oldest_wait_s=.2,
                         estimated_memory_pressure=.7)
    assert (soft.width, soft.reason) == (2, "memory-soft-limit")
    hard = policy.choose(queue_depth=7, active_sequences=1, oldest_wait_s=.2,
                         estimated_memory_pressure=.95)
    assert (hard.width, hard.reason) == (1, "memory-hard-limit")


def test_policy_snapshot_explains_latest_decision():
    policy = AdaptiveBatchController(4)
    policy.choose(queue_depth=2, active_sequences=1, oldest_wait_s=.01,
                  estimated_memory_pressure=.25)
    snapshot = policy.snapshot()
    assert snapshot["decision_count"] == 1
    assert snapshot["last_decision"]["width"] == 3
    assert snapshot["last_decision"]["reason"] == "demand"


@pytest.mark.parametrize("kwargs", [
    {"max_width": 0}, {"max_width": 9}, {"latency_target_s": 0},
    {"memory_soft_limit": .95, "memory_hard_limit": .9},
])
def test_invalid_policy_configuration(kwargs):
    with pytest.raises(ValueError):
        AdaptiveBatchController(**kwargs)


def test_adaptive_engine_preserves_streams_and_releases_reservations():
    async def scenario():
        controller = AdaptiveBatchController(4, latency_target_s=.001)
        service = BatchEngine(FakeBackend, max_batch_size=4, adapter_factory=Adapter,
                              adaptive_controller=controller)
        await service.start(warmup=False)
        try:
            jobs = [service.submit(str(index), 4, 5) for index in range(4)]
            results = await asyncio.gather(*(terminal(job) for job in jobs))
            assert all(result["generated_tokens"] == 4 for result in results)
            assert all(result["execution_mode"] == "adaptive" for result in results)
            assert max(result["max_decode_batch_width"] for result in results) == 4
            assert service.admission.reservations == 0
            assert controller.decision_count > 0
        finally:
            await service.close()
    asyncio.run(scenario())


def test_batch_terminal_includes_phase_and_estimate_provenance():
    async def scenario():
        service = BatchEngine(FakeBackend, adapter_factory=Adapter)
        await service.start(warmup=False)
        try:
            result = await terminal(service.submit("profile", 2, 5))
            assert result["estimated_kv_bytes"] > 0
            assert result["phase_profile"]["measurement_source"] == "synchronized_model_thread_wall_clock"
            assert result["inter_token_latency_s"]
        finally:
            await service.close()
    asyncio.run(scenario())


def test_cancellation_during_prefill_isolated_from_peer():
    class PrefillAdapter(Adapter):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.prefilled = False

        def step(self):
            if not self.prefilled:
                self.prefilled = True
                return BatchStep([], "prefill", .001, len(self.pending))
            return super().step()

    async def scenario():
        service = BatchEngine(FakeBackend, max_batch_size=2, adapter_factory=PrefillAdapter)
        await service.start(warmup=False)
        try:
            cancelled = service.submit("cancel-prefill", 20, 5)
            peer = service.submit("peer", 3, 5)
            while service.profile_totals["prefill_steps"] == 0:
                await asyncio.sleep(0)
            service.cancel(cancelled)
            assert (await terminal(cancelled))["code"] == "cancelled"
            assert (await terminal(peer))["generated_tokens"] == 3
            assert service.ready
        finally:
            await service.close()
    asyncio.run(scenario())


def test_health_exposes_live_adaptive_decisions():
    async def scenario():
        app = create_app(FakeBackend, execution="adaptive", batch_size=2,
                         batch_adapter_factory=Adapter, warmup=False)
        async with TestClient(TestServer(app)) as http:
            jobs = [app[ENGINE].submit(str(index), 2, 5) for index in range(2)]
            await asyncio.gather(*(terminal(job) for job in jobs))
            async with http.get("/health") as response:
                policy = (await response.json())["adaptive_policy"]
                assert policy["decision_count"] > 0
                assert policy["last_decision"]["width"] in (1, 2)
    asyncio.run(scenario())
