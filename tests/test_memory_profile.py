import asyncio
from types import SimpleNamespace

from tensorpilot.backend import Token, cache_profile
from tensorpilot.engine import Engine
from tensorpilot.metrics import Metrics
from tensorpilot.profile import summarize_profiles
from test_serving import FakeBackend, terminal


def test_cache_array_byte_accounting_and_unsupported_cache():
    mx = SimpleNamespace(get_active_memory=lambda: 999)
    sample = cache_profile([SimpleNamespace(nbytes=100), SimpleNamespace(nbytes=200)], mx)
    assert sample == {"cache_allocated_bytes": 300, "cache_layers": 2, "mlx_active_bytes": 999}
    unsupported = cache_profile([object()], mx)
    assert unsupported["cache_allocated_bytes"] is None


def test_engine_preserves_prompt_counts_finish_reason_and_profile_samples():
    class ProfileBackend(FakeBackend):
        def stream(self, prompt, max_tokens):
            yield Token("x", 1, True, prompt_tokens=3, finish_reason="length",
                        memory_profile={"cache_allocated_bytes": 256, "cache_layers": 1, "mlx_active_bytes": 512})

    async def scenario():
        metrics = Metrics()
        engine = Engine(ProfileBackend, metrics=metrics)
        await engine.start(warmup=False)
        try:
            done = await terminal(engine.submit("profile", 1, 5))
            assert done["prompt_tokens"] == 3
            assert done["finish_reason"] == "length"
            assert len(done["memory_profile"]) == 1
            assert done["memory_profile"][0]["cache_allocated_bytes"] == 256
            assert engine.last_memory_sample_s is not None
            assert metrics.mlx_active._value.get() == 0  # Explicit CPU stub sample.
        finally:
            await engine.close()
    asyncio.run(scenario())


def test_profile_summary_keeps_unsupported_values_unknown():
    supported = {"prompt_repeats": 1, "client_elapsed_s": .1,
                 "server": {"prompt_tokens": 10, "memory_profile": [{"cache_allocated_bytes": 256}]}}
    unsupported = {"prompt_repeats": 2, "client_elapsed_s": .2,
                   "server": {"prompt_tokens": 20, "memory_profile": [{"cache_allocated_bytes": None}]}}
    summary = summarize_profiles([supported, unsupported])
    assert summary[0]["median_peak_cache_array_bytes"] == 256
    assert summary[1]["median_peak_cache_array_bytes"] is None
    assert not summary[1]["cache_profile_supported"]
