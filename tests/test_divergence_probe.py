"""The divergence probe separates a rounding-level near-tie from an unexplained mismatch. CPU, tiny random model."""
import pytest

from tensorpilot import divergence_probe as dp


def test_ulp_matches_the_format_spacing():
    assert dp.ulp("float16", 16.0) == 2.0 ** (4 - 10)
    assert dp.ulp("bfloat16", 16.5) == 2.0 ** (4 - 7)
    assert dp.ulp("float32", 1.0) == 2.0 ** -23
    with pytest.raises(ValueError):
        dp.ulp("int8", 1.0)


def step(first, second, high, low):
    return (first, second, high, low)


def test_a_consistent_one_ulp_flip_is_a_near_tie():
    spacing = dp.ulp("bfloat16", 18.0)
    verdict = dp.classify(step(5, 9, 18.0, 18.0 - spacing), step(9, 5, 18.0, 18.0 - spacing), "bfloat16")
    assert verdict["verdict"] == "near_tie" and verdict["gap_solo_ulps"] == pytest.approx(1.0)


def test_a_large_gap_is_not_explained_by_rounding():
    verdict = dp.classify(step(5, 9, 18.0, 15.0), step(9, 5, 18.0, 17.99), "bfloat16")
    assert verdict["verdict"] == "unexplained"


def test_inconsistent_candidates_are_not_a_near_tie_even_with_tiny_gaps():
    spacing = dp.ulp("bfloat16", 18.0)
    verdict = dp.classify(step(5, 9, 18.0, 18.0 - spacing), step(7, 5, 18.0, 18.0 - spacing), "bfloat16")
    assert verdict["verdict"] == "unexplained" and not verdict["candidates_consistent"]


def test_the_tolerance_is_a_small_fixed_number_of_ulps():
    spacing = dp.ulp("float16", 8.0)
    ok = dp.classify(step(1, 2, 8.0, 8.0 - dp.ULP_TOLERANCE * spacing), step(2, 1, 8.0, 8.0 - spacing), "float16")
    bad = dp.classify(step(1, 2, 8.0, 8.0 - (dp.ULP_TOLERANCE + 1) * spacing), step(2, 1, 8.0, 8.0 - spacing), "float16")
    assert ok["verdict"] == "near_tie" and bad["verdict"] == "unexplained"


def test_compare_flags_a_length_mismatch_as_unexplained_never_a_near_tie():
    solo = [{"ids": [1, 2, 3], "steps": [(1, 0, 2.0, 1.0)] * 3}]
    batched = [{"ids": [1, 2], "steps": [(1, 0, 2.0, 1.0)] * 2}]
    rows = dp.compare(solo, batched, "float32")
    assert rows[0]["verdict"] == "unexplained" and "lengths differ" in rows[0]["reason"]


def test_identical_runs_report_no_divergence_and_pass():
    run = [{"ids": [1, 2], "steps": [(1, 0, 2.0, 1.0)] * 2}]
    report = dp.build_report(["p"], run, run, "float32", {})
    assert report["passed"] and report["diverged"] == 0 and report["meaning"] == "no divergence"


def test_a_real_probe_on_the_tiny_model_matches_solo_and_batched(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from tiny_model import make_tiny_model_dir
    from tensorpilot.cuda_backend import CUDABackend
    backend = CUDABackend(str(make_tiny_model_dir(tmp_path)), device="cpu", allow_cpu=True, dtype="float32")
    backend.eos_token_ids = frozenset()
    spec = {"seed": 3, "classes": [{"name": "s", "prompt_tokens": {"dist": "uniform", "min": 3, "max": 12},
                                    "output_tokens": {"dist": "uniform", "min": 4, "max": 12}}]}
    texts, budgets = dp.make_prompts(6, spec)
    assert len(texts) == 6 and all(1 <= b <= 12 for b in budgets)
    report = dp.probe(backend, prompts=6, dtype="float32", pack_tokens=256, prefill_step_size=512,
                       kv_budget_bytes=64 * 1024**2, spec=spec)
    assert report["prompts"] == 6 and report["unexplained"] == 0
    assert report["settings"]["prefill_pack_tokens"] == 256


# ----- the pool the probe asks for must be the pool it gets (the real-GPU probe-7b failure) -------------------------

def test_the_7b_probe_geometry_that_failed_on_the_gpu_is_reproduced_and_the_stage_settings_fix_it():
    """Real 7B numbers: 14.19 GiB of weights, 56 KiB of KV per token, the RTX 4090's allocator cap, 2.5 GiB of headroom.

    The probe built its backend with the default prompt limit (2,048), so rows were 2,560 tokens and only 31 slots fit: the
    32nd insert raised 'batch capacity exhausted'. With the benchmark stages' prompt limit (1,024) all 32 slots fit.
    """
    from tensorpilot.cuda_batch import GIB, plan_kv_pool
    weights, per_token, vram = int(14.19 * GIB), 57344, 25252724736
    as_run = plan_kv_pool(weights_bytes=weights, bytes_per_token=per_token, row_tokens=2048 + 512, requested_capacity=32,
                          total_vram_bytes=vram, memory_fraction=0.9)
    assert as_run.capacity == 31 and as_run.limited_by == "vram"
    fixed = plan_kv_pool(weights_bytes=weights, bytes_per_token=per_token, row_tokens=1024 + 512, requested_capacity=32,
                         total_vram_bytes=vram, memory_fraction=0.9)
    assert fixed.capacity == 32 and fixed.limited_by == "requested"


def test_every_probe_stage_asks_for_a_pool_its_model_can_hold():
    """Guards the next model or setting change: the probe's slots must fit beside the weights on the 24 GB card."""
    from tensorpilot import suite
    from tensorpilot.cuda_batch import GIB, plan_kv_pool
    import json
    from pathlib import Path
    matrix = json.loads((Path(__file__).resolve().parents[1] / "benchmarks" / "models" / "model-matrix-rtx4090.json").read_text())
    for stage, spec in suite.EXTRA_STAGES.items():
        if spec["engine"] != "probe":
            continue
        env = suite.stage_env(stage)
        info = next(m for m in matrix["models"] if m["model"] == suite.MODELS[spec["model"]]["repo"])
        row = int(env["V10_MAX_PROMPT_TOKENS"]) + 512
        plan = plan_kv_pool(weights_bytes=int(info["weights_gib"] * GIB), bytes_per_token=info["kv_bytes_per_token"],
                            row_tokens=row, requested_capacity=int(env.get("V10_PROBE_PROMPTS", 32)),
                            total_vram_bytes=matrix["gpu_total_bytes"], memory_fraction=matrix["memory_fraction"],
                            headroom_bytes=matrix["headroom_bytes"])
        assert plan.capacity == int(env.get("V10_PROBE_PROMPTS", 32)), f"{stage}: only {plan.capacity} slots fit"


def test_a_probe_whose_pool_is_smaller_than_asked_refuses_before_running_and_names_the_cause(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from tiny_model import make_tiny_model_dir
    from tensorpilot.cuda_backend import CUDABackend
    backend = CUDABackend(str(make_tiny_model_dir(tmp_path)), device="cpu", allow_cpu=True, dtype="float32")
    backend.eos_token_ids = frozenset()
    spec = {"seed": 3, "classes": [{"name": "s", "prompt_tokens": {"dist": "uniform", "min": 3, "max": 12},
                                    "output_tokens": {"dist": "uniform", "min": 4, "max": 12}}]}
    from tensorpilot.cuda_batch import CUDABatchAdapter
    sizing = CUDABatchAdapter(backend, capacity=1, kv_budget_bytes=64 * 1024**2)
    row_bytes = sizing.row_tokens * sizing.bytes_per_token
    sizing.close()
    # four rows' worth of budget: one is the staging row, so three slots fit, fewer than the six asked for
    with pytest.raises(ValueError, match=r"granted 3 of the 6 slots.*memory cap is not relaxed"):
        dp.probe(backend, prompts=6, dtype="float32", spec=spec, kv_budget_bytes=4 * row_bytes)
    # and the same probe with room for six slots runs and records what it was granted
    report = dp.probe(backend, prompts=6, dtype="float32", spec=spec, kv_budget_bytes=64 * 1024**2)
    assert report["settings"]["granted_capacity"] == 6 and report["settings"]["row_tokens"] > 0


def test_the_command_line_passes_the_prompt_limit_to_the_backend_so_rows_match_the_benchmark_worker(tmp_path, monkeypatch):
    import sys
    seen = {}

    class FakeBackend:
        def __init__(self, *args, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr("tensorpilot.cuda_backend.CUDABackend", FakeBackend)
    monkeypatch.setattr(dp, "probe", lambda backend, **kw: {"exact_matches": 4, "prompts": 4, "near_tie": 0, "unexplained": 0,
                                                              "passed": True, "meaning": "no divergence"})
    monkeypatch.setattr(sys, "argv", ["divergence_probe", "--model", "m", "--prompts", "4", "--output", str(tmp_path / "o.json")])
    assert dp.main() == 0 and seen["max_prompt_tokens"] == 1024
    monkeypatch.setattr(sys, "argv", ["divergence_probe", "--model", "m", "--prompts", "4", "--max-prompt-tokens", "2048",
                                      "--output", str(tmp_path / "o.json")])
    assert dp.main() == 0 and seen["max_prompt_tokens"] == 2048
