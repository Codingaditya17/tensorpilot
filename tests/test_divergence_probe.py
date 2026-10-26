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
