"""The divergence probe separates a rounding-level near-tie from an unexplained mismatch. CPU, tiny random models."""
import pytest

from tensorpilot import divergence_probe as dp


def top(*pairs):
    """A recorded step: (ids, logits), the run's own choice first."""
    return [p[0] for p in pairs], [p[1] for p in pairs]


def test_ulp_matches_the_format_spacing():
    assert dp.ulp("float16", 16.0) == 2.0 ** (4 - 10)
    assert dp.ulp("bfloat16", 16.5) == 2.0 ** (4 - 7)
    assert dp.ulp("float32", 1.0) == 2.0 ** -23
    with pytest.raises(ValueError):
        dp.ulp("int8", 1.0)


def test_a_two_way_tie_that_each_run_breaks_differently_is_a_near_tie():
    spacing = dp.ulp("bfloat16", 18.0)
    solo = top((5, 18.0), (9, 18.0 - spacing))
    batched = top((9, 18.0), (5, 18.0 - spacing))
    verdict = dp.classify(solo, batched, "bfloat16")
    assert verdict["verdict"] == "near_tie" and verdict["gap_solo_ulps"] == pytest.approx(1.0)


def test_a_three_way_tie_is_a_near_tie_even_though_the_runner_ups_do_not_mirror():
    """The four 'unexplained' divergences of the real probe-7b run had this shape: exact ties, and a different 'second' token in each run."""
    solo = top((5, 18.0), (9, 18.0), (7, 18.0))      # solo's top-2 are 5 and 9; 7 is tied too
    batched = top((7, 18.0), (5, 18.0), (9, 18.0))   # batched picked 7: not solo's runner-up, but tied in solo's own logits
    verdict = dp.classify(solo, batched, "bfloat16")
    assert verdict["verdict"] == "near_tie" and verdict["tied_candidates_solo"] == 3


def test_a_choice_that_is_clearly_behind_in_the_other_runs_own_logits_is_unexplained():
    solo = top((5, 18.0), (9, 17.9), (7, 14.0))      # in solo's logits, 7 is 4.0 behind: a rounding-level flip cannot explain choosing it
    batched = top((7, 18.0), (5, 17.99))
    verdict = dp.classify(solo, batched, "bfloat16")
    assert verdict["verdict"] == "unexplained" and verdict["gap_solo_ulps"] > dp.ULP_TOLERANCE


def test_a_choice_missing_from_the_other_runs_candidates_is_unexplained_not_assumed_tied():
    verdict = dp.classify(top((5, 18.0), (9, 17.99)), top((7, 18.0), (5, 17.99)), "bfloat16")
    assert verdict["verdict"] == "unexplained" and "not among" in verdict["reason"]


def test_the_tolerance_is_a_small_fixed_number_of_ulps():
    spacing = dp.ulp("float16", 8.0)
    ok = dp.classify(top((1, 8.0), (2, 8.0 - dp.ULP_TOLERANCE * spacing)), top((2, 8.0), (1, 8.0 - spacing)), "float16")
    bad = dp.classify(top((1, 8.0), (2, 8.0 - (dp.ULP_TOLERANCE + 1) * spacing)), top((2, 8.0), (1, 8.0 - spacing)), "float16")
    assert ok["verdict"] == "near_tie" and bad["verdict"] == "unexplained"


def test_both_sides_must_be_near_the_tie_not_just_one():
    spacing = dp.ulp("bfloat16", 18.0)
    verdict = dp.classify(top((5, 18.0), (9, 18.0 - spacing)), top((9, 18.0), (5, 18.0 - 40 * spacing)), "bfloat16")
    assert verdict["verdict"] == "unexplained"


def test_compare_flags_a_length_mismatch_as_unexplained_never_a_near_tie():
    solo = [{"ids": [1, 2, 3], "steps": [top((1, 2.0), (0, 1.0))] * 3}]
    batched = [{"ids": [1, 2], "steps": [top((1, 2.0), (0, 1.0))] * 2}]
    rows = dp.compare(solo, batched, "float32")
    assert rows[0]["verdict"] == "unexplained" and "lengths differ" in rows[0]["reason"]


def test_identical_runs_report_no_divergence_and_pass():
    run = [{"ids": [1, 2], "steps": [top((1, 2.0), (0, 1.0))] * 2}]
    report = dp.build_report(["p"], run, run, "float32", {})
    assert report["passed"] and report["diverged"] == 0 and report["meaning"] == "no divergence"


def test_tie_exposure_counts_the_steps_that_were_within_one_ulp_of_a_tie():
    spacing = dp.ulp("bfloat16", 18.0)
    steps = [top((1, 18.0), (2, 18.0)), top((1, 18.0), (2, 18.0 - spacing)), top((1, 18.0), (2, 12.0))]
    exposure = dp.tie_exposure([{"ids": [1, 1, 1], "steps": steps}], "bfloat16")
    assert exposure == {"steps": 3, "within_one_ulp": 2, "exact_ties": 1, "fraction_within_one_ulp": pytest.approx(2 / 3)}


def test_the_audit_says_whether_the_batched_logits_are_as_close_to_the_reference_as_the_serial_ones():
    reference = [10.0, 9.0, 8.0] + [0.0] * 5
    close = [10.0, 9.0, 8.0] + [0.0] * 5
    off = [10.5, 9.0, 8.0] + [0.0] * 5
    good = dp.audit(close, close, reference, 0, 1, "bfloat16")
    assert good["batched_not_worse"] and good["solo_error_ulps"] == 0
    bad = dp.audit(close, off, reference, 0, 1, "bfloat16")
    assert not bad["batched_not_worse"] and bad["batched_error_ulps"] > dp.AUDIT_SLACK_ULPS


# ----- end to end on a tiny model: a forced flip is judged by the logits, never by luck --------------------------

SPEC = {"seed": 3, "classes": [{"name": "s", "prompt_tokens": {"dist": "uniform", "min": 3, "max": 12},
                                "output_tokens": {"dist": "uniform", "min": 4, "max": 12}}]}


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from tiny_model import make_tiny_model_dir
    from tensorpilot.cuda_backend import CUDABackend
    backend = CUDABackend(str(make_tiny_model_dir(tmp_path_factory.mktemp("probe"))), device="cpu", allow_cpu=True, dtype="float32")
    backend.eos_token_ids = frozenset()
    return backend


def test_a_real_probe_on_the_tiny_model_matches_solo_and_batched(tiny):
    report = dp.probe(tiny, prompts=6, dtype="float32", pack_tokens=256, prefill_step_size=512,
                      kv_budget_bytes=64 * 1024**2, spec=SPEC)
    assert report["prompts"] == 6 and report["unexplained"] == 0 and report["schema"] == dp.SCHEMA
    assert report["settings"]["prefill_pack_tokens"] == 256 and report["tie_exposure"]["solo"]["steps"] > 0


def _flip(target_index, target_step, *, solo_offset, batched_offset):
    """Hook: at one step of one request, set the runner-up's logit to ``best + offset`` (in units of the logit's own spacing).

    A negative solo offset and a small positive batched offset make a flip between candidates that BOTH runs saw as nearly tied.
    The change is applied to the first pass and to the replay alike, so the audit compares like with like.
    """
    def install(adapter, role):
        original = adapter._choose
        offset = solo_offset if role == "solo" else batched_offset

        def choose(rows, uids):
            index_of = getattr(adapter, "request_index", lambda uid: uid)
            for r, uid in enumerate(uids):
                step = len(adapter.margin_probe.get(uid, [])) if adapter.margin_probe is not None else -1
                if index_of(uid) == target_index and step == target_step and offset is not None:
                    order = rows[r].argsort(descending=True)
                    best, second = int(order[0]), int(order[1])
                    rows[r, second] = rows[r, best] + offset * dp.ulp("float32", float(rows[r, best]))
            return original(rows, uids)
        adapter._choose = choose
    return install


def test_a_flip_between_logits_that_were_nearly_tied_in_both_runs_is_a_near_tie_and_is_audited(tiny):
    report = dp.probe(tiny, prompts=6, dtype="float32", kv_budget_bytes=64 * 1024**2, spec=SPEC,
                      hook=_flip(2, 3, solo_offset=-1.0, batched_offset=1.0))
    [row] = [r for r in report["diverged_requests"] if r["index"] == 2]
    assert row["first_divergence"] == 3 and row["verdict"] == "near_tie", row
    assert report["replay"]["audited"] >= 1 and report["replay"]["deterministic"]
    assert row["audit"]["batched_not_worse"] and report["passed"] is True


def test_a_flip_where_the_solo_run_had_a_clear_winner_is_unexplained_and_fails_the_probe(tiny):
    report = dp.probe(tiny, prompts=6, dtype="float32", kv_budget_bytes=64 * 1024**2, spec=SPEC,
                      hook=_flip(2, 3, solo_offset=None, batched_offset=1e6))
    [row] = [r for r in report["diverged_requests"] if r["index"] == 2]
    assert row["verdict"] == "unexplained" and report["passed"] is False and report["unexplained"] >= 1


def test_a_nearly_tied_flip_is_still_unexplained_if_the_batched_logits_are_farther_from_the_reference_than_the_serial_ones(tiny):
    """The audit's job: a flip that LOOKS like a tie must not pass if only the batched run's logits were disturbed."""
    def hook(adapter, role):
        _flip(2, 3, solo_offset=-1.0, batched_offset=1.0)(adapter, role)
        if role == "batched":
            inner = adapter._choose

            def choose(rows, uids):
                for r, uid in enumerate(uids):
                    if uid == 2 and len(adapter.margin_probe.get(uid, [])) == 3:
                        rows[r] += 3.0      # a leak that shifts every logit: the choice is unchanged, the logits are not
                return inner(rows, uids)
            adapter._choose = choose

    report = dp.probe(tiny, prompts=6, dtype="float32", kv_budget_bytes=64 * 1024**2, spec=SPEC, hook=hook)
    [row] = [r for r in report["diverged_requests"] if r["index"] == 2]
    assert row["verdict"] == "unexplained" and "farther from the reference" in row["reason"]
    assert report["passed"] is False


def test_a_replay_that_does_not_reproduce_the_tokens_is_never_a_pass(tiny):
    """The divergence itself is a clean, audited near-tie; only the replay of OTHER requests drifts. That alone must fail the probe."""
    def hook(adapter, role):
        _flip(2, 3, solo_offset=-1.0, batched_offset=1.0)(adapter, role)
        if role == "batched":
            inner = adapter._choose

            def choose(rows, uids):
                if adapter.logit_capture_filter is not None:      # only the replay adapters capture logits
                    for r, uid in enumerate(uids):
                        if uid != 2:
                            rows[r, 0] += 1000.0                    # other requests now emit a different token
                return inner(rows, uids)
            adapter._choose = choose

    report = dp.probe(tiny, prompts=6, dtype="float32", kv_budget_bytes=64 * 1024**2, spec=SPEC, hook=hook)
    [row] = [r for r in report["diverged_requests"] if r["index"] == 2]
    assert report["replay"]["deterministic"] is False
    assert row["audit"]["batched_not_worse"], "the audit alone would have passed it"
    assert row["verdict"] == "unexplained" and "did not reproduce" in row["reason"] and report["passed"] is False


# ----- the pool the probe asks for must be the pool it gets (the first real-GPU probe-7b failure) -----------------

def test_the_7b_probe_geometry_that_failed_on_the_gpu_is_reproduced_and_the_stage_settings_fix_it():
    """Real 7B numbers: 14.19 GiB of weights, 56 KiB of KV per token, the RTX 4090's allocator cap, 2.5 GiB of headroom."""
    from tensorpilot.cuda_batch import GIB, plan_kv_pool
    weights, per_token, vram = int(14.19 * GIB), 57344, 25252724736
    as_run = plan_kv_pool(weights_bytes=weights, bytes_per_token=per_token, row_tokens=2048 + 512, requested_capacity=32,
                          total_vram_bytes=vram, memory_fraction=0.9)
    assert as_run.capacity == 31 and as_run.limited_by == "vram"
    fixed = plan_kv_pool(weights_bytes=weights, bytes_per_token=per_token, row_tokens=1024 + 512, requested_capacity=32,
                         total_vram_bytes=vram, memory_fraction=0.9)
    assert fixed.capacity == 32 and fixed.limited_by == "requested"


def test_every_probe_stage_asks_for_a_pool_its_model_can_hold():
    import json
    from pathlib import Path
    from tensorpilot import suite
    from tensorpilot.cuda_batch import GIB, plan_kv_pool
    matrix = json.loads((Path(__file__).resolve().parents[1] / "benchmarks" / "models" / "model-matrix-rtx4090.json").read_text())
    for stage, spec in suite.EXTRA_STAGES.items():
        if spec["engine"] != "probe":
            continue
        env = suite.stage_env(stage)
        info = next(m for m in matrix["models"] if m["model"] == suite.MODELS[spec["model"]]["repo"])
        plan = plan_kv_pool(weights_bytes=int(info["weights_gib"] * GIB), bytes_per_token=info["kv_bytes_per_token"],
                            row_tokens=int(env["V10_MAX_PROMPT_TOKENS"]) + 512, requested_capacity=int(env.get("V10_PROBE_PROMPTS", 32)),
                            total_vram_bytes=matrix["gpu_total_bytes"], memory_fraction=matrix["memory_fraction"],
                            headroom_bytes=matrix["headroom_bytes"])
        assert plan.capacity == int(env.get("V10_PROBE_PROMPTS", 32)), f"{stage}: only {plan.capacity} slots fit"


def test_a_probe_whose_pool_is_smaller_than_asked_refuses_before_running_and_names_the_cause(tiny):
    from tensorpilot.cuda_batch import CUDABatchAdapter
    sizing = CUDABatchAdapter(tiny, capacity=1, kv_budget_bytes=64 * 1024**2)
    row_bytes = sizing.row_tokens * sizing.bytes_per_token
    sizing.close()
    with pytest.raises(ValueError, match=r"granted 3 of the 6 slots.*memory cap is not relaxed"):
        dp.probe(tiny, prompts=6, dtype="float32", spec=SPEC, kv_budget_bytes=4 * row_bytes)
    report = dp.probe(tiny, prompts=6, dtype="float32", spec=SPEC, kv_budget_bytes=64 * 1024**2)
    assert report["settings"]["granted_capacity"] == 6 and report["settings"]["row_tokens"] > 0


def test_the_command_line_passes_the_prompt_limit_to_the_backend_so_rows_match_the_benchmark_worker(tmp_path, monkeypatch):
    import sys
    seen = {}

    class FakeBackend:
        def __init__(self, *args, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr("tensorpilot.cuda_backend.CUDABackend", FakeBackend)
    monkeypatch.setattr(dp, "probe", lambda backend, **kw: {"exact_matches": 4, "prompts": 4, "near_tie": 0, "unexplained": 0,
                                                              "passed": True, "meaning": "no divergence",
                                                              "tie_exposure": {"solo": {"within_one_ulp": 0, "steps": 1}}})
    monkeypatch.setattr(sys, "argv", ["divergence_probe", "--model", "m", "--prompts", "4", "--output", str(tmp_path / "o.json")])
    assert dp.main() == 0 and seen["max_prompt_tokens"] == 1024
    monkeypatch.setattr(sys, "argv", ["divergence_probe", "--model", "m", "--prompts", "4", "--max-prompt-tokens", "2048",
                                      "--output", str(tmp_path / "o.json")])
    assert dp.main() == 0 and seen["max_prompt_tokens"] == 2048
