"""The acceptance checks: a complete good session passes, and each broken property fails or is inconclusive on its own item.

Fixtures are synthetic (tests/acceptance_fixture.py). They prove the checks are wired to the right evidence and thresholds;
they say nothing about TensorPilot's performance.
"""
import json
import sys

import pytest

from acceptance_fixture import Session, archive, capture, good_records, invariance, probe_report, record
from tensorpilot import acceptance
from tensorpilot.acceptance import FAIL, INCONCLUSIVE, PASS


def run(session, tmp_path, *, report="", spent=6.4, archives=True, criteria=None, ledger=True, secrets=()):
    arch = []
    if archives:
        arch = [archive(tmp_path / "archives", "7b-session")]
    report_path = None
    if report is not None:
        report_path = tmp_path / "report.md"
        report_path.write_text(report or "## Why vLLM differs\nTensorPilot decodes at 21 ms per step against 9 ms, 41 tok/s vs 98 tok/s, GPU util 19%.\n")
    ledger_path = tmp_path / "ledger.jsonl"
    ledger_path.write_text("\n".join(json.dumps(e) for e in session.ledger) + "\n")
    return acceptance.evaluate(session.root, criteria=criteria or session.criteria, ledgers=[ledger_path] if ledger else [],
                               archives=arch, report=report_path, spent_usd=spent, secrets=secrets)


def status(verdict, prefix):
    rows = [r for r in verdict["items"] if r["id"].startswith(prefix)]
    assert rows, f"no item {prefix}: {[r['id'] for r in verdict['items']]}"
    return rows[0]["status"] if len(rows) == 1 else [r["status"] for r in rows]


@pytest.fixture
def good(tmp_path):
    session = Session(tmp_path / "session").build()
    return session


def test_a_complete_good_session_passes_every_gate_and_every_performance_target(good, tmp_path):
    verdict = run(good, tmp_path)
    failing = [(r["id"], r["status"], r["detail"]) for r in verdict["items"] if r["status"] != PASS]
    assert verdict["evaluation"] == PASS and verdict["performance"] == PASS, failing


def test_an_empty_session_is_inconclusive_never_a_pass(tmp_path):
    (tmp_path / "empty").mkdir()
    verdict = acceptance.evaluate(tmp_path / "empty", criteria=Session(tmp_path / "x").criteria)
    assert verdict["evaluation"] == INCONCLUSIVE and verdict["performance"] == INCONCLUSIVE
    assert verdict["counts"][PASS] == 0 and verdict["counts"][FAIL] == 0


def test_every_item_has_an_explicit_verdict_and_a_stated_reason(good, tmp_path):
    verdict = run(good, tmp_path)
    assert all(r["status"] in (PASS, FAIL, INCONCLUSIVE) and r["detail"] for r in verdict["items"])
    assert {r["criterion"] for r in verdict["items"]} == {1, 2, 3, 4, 5, 6, 7}


def test_the_criteria_file_pins_the_stated_numbers():
    c = acceptance.load_criteria()
    assert c["correctness"]["protocol_valid_min"] == 0.999
    assert c["scale"]["concurrency_levels"] == [1, 32, 64, 128, 256, 512, 1024]
    assert c["scale"]["min_total_requests"] == 10000 and c["scale"]["stretch_total_requests"] == 100000
    assert c["scale"]["min_speedup_over_serial"] == 1.5
    assert c["reliability"]["sustained_success_min"] == 0.999 and c["reliability"]["recovery_max_s"] == 60
    assert c["latency"]["ttft_p95_s_max"] == 2.0 and c["latency"]["itl_p95_s_max"] == 0.15
    assert c["vllm"]["min_repeats"] == 3 and c["budget"]["usd"] == 9.0
    assert c["declared_before_execution"] is True


def test_the_slo_table_matches_the_suite_so_the_slo_is_the_one_declared_before_execution():
    slo = acceptance.load_criteria()["latency"]["slo_by_model"]
    from tensorpilot import suite
    assert {k: slo[k] for k in ("7b", "14b")} == {k: suite.MODELS[k]["slo"] for k in ("7b", "14b")}


# ----- 1. correctness -------------------------------------------------------------------------------------------

def test_protocol_invalid_responses_below_the_bar_fail(good, tmp_path):
    good.experiment("7b-bf16-prod", "cancellation", good_records(1990) + [record(2000 + i, outcome="failed", error="invalid_stream")
                                                                          for i in range(10)])
    assert status(run(good, tmp_path), "1.protocol-valid") == FAIL


def test_a_single_invalid_response_in_many_still_passes_the_999_bar(good, tmp_path):
    good.experiment("7b-bf16-prod", "cancellation", good_records(3000) + [record(4000, outcome="failed", error="invalid_stream")])
    assert status(run(good, tmp_path), "1.protocol-valid") == PASS


def test_rejections_timeouts_and_cancels_are_not_protocol_violations(good, tmp_path):
    extra = [record(100 + i, outcome=o, error=e) for i, (o, e) in enumerate(
        [("rejected", "overloaded"), ("timed_out", "deadline_exceeded"), ("client_cancelled", None)] * 50)]
    good.experiment("7b-bf16-prod", "timeouts", good_records(12) + extra)
    assert status(run(good, tmp_path), "1.protocol-valid") == PASS


def test_silent_truncation_is_caught(good, tmp_path):
    cut = good_records(10) + [record(10, tokens=12, max_tokens=20, finish="length")]
    good.experiment("7b-bf16-prod", "timeouts", cut)
    assert status(run(good, tmp_path), "1.no-truncation") == FAIL


def test_a_missing_finish_reason_counts_as_truncation(good, tmp_path):
    good.experiment("7b-bf16-prod", "timeouts", good_records(10) + [record(10, finish=None)])
    assert status(run(good, tmp_path), "1.no-truncation") == FAIL


def test_duplicate_completions_are_caught(good, tmp_path):
    good.experiment("7b-bf16-prod", "timeouts", [record(0, request_id="same"), record(1, request_id="same")])
    assert status(run(good, tmp_path), "1.no-truncation") == FAIL


def test_a_request_missing_from_the_journal_is_unaccounted(good, tmp_path):
    good.experiment("7b-bf16-prod", "timeouts", good_records(12), drop_journal=1)
    verdict = run(good, tmp_path)
    assert status(verdict, "1.no-truncation") == FAIL
    assert "journal has 11 records" in next(r for r in verdict["items"] if r["id"].startswith("1.no-trunc"))["detail"]


def probe_file(session, stage="probe-7b"):
    return session.stage_dir(stage) / "divergence-probe.json"


def test_bf16_divergence_needs_a_probe_that_reproduced_it_with_the_same_packing(good, tmp_path):
    (good.stage_dir("7b-bf16-prod") / "invariance.json").write_text(json.dumps(invariance(0.96875, diverged=(21,))))
    path = probe_file(good)
    path.unlink()
    assert status(run(good, tmp_path), "1.exact-match-7b") == INCONCLUSIVE            # no probe at all
    path.write_text(json.dumps(probe_report(2048, [])))
    assert status(run(good, tmp_path), "1.exact-match-7b") == INCONCLUSIVE            # a passing probe that did not reproduce 21
    path.write_text(json.dumps(probe_report(0, [21])))
    assert status(run(good, tmp_path), "1.exact-match-7b") == INCONCLUSIVE            # reproduced, but with different packing
    path.write_text(json.dumps(probe_report(2048, [21])))
    assert status(run(good, tmp_path), "1.exact-match-7b") == PASS                    # reproduced with the same packing
    path.write_text(json.dumps(probe_report(2048, [], unexplained_indices=[5])))
    assert status(run(good, tmp_path), "1.exact-match-7b") == FAIL


def test_every_diverged_request_must_be_reproduced_not_just_one(good, tmp_path):
    (good.stage_dir("7b-bf16-prod") / "invariance.json").write_text(json.dumps(invariance(0.9375, diverged=(21, 22))))
    probe_file(good).write_text(json.dumps(probe_report(2048, [21])))
    verdict = run(good, tmp_path)
    row = next(r for r in verdict["items"] if r["id"] == "1.exact-match-7b-bf16")
    assert row["status"] == INCONCLUSIVE and "[22]" in row["detail"]


def test_the_operator_packing_override_is_what_the_probe_must_match(good, tmp_path):
    info = good.stage_dir("7b-bf16-prod") / "stage-info.json"
    data = json.loads(info.read_text())
    data["environment"]["PREFILL_PACK_TOKENS"] = "512"
    info.write_text(json.dumps(data))
    (good.stage_dir("7b-bf16-prod") / "invariance.json").write_text(json.dumps(invariance(0.96875)))
    probe_file(good).write_text(json.dumps(probe_report(2048, [21])))
    assert status(run(good, tmp_path), "1.exact-match-7b") == INCONCLUSIVE
    probe_file(good).write_text(json.dumps(probe_report(512, [21])))
    assert status(run(good, tmp_path), "1.exact-match-7b") == PASS


def test_probes_run_with_different_packing_settings_are_all_considered(good, tmp_path):
    good.root.joinpath("probe-0.5b-20260101T000000Z").mkdir()
    (good.root / "probe-0.5b-20260101T000000Z" / "divergence-probe.json").write_text(json.dumps(probe_report(2048, [2])))
    assert status(run(good, tmp_path), "1.quick-session-divergence") == PASS
    good.root.joinpath("probe-0.5b-20260101T000001Z").mkdir()
    (good.root / "probe-0.5b-20260101T000001Z" / "divergence-probe.json").write_text(
        json.dumps(probe_report(2048, [], unexplained_indices=[9])))
    assert status(run(good, tmp_path), "1.quick-session-divergence") == FAIL


def test_the_quick_session_divergence_needs_its_own_logit_evidence(good, tmp_path):
    (good.stage_dir("probe-0.5b") / "divergence-probe.json").unlink()
    assert status(run(good, tmp_path), "1.quick-session-divergence") == INCONCLUSIVE


def test_nf4_fidelity_thresholds(good, tmp_path):
    comparison = good.stage_dir("quality-7b") / "comparison.json"
    for ppl, top1, expected in ((1.03, 0.95, PASS), (1.25, 0.95, FAIL), (1.03, 0.80, FAIL)):
        comparison.write_text(json.dumps({"perplexity_ratio_test_over_reference": ppl, "top1_agreement": top1,
                                          "top5_overlap": 0.9}))
        assert status(run(good, tmp_path), "1.nf4-fidelity-7b") == expected


def test_14b_nf4_must_not_be_worse_than_7b_bf16_on_the_same_text(good, tmp_path):
    path = good.stage_dir("quality-14b") / "nf4-absolute.json"
    path.write_text(json.dumps(capture(-2.5)))      # much worse perplexity than the 7B reference
    assert status(run(good, tmp_path), "1.nf4-fidelity-14b") == FAIL
    other = capture(-1.3)
    other["corpus_sha256"] = "d" * 64
    path.write_text(json.dumps(other))
    assert status(run(good, tmp_path), "1.nf4-fidelity-14b") == INCONCLUSIVE


def test_a_model_that_never_loaded_fails_its_loads_item(good, tmp_path):
    (good.stage_dir("14b-nf4-prod") / "smoke.json").write_text(json.dumps({"summary": {"successful_requests": 0}}))
    assert status(run(good, tmp_path), "1.loads-and-generates-14B") == FAIL


# ----- 2. scale and throughput ----------------------------------------------------------------------------------

def test_a_missing_concurrency_level_is_inconclusive_not_silently_skipped(good, tmp_path):
    import shutil
    shutil.rmtree(good.stage_dir("7b-bf16-prod") / "bench" / "scale-c1024")
    verdict = run(good, tmp_path)
    assert status(verdict, "2.concurrency-levels-7b") == INCONCLUSIVE
    assert verdict["evaluation"] == INCONCLUSIVE


def test_a_level_cut_short_by_its_time_cap_is_not_a_measurement(good, tmp_path):
    good.experiment("7b-bf16-prod", "scale-c512", good_records(12), status="aborted", reason="time_cap")
    assert status(run(good, tmp_path), "2.concurrency-levels-7b") == INCONCLUSIVE


def test_a_level_ended_by_the_error_rate_rule_is_a_measurement_of_saturation(good, tmp_path):
    good.experiment("7b-bf16-prod", "scale-c1024", [record(i, outcome="timed_out", error="deadline_exceeded") for i in range(12)],
                    status="aborted", reason="error_rate")
    assert status(run(good, tmp_path), "2.concurrency-levels-7b") == PASS


def test_the_real_ten_thousand_request_floor_is_enforced(good, tmp_path):
    verdict = run(good, tmp_path, criteria=acceptance.load_criteria())
    row = next(r for r in verdict["items"] if r["id"] == "2.request-volume")
    assert row["status"] == INCONCLUSIVE and row["measured"]["stretch_met"] is False


def test_a_ladder_that_never_saturates_gives_only_a_lower_bound(good, tmp_path):
    for rate in (12, 16):
        good.experiment("7b-bf16-prod", "ladder-poisson", good_records(12), rate=float(rate), label=f"rate{rate}.0")
    assert status(run(good, tmp_path), "2.saturation-found-7b") == INCONCLUSIVE


def test_the_ladder_reports_the_highest_sustained_rate(good, tmp_path):
    verdict = run(good, tmp_path)
    row = next(r for r in verdict["items"] if r["id"] == "2.saturation-found-7b")
    assert row["status"] == PASS and row["measured"]["max_sustainable_rate"] == 8


def test_speedup_below_1_5x_is_a_performance_fail_that_does_not_fail_the_evaluation(tmp_path):
    session = Session(tmp_path / "s").build(serial_tps=250.0, batch_tps=300.0)
    verdict = run(session, tmp_path)
    assert status(verdict, "2.speedup-over-serial") == FAIL
    assert verdict["performance"] == FAIL and verdict["evaluation"] == PASS


def test_speedup_uses_comparable_workloads_from_both_stages(good, tmp_path):
    verdict = run(good, tmp_path)
    row = next(r for r in verdict["items"] if r["id"] == "2.speedup-over-serial")
    assert set(row["measured"]["ratios"]) == {"cmp-c8", "cmp-c32"} and row["measured"]["ratios"]["cmp-c8"] == pytest.approx(6.0)


# ----- 3. reliability ------------------------------------------------------------------------------------------

def test_sustained_success_below_999_fails_and_too_small_a_sample_is_inconclusive(good, tmp_path):
    flawed = good_records(1000) + [record(1000 + i, outcome="failed", error="inference_error") for i in range(5)]
    good.experiment("7b-bf16-prod", "sustained-c32", flawed)
    assert status(run(good, tmp_path), "3.sustained-success") == FAIL
    good.experiment("7b-bf16-prod", "sustained-c32", good_records(200))
    assert status(run(good, tmp_path), "3.sustained-success") == INCONCLUSIVE


def test_sustained_success_excludes_cancels_and_overload_rejections_but_not_timeouts(good, tmp_path):
    excluded = good_records(1200) + [record(2000 + i, outcome="client_cancelled") for i in range(300)] + \
        [record(3000 + i, outcome="rejected", error="overloaded") for i in range(300)]
    good.experiment("7b-bf16-prod", "sustained-c32", excluded)
    assert status(run(good, tmp_path), "3.sustained-success") == PASS
    timeouts = good_records(1200) + [record(2000 + i, outcome="timed_out", error="deadline_exceeded") for i in range(5)]
    good.experiment("7b-bf16-prod", "sustained-c32", timeouts)
    assert status(run(good, tmp_path), "3.sustained-success") == FAIL


def test_any_cuda_oom_event_fails(good, tmp_path):
    good.experiment("7b-bf16-prod", "cancellation", good_records(12), oom=1)
    assert status(run(good, tmp_path), "3.zero-cuda-oom") == FAIL


def test_oom_text_in_an_engine_log_fails_even_when_the_counter_is_zero(good, tmp_path):
    (good.stage_dir("7b-bf16-prod") / "engine.log").write_text("torch.OutOfMemoryError: CUDA out of memory\n")
    assert status(run(good, tmp_path), "3.zero-cuda-oom") == FAIL


def test_overload_must_exceed_the_queue_to_prove_it_is_bounded(good, tmp_path):
    good.experiment("7b-bf16-prod", "overload", good_records(20), slo=False, rate=400.0, label="rate400.0")
    assert status(run(good, tmp_path), "3.overload") == INCONCLUSIVE


def test_a_queue_deeper_than_its_capacity_fails(good, tmp_path):
    capacity = 1100
    overload = good_records(12) + [record(100 + i, outcome="rejected", error="overloaded") for i in range(8)]
    good.experiment("7b-bf16-prod", "overload", overload, slo=False, rate=400.0, label="rate400.0", queue=[capacity + 5])
    assert status(run(good, tmp_path), "3.overload") == FAIL


def test_uncontrolled_failures_during_overload_fail(good, tmp_path):
    overload = good_records(12) + [record(100 + i, outcome="rejected", error="overloaded") for i in range(8)] + \
        [record(200 + i, outcome="failed", error="inference_error") for i in range(5)]
    good.experiment("7b-bf16-prod", "overload", overload, slo=False, rate=400.0, label="rate400.0", queue=[0, 1100])
    assert status(run(good, tmp_path), "3.overload") == FAIL


def test_a_leak_across_overload_fails(good, tmp_path):
    good.experiment("7b-bf16-prod", "recovery-after-overload", good_records(64), slo=False, allocated=3_000_000_000 + 500 * 1024**2)
    assert status(run(good, tmp_path), "3.overload") == FAIL


def test_a_queue_that_has_not_drained_after_overload_fails(good, tmp_path):
    good.experiment("7b-bf16-prod", "recovery-after-overload", good_records(64), slo=False, queue=[50])
    assert status(run(good, tmp_path), "3.overload") == FAIL


def test_the_worker_must_serve_normally_right_after_overload(good, tmp_path):
    good.experiment("7b-bf16-prod", "recovery-after-overload", good_records(60) + [record(60 + i, outcome="failed", error="x")
                                                                                    for i in range(4)], slo=False)
    assert status(run(good, tmp_path), "3.overload") == FAIL


def test_crash_recovery_within_60_seconds_passes_and_slower_fails(good, tmp_path):
    path = good.stage_dir("faults-7b") / "fault" / "fault-report.json"
    for seconds, expected in ((59.9, PASS), (60.1, FAIL)):
        path.write_text(json.dumps({"passed": True, "kill": {"recovery_s": seconds}, "checks": []}))
        assert status(run(good, tmp_path), "3.crash-recovery") == expected


def test_a_crash_with_lost_request_accounting_fails_even_if_recovery_is_fast(good, tmp_path):
    path = good.stage_dir("faults-7b") / "fault" / "fault-report.json"
    path.write_text(json.dumps({"passed": False, "kill": {"recovery_s": 10.0},
                                "checks": [{"name": "one_record_per_request", "passed": False}]}))
    verdict = run(good, tmp_path)
    assert status(verdict, "3.crash-recovery") == FAIL and "one_record_per_request" in \
        next(r for r in verdict["items"] if r["id"] == "3.crash-recovery")["detail"]


# ----- 4. latency ----------------------------------------------------------------------------------------------

def test_latency_misses_at_the_operating_point_are_a_performance_fail_only(tmp_path):
    # The 14B SLO (4 s TTFT, 350 ms per token) is looser than the 2 s / 150 ms target, so a rate can be "sustainable" under the
    # declared SLO and still miss the target. That is a performance FAIL; it must not change the evaluation verdict.
    session = Session(tmp_path / "s").build()
    for rate in (1, 2, 4, 8):
        session.experiment("14b-nf4-prod", "ladder-poisson", [record(i, ttft=3.0, itl=0.2) for i in range(12)],
                           rate=float(rate), label=f"rate{rate}.0")
    verdict = run(session, tmp_path)
    row = next(r for r in verdict["items"] if r["id"] == "4.latency-at-operating-point-14b")
    assert row["status"] == FAIL and "3.000" in row["detail"]
    assert status(verdict, "4.latency-at-operating-point-7b") == PASS
    assert verdict["evaluation"] == PASS and verdict["performance"] == FAIL


def test_no_sustainable_rate_means_no_operating_point(tmp_path):
    session = Session(tmp_path / "s").build()
    for rate in (1, 2, 4, 8, 12, 16):
        session.experiment("7b-bf16-prod", "ladder-poisson", [record(i, ttft=9.0) for i in range(12)], rate=float(rate),
                           label=f"rate{rate}.0")
    assert status(run(session, tmp_path), "4.latency-at-operating-point-7b") == FAIL


def test_a_run_that_used_a_different_slo_than_declared_fails(good, tmp_path):
    good.experiment("7b-bf16-prod", "cancellation", good_records(12), slo={"ttft_s": 30.0, "e2e_s": 300.0, "tpot_s": 5.0})
    assert status(run(good, tmp_path), "4.slo-declared-before-execution") == FAIL


def test_an_attainment_that_ignores_non_cancelled_failures_is_caught(good, tmp_path):
    path = good.stage_dir("7b-bf16-prod") / "bench" / "cancellation" / "repeat01_closed.json"
    report = json.loads(path.read_text())
    report["summary"]["slo"]["attainment"] = 1.0
    report["summary"]["slo"]["good_requests"] = report["summary"]["requests_total"] // 2
    path.write_text(json.dumps(report))
    assert status(run(good, tmp_path), "4.failures-not-hidden") == FAIL


# ----- 5. vLLM comparison -------------------------------------------------------------------------------------

def test_a_vllm_run_on_a_different_gpu_or_pod_fails_the_same_hardware_item(good, tmp_path):
    good.finish_stage("vllm-7b-bf16-prod", uuid="GPU-other")
    assert status(run(good, tmp_path), "5.same-hardware-7b") == FAIL
    good.finish_stage("vllm-7b-bf16-prod", pod="pod-zzz")
    assert status(run(good, tmp_path), "5.same-hardware-7b") == FAIL
    good.finish_stage("vllm-7b-bf16-prod")
    assert status(run(good, tmp_path), "5.same-hardware-7b") == PASS


def rerun_session(good, *, with_control):
    """The vLLM stage failed in the main session: replace it by the lean re-run, on a second Pod, with or without a TensorPilot control."""
    import shutil
    shutil.rmtree(good.stage_dir("vllm-7b-bf16-prod"))
    for stage in ("vllm-7b-bf16-cmp",) + (("7b-bf16-cmp",) if with_control else ()):
        good.experiment(stage, "cold-start", good_records(4), slo=False)
        for point in ("cmp-c1", "cmp-c8", "cmp-c32", "cmp-c128"):
            for repeat in (1, 2, 3):
                good.experiment(stage, point, good_records(12), repeat=repeat)
        good.finish_stage(stage, pod="pod-rerun", uuid="GPU-rerun")
    return good


def test_a_lean_vllm_rerun_is_compared_with_the_tensorpilot_control_from_the_same_pod(good, tmp_path):
    from tensorpilot.acceptance import vllm_pair_7b
    rerun_session(good, with_control=True)
    verdict = run(good, tmp_path)
    assert status(verdict, "5.same-hardware-7b") == PASS, "the control and vLLM share a Pod and a card"
    assert status(verdict, "5.repeats-and-workloads-7b") == PASS and status(verdict, "5.metrics-reported-7b") == PASS
    assert vllm_pair_7b({"vllm-7b-bf16-cmp": 1, "7b-bf16-cmp": 1, "7b-bf16-prod": 1}) == ("7b-bf16-cmp", "vllm-7b-bf16-cmp", "7b")


def test_a_lean_vllm_rerun_without_the_control_is_honest_that_the_cards_differ(good, tmp_path):
    """Pairing vLLM with the archived run from another Pod is allowed, but the same-hardware item must say it is not the same card."""
    rerun_session(good, with_control=False)
    row = next(r for r in run(good, tmp_path)["items"] if r["id"] == "5.same-hardware-7b")
    assert row["status"] == FAIL and "GPU-rerun" in row["detail"]


def test_a_vllm_stage_that_never_started_is_inconclusive_not_a_hardware_mismatch(good, tmp_path):
    """The 7B session's vLLM engine died during start-up: the stage has a stage-info.json (same card, same Pod) and no manifest."""
    (good.stage_dir("vllm-7b-bf16-prod") / "bench" / "manifest.json").unlink()
    row = next(r for r in run(good, tmp_path)["items"] if r["id"] == "5.same-hardware-7b")
    assert row["status"] == INCONCLUSIVE and "no benchmark manifest" in row["detail"]


def test_unproven_hardware_identity_is_inconclusive(good, tmp_path):
    (good.stage_dir("vllm-7b-bf16-prod") / "stage-info.json").unlink()
    assert status(run(good, tmp_path), "5.same-hardware-7b") == INCONCLUSIVE


def test_two_repeats_are_not_three(good, tmp_path):
    for suffix in ("repeat03_closed.json", "repeat03_closed.requests.jsonl"):
        (good.stage_dir("vllm-7b-bf16-prod") / "bench" / "cmp-c32" / suffix).unlink()
    assert status(run(good, tmp_path), "5.repeats-and-workloads-7b") == INCONCLUSIVE


def test_engines_that_did_not_get_the_same_workload_fail(good, tmp_path):
    good.experiment("vllm-7b-bf16-prod", "cmp-c8", good_records(12), repeat=1, workload="x" * 64)
    assert status(run(good, tmp_path), "5.repeats-and-workloads-7b") == FAIL


def test_the_gap_analysis_needs_numbers_not_assertions(good, tmp_path):
    assert status(run(good, tmp_path, report=None), "5.gaps-explained") == INCONCLUSIVE
    assert status(run(good, tmp_path, report="# Report\nno section\n"), "5.gaps-explained") == INCONCLUSIVE
    assert status(run(good, tmp_path, report="## Why vLLM differs\nvLLM is better engineered.\n"), "5.gaps-explained") == INCONCLUSIVE
    assert status(run(good, tmp_path), "5.gaps-explained") == PASS


def test_the_14b_vllm_comparison_is_optional_and_does_not_block_a_pass(good, tmp_path):
    verdict = run(good, tmp_path)
    assert not [r for r in verdict["items"] if r["id"].endswith("-14b") and r["id"].startswith("5.")]
    assert verdict["evaluation"] == PASS


# ----- 6. evidence ---------------------------------------------------------------------------------------------

def test_a_gap_in_gpu_telemetry_fails(good, tmp_path):
    path = good.stage_dir("7b-bf16-prod") / "bench" / "scale-c128" / "telemetry" / "gpu.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    path.write_text("\n".join(json.dumps(r) for r in rows if not 5 <= r["t"] <= 14) + "\n")
    assert status(run(good, tmp_path), "6.gpu-telemetry") == FAIL


def test_missing_gpu_telemetry_fails(good, tmp_path):
    (good.stage_dir("7b-bf16-prod") / "bench" / "scale-c128" / "telemetry" / "gpu.jsonl").unlink()
    assert status(run(good, tmp_path), "6.gpu-telemetry") == FAIL


def test_a_run_that_did_not_use_the_committed_config_fails(good, tmp_path):
    good.finish_stage("7b-bf16-prod", config_ok=False)
    assert status(run(good, tmp_path), "6.pinned") == FAIL


def test_an_unpinned_model_revision_fails(good, tmp_path):
    good.finish_stage("7b-bf16-prod", revision="main")
    assert status(run(good, tmp_path), "6.pinned") == FAIL


def test_an_undeclared_image_digest_is_inconclusive(good, tmp_path):
    good.finish_stage("7b-bf16-prod", image="ghcr.io/x/cuda-experiment:v1.0.0-rc2-1")
    assert status(run(good, tmp_path), "6.pinned") == INCONCLUSIVE


def test_archives_with_a_wrong_hash_fail(good, tmp_path):
    verdict = run(good, tmp_path)
    assert status(verdict, "6.archives") == PASS
    path = tmp_path / "archives" / "7b-session.tar.gz.sha256"
    path.write_text("0" * 64 + "  7b-session.tar.gz\n")
    assert status(acceptance.evaluate(good.root, criteria=good.criteria, archives=[tmp_path / "archives" / "7b-session.tar.gz"]),
                  "6.archives") == FAIL


@pytest.mark.parametrize("secret", [b"Authorization: Bearer abcdefghijklmnop1234567890", b"hf_" + b"a" * 30,
                                    b"ghp_" + b"b" * 36, b"-----BEGIN OPENSSH PRIVATE KEY-----", b"AKIA" + b"A" * 16])
def test_a_credential_shape_inside_an_archive_fails(good, tmp_path, secret):
    path = archive(tmp_path / "bad", "leaky", b"log line\n" + secret + b"\n")
    verdict = acceptance.evaluate(good.root, criteria=good.criteria, archives=[path])
    assert status(verdict, "6.archives") == FAIL


def test_the_literal_pod_key_is_searched_for_even_without_a_known_shape(good, tmp_path):
    path = archive(tmp_path / "bad", "leaky", b"key=zzzz-plain-looking-secret\n")
    assert status(acceptance.evaluate(good.root, criteria=good.criteria, archives=[path], secrets=["zzzz-plain-looking-secret"]),
                  "6.archives") == FAIL
    assert status(acceptance.evaluate(good.root, criteria=good.criteria, archives=[path]), "6.archives") == PASS


def test_no_archives_supplied_is_inconclusive(good, tmp_path):
    assert status(run(good, tmp_path, archives=False), "6.archives") == INCONCLUSIVE


# ----- 7. budget ----------------------------------------------------------------------------------------------

def test_a_stage_beyond_its_cap_fails_but_one_within_the_grace_does_not(good, tmp_path):
    good.ledger = [{"stage": "7b-bf16-prod", "elapsed_min": 31.0, "cap_min": 30.0}]
    assert status(run(good, tmp_path), "7.stage-caps") == PASS
    good.ledger = [{"stage": "7b-bf16-prod", "elapsed_min": 33.0, "cap_min": 30.0}]
    assert status(run(good, tmp_path), "7.stage-caps") == FAIL


def test_spend_over_nine_dollars_fails_and_unknown_spend_is_inconclusive(good, tmp_path):
    assert status(run(good, tmp_path, spent=8.99), "7.within-budget") == PASS
    assert status(run(good, tmp_path, spent=9.01), "7.within-budget") == FAIL
    assert status(run(good, tmp_path, spent=None), "7.within-budget") == INCONCLUSIVE


def test_without_a_ledger_the_cap_item_is_inconclusive(good, tmp_path):
    assert status(run(good, tmp_path, ledger=False), "7.stage-caps") == INCONCLUSIVE


# ----- the verdict rule and the command line ---------------------------------------------------------------------

def test_a_gate_fail_dominates_an_inconclusive_and_a_gate_inconclusive_blocks_pass():
    items = [acceptance.Item("a", 1, "t", "correctness", PASS, "d"), acceptance.Item("b", 1, "t", "safety", INCONCLUSIVE, "d")]
    assert acceptance.worst(i.status for i in items) == INCONCLUSIVE
    items.append(acceptance.Item("c", 1, "t", "safety", FAIL, "d"))
    assert acceptance.worst(i.status for i in items) == FAIL
    assert acceptance.worst([]) == INCONCLUSIVE


def test_the_command_line_writes_json_and_markdown_and_exits_by_verdict(good, tmp_path, monkeypatch, capsys):
    (tmp_path / "ledger.jsonl").write_text("\n".join(json.dumps(e) for e in good.ledger) + "\n")
    good.criteria["scale"]["min_total_requests"] = 100
    criteria_path = tmp_path / "criteria.json"
    criteria_path.write_text(json.dumps(good.criteria))
    archive(tmp_path / "arch", "s")
    (tmp_path / "report.md").write_text("## Why vLLM differs\n21 ms vs 9 ms; 41 tok/s vs 98 tok/s; 19%.\n")
    out, md = tmp_path / "verdict.json", tmp_path / "verdict.md"
    monkeypatch.setattr(sys, "argv", ["acceptance", str(good.root), "--criteria", str(criteria_path),
                                      "--ledger", str(tmp_path / "ledger.jsonl"), "--archives", str(tmp_path / "arch"),
                                      "--report", str(tmp_path / "report.md"), "--spent-usd", "6.4",
                                      "--output", str(out), "--markdown", str(md)])
    assert acceptance.main() == 0
    assert json.loads(out.read_text())["evaluation"] == PASS and "Evaluation: PASS" in md.read_text()
    monkeypatch.setattr(sys, "argv", ["acceptance", str(tmp_path / "arch"), "--output", str(tmp_path / "v2.json")])
    assert acceptance.main() == 2


def test_a_crash_report_without_a_recorded_recovery_time_is_never_a_pass(good, tmp_path):
    path = good.stage_dir("faults-7b") / "fault" / "fault-report.json"
    path.write_text(json.dumps({"passed": True, "kill": {}, "checks": []}))
    assert status(run(good, tmp_path), "3.crash-recovery") == INCONCLUSIVE
    path.write_text(json.dumps({"passed": False, "kill": {}, "checks": [{"name": "worker_became_ready", "passed": False}]}))
    assert status(run(good, tmp_path), "3.crash-recovery") == FAIL


def test_a_level_where_the_load_generator_itself_fell_behind_is_not_a_server_measurement(good, tmp_path):
    starved = good_records(12)
    for rec in starved:
        rec["dispatch_lag_s"] = 3.0
    good.experiment("7b-bf16-prod", "scale-c1024", starved)
    verdict = run(good, tmp_path)
    row = next(r for r in verdict["items"] if r["id"] == "2.concurrency-levels-7b")
    assert row["status"] == INCONCLUSIVE and "fell behind" in row["detail"]
