"""The report generator against synthetic evidence with known numbers. It must derive correctly and never fill a gap."""
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from tensorpilot import report_v10 as report

REV = {"7b": "a09a35458c702b33eeacc393d103063234e8bc28", "0.5b": "7ae557604adf67be50417f59c2c2f167def9a775",
       "14b": "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"}


def summary(*, completed, tok_s, e2e_p95=2.0, ttft_p95=0.5, error_rate=0.0, attainment=1.0, rejected=0, timed_out=0,
            cancelled=0, active=None):
    return {"requests_total": completed + rejected + timed_out + cancelled, "successful_requests": completed,
            "failed_requests": rejected + timed_out, "rejected_requests": rejected, "timed_out_requests": timed_out,
            "client_cancelled_requests": cancelled, "outcomes": {"completed": completed}, "error_rate": error_rate,
            "completed_output_tokens_per_s": tok_s, "active_window_output_tokens_per_s": active or tok_s * 1.1,
            "successful_requests_per_s": completed / 60, "p50_client_ttft_s": ttft_p95 / 2, "p95_client_ttft_s": ttft_p95,
            "p99_client_ttft_s": ttft_p95 * 1.2, "p50_client_latency_s": e2e_p95 / 2, "p95_client_latency_s": e2e_p95,
            "p99_client_latency_s": e2e_p95 * 1.2, "p50_client_tpot_s": 0.02, "p95_client_tpot_s": 0.04,
            "p95_client_inter_token_s": 0.05,
            "slo": {"attainment": attainment, "goodput_requests_per_s": attainment * completed / 60, "good_requests": 1}}


def run_report(summary_value, *, rate=None, util=50.0, mem=15000.0, limits=None):
    return {"summary": summary_value, "experiment": {"rate_per_s": rate}, "limitations": limits or [],
            "telemetry": {"gpu": {"0": {"utilization_gpu_pct": {"mean": util}, "memory_used_mib": {"max": mem},
                                        "power_w": {"mean": 300.0}}}, "active_sequences": {"mean": 12.0},
                          "active_decode_tokens_per_s": 900.0, "cuda_oom_events_delta": 0}}


def write_stage(root, stage, model, experiments, *, engine="tensorpilot", quantization="none", status=None, extra=None):
    directory = root / f"{stage}-20261101T000000Z"
    (directory / "bench").mkdir(parents=True)
    manifest_experiments = {}
    for name, reports in experiments.items():
        folder = directory / "bench" / name
        folder.mkdir()
        for index, value in enumerate(reports, 1):
            (folder / f"repeat{index:02d}_x.json").write_text(json.dumps(value))
        manifest_experiments[name] = {"status": "completed", "abort_reason": None, "elapsed_s": 30.0, "cap_s": 300.0}
    manifest_experiments.update(status or {})
    (directory / "bench" / "manifest.json").write_text(json.dumps({
        "target": {"model": f"Qwen/{model}", "revision": REV["7b"], "engine": engine, "quantization": quantization,
                   "dtype": "bfloat16"},
        "server_at_start": {"execution_mode": "batch"}, "experiments": manifest_experiments,
        "limitations": ["Planning estimates are assumptions and are never reported as measurements."]}))
    for name, value in (extra or {}).items():
        target = directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(value))
    return directory


def sweep(tokens, *, small_at=(1,)):
    out = {}
    for concurrency, tok_s in tokens.items():
        completed = 60 if concurrency in small_at else 400
        out[f"sweep-c{concurrency}"] = [run_report(summary(completed=completed, tok_s=tok_s, e2e_p95=1 + concurrency / 10))
                                        for _ in range(2)]
    return out


def ladder(rows):
    return {"ladder-poisson": [run_report(summary(completed=120, tok_s=100 * rate, attainment=att, error_rate=err), rate=rate)
                                for rate, att, err in rows]}


@pytest.fixture
def session(tmp_path):
    write_stage(tmp_path, "7b-bf16-batch", "Qwen2.5-7B-Instruct", {
        **sweep({1: 40, 8: 250, 16: 400, 32: 500, 64: 700, 128: 720}),
        **ladder([(0.5, 1.0, 0), (1, 0.99, 0), (2, 0.96, 0), (4, 0.80, 0.1), (8, 0.4, 0.5)]),
        "overhead-telemetry-on": [run_report(summary(completed=300, tok_s=t)) for t in (498, 502, 500)],
        "overhead-telemetry-off": [run_report(summary(completed=300, tok_s=t)) for t in (499, 503, 501)],
        "overload": [run_report(summary(completed=300, tok_s=400, rejected=250, error_rate=0.45))]},
        status={"bursty": {"status": "aborted", "abort_reason": "error_rate_0.350_above_0.2", "elapsed_s": 12, "cap_s": 60}},
        extra={"invariance.json": {"together_vs_solo": {"exact_matches": 30, "comparable": 32, "mean_prefix_fraction": 0.97,
                                                         "earliest_divergence": 41, "median_divergence": 50},
                                   "staggered_vs_solo": {"exact_matches": 31, "comparable": 32, "mean_prefix_fraction": 0.98,
                                                          "earliest_divergence": 44, "median_divergence": 44}}})
    write_stage(tmp_path, "7b-bf16-serial", "Qwen2.5-7B-Instruct", {
        "serial-c1": [run_report(summary(completed=80, tok_s=38))], "serial-c8": [run_report(summary(completed=128, tok_s=39))]})
    write_stage(tmp_path, "vllm-7b-bf16", "Qwen2.5-7B-Instruct", {
        **sweep({1: 44, 8: 300, 16: 560, 32: 800, 64: 1300, 128: 1500}),
        "sweep-c1": [run_report(summary(completed=60, tok_s=44), limits=["different engine, not a configuration of TensorPilot"])]},
        engine="vllm")
    write_stage(tmp_path, "0.5b-serial", "Qwen2.5-0.5B-Instruct", {
        "regression-c1": [run_report(summary(completed=120, tok_s=56.0))]})
    return tmp_path


def test_every_planned_experiment_is_listed_and_gaps_are_never_filled(session, tmp_path):
    out = tmp_path / "out"
    result = report.build(session, out)
    text = (out / "report.md").read_text()
    assert result["stages"] == 4 and "No stage evidence" not in text
    assert re.search(r"\| bursty \| ABORTED \| error_rate_0.350_above_0.2 \|", text)
    assert re.search(r"\| cancellation \| NOT RUN \| no record in the manifest \|", text)
    assert re.search(r"\| sweep-c64 \| completed \|", text)
    assert "Nothing here is estimated" in text and "does not establish" in text


def test_small_samples_are_marked_in_tables_and_hollow_in_charts(session, tmp_path):
    out = tmp_path / "out"
    report.build(session, out)
    text = (out / "report.md").read_text()
    row = next(line for line in text.splitlines() if line.startswith("| 1 | 2 | 120†"))
    assert "†" in row, "c=1 has 60 completed requests per repeat, fewer than 100"
    big = next(line for line in text.splitlines() if line.startswith("| 128 | 2 | 800"))
    assert "†" not in big.split("|")[3]
    svg = (out / "charts" / "throughput-7b-bf16-batch.svg").read_text()
    assert 'fill="var(--surface)" stroke="var(--s1)"' in svg, "a small-sample point is a hollow marker"


def test_charts_are_valid_accessible_svg_using_only_the_validated_series_colors(session, tmp_path):
    out = tmp_path / "out"
    result = report.build(session, out)
    assert result["charts"] >= 8
    for path in (out / "charts").glob("*.svg"):
        root = ET.fromstring(path.read_text())
        assert root.tag.endswith("svg") and root.get("role") == "img"
        assert any(child.tag.endswith("title") for child in root) and any(child.tag.endswith("desc") for child in root)
        colors = set(re.findall(r"var\(--(s\d)\)", path.read_text()))
        assert colors <= {"s1", "s2", "s3"}, f"{path.name} uses more than three series colors"
        assert "prefers-color-scheme: dark" in path.read_text()


def test_the_maximum_sustainable_rate_is_the_highest_rate_with_every_lower_rate_also_meeting_the_objective(session, tmp_path):
    rows = report.ladder_rows(report.load_session(session)["7b-bf16-batch"])
    assert report.max_sustainable(rows) == 2
    assert report.max_sustainable([dict(r, attainment=0.5) for r in rows]) is None
    broken_then_fine = [dict(rows[0], attainment=0.5), dict(rows[1], attainment=0.99, error_rate=0.0)]
    assert report.max_sustainable(broken_then_fine) is None, "a recovery above a failed rate does not count"
    assert report.max_sustainable([dict(rows[0], error_rate=0.05)]) is None
    out = tmp_path / "out"
    report.build(session, out)
    assert "Maximum sustainable rate under it: 2 requests/s" in (out / "report.md").read_text()


def test_the_vllm_comparison_is_computed_and_labeled_and_only_appears_when_both_engines_ran(session, tmp_path):
    out = tmp_path / "out"
    report.build(session, out)
    text = (out / "report.md").read_text()
    assert "TensorPilot against vLLM: 7b-bf16-batch vs vllm-7b-bf16" in text
    assert re.search(r"\| 32 \| 500\.0 \(500\.0–500\.0\) \| 800\.0 \(800\.0–800\.0\) \| 1\.60× \|", text)
    assert "never run at the same time" in text and "CUDA 13.0" in text
    assert "different engine, not a configuration of TensorPilot" in text
    only_ours = tmp_path / "only"
    only_ours.mkdir()
    write_stage(only_ours, "7b-bf16-batch", "Qwen2.5-7B-Instruct", sweep({1: 40, 8: 250}))
    report.build(only_ours, tmp_path / "out2")
    assert "TensorPilot against vLLM" not in (tmp_path / "out2" / "report.md").read_text()


def test_serial_against_batch_reports_the_throughput_ratio_for_matched_experiments(session, tmp_path):
    out = tmp_path / "out"
    report.build(session, out)
    text = (out / "report.md").read_text()
    assert "Serial against continuous batching: 7b-bf16-serial vs 7b-bf16-batch" in text
    row = next(line for line in text.splitlines() if line.startswith("| serial-c8 vs sweep-c8"))
    assert "39.0" in row and "250.0" in row and "6.41×" in row  # 250 / 39


def test_telemetry_overhead_says_when_the_difference_is_within_run_to_run_variation(session, tmp_path):
    out = tmp_path / "out"
    report.build(session, out)
    text = (out / "report.md").read_text()
    assert "Telemetry overhead: 7b-bf16-batch" in text and "not distinguishable from run-to-run variation" in text


def test_the_baseline_regression_ratio_uses_the_immutable_baseline(session, tmp_path):
    out = tmp_path / "out"
    baseline = Path(__file__).resolve().parents[1] / "benchmarks" / "baselines" / "v0.9.1-4-rtx4090"
    report.build(session, out, baseline=baseline)
    text = (out / "report.md").read_text()
    assert "Regression against the immutable v0.9.1-4 baseline" in text
    assert "Baseline 57.51 tok/s; now 56.0 tok/s (0.974× of baseline)" in text


def test_overload_rows_show_client_outcomes_beside_the_servers_view(session, tmp_path):
    out = tmp_path / "out"
    report.build(session, out)
    text = (out / "report.md").read_text()
    row = next(line for line in text.splitlines() if line.startswith("| 7b-bf16-batch | overload"))
    assert "| 250 |" in row and "0.450" in row


def test_crash_fidelity_and_invariance_sections_render_from_their_evidence(session, tmp_path):
    write_stage(session, "faults-7b", "Qwen2.5-7B-Instruct", {}, extra={"fault/fault-report.json": {
        "passed": False, "checks": [{"name": "one_record_per_request", "passed": True, "detail": "300 records"},
                                    {"name": "no_request_completed_across_the_crash", "passed": False,
                                     "detail": "completed although in flight: [4]"}],
        "readiness_timeline": [{"offset_s": 0.0, "state": "ready"}, {"offset_s": 25.1, "state": "unreachable"}]}})
    write_stage(session, "quality-7b", "Qwen2.5-7B-Instruct", {}, extra={"comparison.json": {
        "positions_compared": 900, "perplexity_reference": 7.1, "perplexity_test": 7.3,
        "perplexity_ratio_test_over_reference": 1.028, "top1_agreement": 0.91, "top5_overlap": 0.95,
        "kl_reference_to_test_nats_topk_support": 0.021, "greedy_mean_prefix_agreement": 0.62}})
    out = tmp_path / "out"
    report.build(session, out)
    text = (out / "report.md").read_text()
    assert "**Overall: FAILED.**" in text and "| no_request_completed_across_the_crash | FAIL |" in text
    assert "25.1 s unreachable" in text
    assert "Batch invariance: 7b-bf16-batch" in text and "30/32" in text
    assert "NF4 fidelity against bf16: quality-7b" in text and "closest available evidence" in text
    assert "1.0280" in text.replace("1.028 ", "1.0280") or "1.0280" in text


def test_an_empty_session_says_nothing_was_measured(tmp_path):
    (tmp_path / "empty").mkdir()
    result = report.build(tmp_path / "empty", tmp_path / "out")
    assert result == {"stages": 0, "charts": 0}
    assert "No stage evidence was found" in (tmp_path / "out" / "report.md").read_text()


def test_a_chart_with_no_data_is_not_drawn(tmp_path):
    assert report.line_chart(tmp_path / "x.svg", title="t", x_label="x", y_label="y",
                             series=[{"name": "a", "slot": 1, "points": [(1, None, False)]}]) is False
    assert not (tmp_path / "x.svg").exists()


def test_the_newest_run_of_a_stage_wins_and_missing_files_do_not_break_loading(tmp_path):
    old = tmp_path / "7b-bf16-batch-20261101T000000Z"
    new = tmp_path / "7b-bf16-batch-20261102T000000Z"
    (new / "bench").mkdir(parents=True)
    old.mkdir()
    (tmp_path / "ledger.jsonl").write_text(json.dumps({"stage": "7b-bf16-batch", "exit_status": 124, "elapsed_min": 9}) + "\n")
    stages = report.load_session(tmp_path)
    assert stages["7b-bf16-batch"]["dir"] == new and stages["7b-bf16-batch"]["manifest"] is None
    assert stages["7b-bf16-batch"]["ledger"]["exit_status"] == 124
    out = tmp_path / "out"
    report.build(tmp_path, out)
    assert "NOT RUN" in (out / "report.md").read_text()


def test_the_command_line_writes_the_report(session, tmp_path, monkeypatch, capsys):
    import sys
    monkeypatch.setattr(sys, "argv", ["report", str(session), "--output", str(tmp_path / "cli")])
    assert report.main() == 0
    assert (tmp_path / "cli" / "report.md").exists() and "4 stage(s)" in capsys.readouterr().out


@pytest.mark.parametrize("top", [0.8, 7, 45, 800, 1000, 1500, 7200, 123456])
def test_axis_ticks_are_round_few_and_cover_the_data(top):
    ticks = report._ticks(top)
    assert ticks[0] == 0 and ticks[-1] >= top and 3 <= len(ticks) <= 9
    steps = {round(b - a, 9) for a, b in zip(ticks, ticks[1:])}
    assert len(steps) == 1, "evenly spaced"


def test_the_ladder_chart_draws_the_ideal_as_a_dashed_reference_not_as_a_series(session, tmp_path):
    out = tmp_path / "out"
    report.build(session, out)
    svg = (out / "charts" / "ladder-7b-bf16-batch.svg").read_text()
    assert 'stroke-dasharray="6 4"' in svg and "ideal: every offered request meets the SLO" in svg
    assert len(re.findall(r"var\(--s\d\)", svg.split("</style>")[1])) >= 1 and "var(--s2)" not in svg.split("</style>")[1]
