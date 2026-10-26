"""The quick-session verdict reads only evidence and fails closed when any of it is missing."""
import json

import pytest

from tensorpilot import quick_compare


def write_experiment(stage_dir, name, *, tokens, wall, completed=10, failed=0, ttft=0.02, status="completed"):
    directory = stage_dir / "bench" / name
    directory.mkdir(parents=True, exist_ok=True)
    summary = {"completed_output_tokens": tokens, "wall_s": wall, "requests_total": completed + failed,
               "successful_requests": completed, "failed_requests": failed, "p95_client_ttft_s": ttft}
    (directory / "repeat01_closed.json").write_text(json.dumps({"summary": summary}))
    (directory / "repeat01_closed.workload.json").write_text("[]")
    manifest_path = stage_dir / "bench" / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"experiments": {}}
    manifest["experiments"][name] = {"status": status}
    manifest_path.write_text(json.dumps(manifest))


def invariance(fraction=1.0, earliest=None):
    row = {"exact_match_fraction": fraction, "earliest_divergence": earliest}
    return {"together_vs_solo": row, "staggered_vs_solo": row}


def stage(tmp_path, name, *, c1, c8, inv=None, gate=True):
    directory = tmp_path / name
    write_experiment(directory, "quick-c1", tokens=c1 * 10, wall=10)
    write_experiment(directory, "quick-c8", tokens=c8 * 10, wall=10)
    if inv is not None:
        (directory / "invariance.json").write_text(json.dumps(inv))
    if gate is not None:
        (directory / "memory-gate.json").write_text(json.dumps({"ok": gate, "checks": [{"detail": "3.1 GiB free"}]}))
    return directory


def verdict(tmp_path, serial_kwargs, batch_kwargs):
    root = tmp_path / f"case{len(list(tmp_path.iterdir()))}"   # each call gets its own evidence, never a previous call's files
    serial = quick_compare.stage_facts(stage(root, "serial", gate=None, **serial_kwargs))
    batch = quick_compare.stage_facts(stage(root, "batch", **batch_kwargs))
    return quick_compare.judge(serial, batch)


def failed(result):
    return {c["name"] for c in result["checks"] if not c["passed"]}


def test_batching_that_is_faster_correct_and_clean_passes_the_gate(tmp_path):
    result = verdict(tmp_path, dict(c1=58, c8=58), dict(c1=55, c8=400, inv=invariance()))
    assert result["passed"], failed(result)


def test_batching_that_is_not_faster_than_serial_fails(tmp_path):
    result = verdict(tmp_path, dict(c1=58, c8=58), dict(c1=55, c8=70, inv=invariance()))
    assert failed(result) == {"batched_c8_beats_serial_c8"}


def test_batching_that_slows_a_single_stream_badly_fails(tmp_path):
    result = verdict(tmp_path, dict(c1=58, c8=58), dict(c1=20, c8=400, inv=invariance()))
    assert failed(result) == {"batched_c1_not_much_slower"}


def test_outputs_that_depend_on_the_neighbours_fail_even_if_batching_is_fast(tmp_path):
    result = verdict(tmp_path, dict(c1=58, c8=58), dict(c1=55, c8=400, inv=invariance(0.5, 0)))
    assert failed(result) == {"batch_invariance"}
    assert "very start" in next(c for c in result["checks"] if c["name"] == "batch_invariance")["detail"]


def test_missing_invariance_or_memory_evidence_fails_closed(tmp_path):
    assert "batch_invariance_present" in failed(verdict(tmp_path, dict(c1=58, c8=58), dict(c1=55, c8=400)))
    assert "memory_gate" in failed(verdict(tmp_path, dict(c1=58, c8=58), dict(c1=55, c8=400, inv=invariance(), gate=None)))
    assert "memory_gate" in failed(verdict(tmp_path, dict(c1=58, c8=58), dict(c1=55, c8=400, inv=invariance(), gate=False)))


def test_a_missing_experiment_fails_closed_instead_of_being_inferred(tmp_path):
    empty = quick_compare.stage_facts(tmp_path / "nothing")
    result = quick_compare.judge(empty, quick_compare.stage_facts(stage(tmp_path, "batch", c1=55, c8=400, inv=invariance())))
    assert "evidence_present" in failed(result) and not result["passed"]


def test_a_failed_request_or_an_unfinished_experiment_fails_but_the_cancel_experiment_may_lose_requests(tmp_path):
    batch = stage(tmp_path, "batch", c1=55, c8=400, inv=invariance())
    write_experiment(batch, "quick-cancel", tokens=100, wall=5, completed=60, failed=36)
    serial = quick_compare.stage_facts(stage(tmp_path, "serial", gate=None, c1=58, c8=58))
    assert quick_compare.judge(serial, quick_compare.stage_facts(batch))["passed"]
    write_experiment(batch, "quick-c32", tokens=100, wall=5, completed=9, failed=1)
    assert "no_failed_requests_and_every_experiment_completed" in failed(
        quick_compare.judge(serial, quick_compare.stage_facts(batch)))
    write_experiment(batch, "quick-c32", tokens=100, wall=5, status="aborted")
    assert "no_failed_requests_and_every_experiment_completed" in failed(
        quick_compare.judge(serial, quick_compare.stage_facts(batch)))


def test_an_interrupted_stage_cannot_pass(tmp_path):
    batch = stage(tmp_path, "batch", c1=55, c8=400, inv=invariance())
    manifest = json.loads((batch / "bench" / "manifest.json").read_text())
    manifest["interrupted"] = True
    (batch / "bench" / "manifest.json").write_text(json.dumps(manifest))
    serial = quick_compare.stage_facts(stage(tmp_path, "serial", gate=None, c1=58, c8=58))
    assert "neither_stage_was_interrupted" in failed(quick_compare.judge(serial, quick_compare.stage_facts(batch)))


@pytest.mark.parametrize("passes,code", [(True, 0), (False, 1)])
def test_the_command_line_prints_the_verdict_writes_json_and_sets_the_exit_code(tmp_path, monkeypatch, capsys, passes, code):
    serial = stage(tmp_path, "serial", gate=None, c1=58, c8=58)
    batch = stage(tmp_path, "batch", c1=55, c8=400 if passes else 60, inv=invariance())
    out = tmp_path / "verdict.json"
    monkeypatch.setattr("sys.argv", ["qc", str(serial), str(batch), "--output", str(out)])
    assert quick_compare.main() == code
    printed = capsys.readouterr().out
    assert ("QUICK GATE: PASS" if passes else "QUICK GATE: FAIL") in printed
    report = json.loads(out.read_text())
    assert report["passed"] is passes and report["criteria"]["batched_c8_throughput_over_serial_c8"] == 1.5
