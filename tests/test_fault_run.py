"""Crash recovery: a real worker process is SIGKILLed under load and the request journal is audited."""
import asyncio
import json
import socket
import sys
from pathlib import Path

import pytest

from tensorpilot import fault_run
from tensorpilot.workloads import build_spec_workload

STUB = str(Path(__file__).with_name("stub_worker.py"))
KEY = "fault-test-key-1234567890"
SPEC = {"seed": 2, "classes": [{"name": "s", "prompt_tokens": 4, "output_tokens": 12}]}


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def record(index, dispatch, elapsed, outcome="completed", tokens=8):
    base = {"index": index, "dispatch_offset_s": dispatch, "client_elapsed_s": elapsed, "outcome": outcome,
            "received_output_tokens": tokens}
    if outcome == "completed":
        base["server"] = {"generated_tokens": tokens}
    return base


def names(checks, passed):
    return {c["name"] for c in checks if c["passed"] is passed}


# ----- the audit, against crafted journals ---------------------------------------------------------

def audit(records, **overrides):
    options = dict(kill_offset_s=5.0, restored_offset_s=7.0, expected_requests=len(records))
    options.update(overrides)
    return fault_run.verify_accounting(records, **options)


def clean_journal():
    return [record(0, 1.0, 0.3), record(1, 4.9, 0.4, "failed", tokens=2),      # in flight at the kill, failed
            record(2, 5.5, 0.01, "failed", tokens=0), record(3, 6.2, 0.01, "failed", tokens=0),  # outage
            record(4, 8.5, 0.3), record(5, 9.0, 0.3)]                             # after recovery


def test_a_correct_crash_journal_passes_every_check():
    checks = audit(clean_journal())
    assert all(c["passed"] for c in checks), [c for c in checks if not c["passed"]]
    assert len(checks) == 7


def test_a_duplicated_request_is_caught():
    journal = clean_journal() + [record(2, 5.6, 0.01, "failed", tokens=0)]
    assert "one_record_per_request" in names(audit(journal, expected_requests=6), False)


def test_a_missing_or_unknown_outcome_is_caught():
    journal = clean_journal()
    journal[3]["outcome"] = "mystery"
    assert "every_request_has_exactly_one_terminal_outcome" in names(audit(journal), False)


def test_a_request_that_completed_across_the_crash_is_caught():
    journal = clean_journal()
    journal[1] = record(1, 4.5, 1.5, "completed")  # in flight when the worker died, yet "completed"
    failed = names(audit(journal), False)
    assert {"no_request_completed_across_the_crash", "in_flight_requests_ended_in_a_terminal_failure"} <= failed


def test_a_completion_during_the_outage_is_caught():
    journal = clean_journal()
    journal[2] = record(2, 5.5, 0.01, "completed")
    assert "requests_during_the_outage_failed_fast_and_none_completed" in names(audit(journal), False)


def test_failures_after_recovery_and_an_empty_post_recovery_window_are_both_failures():
    journal = clean_journal()
    journal[4] = record(4, 8.5, 0.3, "failed", tokens=0)
    assert "requests_after_recovery_complete" in names(audit(journal), False)
    short = [r for r in clean_journal() if r["dispatch_offset_s"] < 8]
    result = {c["name"]: c for c in audit(short)}["requests_after_recovery_complete"]
    assert not result["passed"] and "no request was dispatched after recovery" in result["detail"]


def test_a_token_count_mismatch_on_a_completed_request_is_caught():
    journal = clean_journal()
    journal[0]["server"]["generated_tokens"] = 9
    assert "completed_requests_received_exactly_the_tokens_generated" in names(audit(journal), False)


def test_client_cancellations_are_not_counted_as_crash_victims():
    journal = clean_journal() + [record(6, 4.8, 0.5, "client_cancelled", tokens=1)]
    assert all(c["passed"] for c in audit(journal, expected_requests=7))


# ----- the real thing: a process is killed, restarted and audited ------------------------------------

def run_experiment(tmp_path, monkeypatch, **overrides):
    monkeypatch.setenv("TENSORPILOT_API_KEY", KEY)
    port = free_port()
    options = dict(worker_command=[sys.executable, STUB, "--port", str(port)], url=f"http://127.0.0.1:{port}",
                   api_key=KEY, output_dir=tmp_path / "fault", plan=build_spec_workload(SPEC, 160), rate=20.0,
                   concurrency=32, kill_after_s=2.5, ready_timeout_s=30, recovery_timeout_s=30, max_seconds=120)
    options.update(overrides)
    return asyncio.run(fault_run.run_fault_experiment(**options))


def test_a_worker_killed_under_load_is_restarted_and_every_request_is_accounted_for(tmp_path, monkeypatch):
    report = run_experiment(tmp_path, monkeypatch)
    failures = [c for c in report["checks"] if not c["passed"]]
    assert report["passed"], failures
    checks = {c["name"] for c in report["checks"]}
    assert {"worker_became_ready", "worker_was_killed", "supervisor_restarted_the_worker_with_a_new_process",
            "one_record_per_request", "no_request_completed_across_the_crash", "requests_after_recovery_complete",
            "fresh_requests_succeed_after_recovery", "authentication_survives_the_restart",
            "no_worker_process_left_running"} <= checks
    ready = [e for e in report["lifecycle"] if e["event"] == "worker_ready"]
    assert len(ready) == 2 and ready[0]["pid"] != ready[1]["pid"] and report["kill"]["pid"] == ready[0]["pid"]
    assert report["outcomes"].get("completed", 0) > 0 and report["outcomes"].get("failed", 0) > 0
    states = [entry["state"] for entry in report["readiness_timeline"]]
    assert states[0] == "ready" and "unreachable" in states and states[-1] == "ready"
    saved = json.loads((tmp_path / "fault" / "fault-report.json").read_text())
    assert saved["passed"] is True and saved["schema"] == fault_run.SCHEMA
    assert KEY not in json.dumps(saved), "the credential must never reach the evidence"


def test_a_worker_that_never_becomes_ready_fails_fast_and_leaves_nothing_behind(tmp_path, monkeypatch):
    monkeypatch.setenv("TENSORPILOT_API_KEY", KEY)
    report = asyncio.run(fault_run.run_fault_experiment(
        worker_command=[sys.executable, "-c", "raise SystemExit(3)"], url=f"http://127.0.0.1:{free_port()}",
        api_key=KEY, output_dir=tmp_path / "dead", plan=build_spec_workload(SPEC, 40), rate=20.0, concurrency=8,
        kill_after_s=1.0, ready_timeout_s=3, recovery_timeout_s=3, max_seconds=60, max_restarts=0))
    assert report["passed"] is False
    assert {c["name"]: c["passed"] for c in report["checks"]}["worker_became_ready"] is False


def test_the_kill_must_fall_inside_the_load_window_and_output_is_never_overwritten(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="inside the load window"):
        run_experiment(tmp_path, monkeypatch, kill_after_s=60)
    run_experiment(tmp_path, monkeypatch, output_dir=tmp_path / "once", plan=build_spec_workload(SPEC, 80),
                   kill_after_s=1.5, rate=20.0)
    with pytest.raises(FileExistsError):
        run_experiment(tmp_path, monkeypatch, output_dir=tmp_path / "once")
