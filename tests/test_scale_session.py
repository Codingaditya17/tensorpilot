"""The stage runner (deploy/runpod/scale_session.sh): refusals, caps, ledger, evidence and cleanup, via stub engines.

The script targets a Linux Pod. It needs GNU timeout and bash 4, so these tests skip on macOS and run in CI and in the
experiment image. Nothing here touches a GPU; test mode is the only way the script ever skips a GPU check.
"""
import json
import os
import shutil
import socket
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "deploy" / "runpod" / "scale_session.sh"
STUB_WORKER, STUB_VLLM = ROOT / "tests" / "stub_worker.py", ROOT / "tests" / "stub_vllm.py"
KEY = "scale-session-test-key-0123456789"
REVISION = "a" * 40

pytestmark = pytest.mark.skipif(
    shutil.which("timeout") is None or subprocess.run(["bash", "-c", "mapfile -t x </dev/null"]).returncode != 0,
    reason="needs GNU timeout and bash 4 (a Linux host)")


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def run(args, tmp_path, *, env=None, timeout=240):
    environment = {**os.environ, "TENSORPILOT_V10_SESSION": str(tmp_path / "session"),
                   "TENSORPILOT_CACHE_DIR": str(tmp_path / "cache"), "PYTHONDONTWRITEBYTECODE": "1"}
    # The experiment image sets TENSORPILOT_POD_MAX_MINUTES; only the lifetime tests should see it.
    for name in ("TENSORPILOT_API_KEY", "SESSION_BUDGET_MINUTES", "V10_TEST_MODE", "PRICE_PER_HOUR_USD",
                 "TENSORPILOT_POD_MAX_MINUTES"):
        environment.pop(name, None)
    environment.update(env or {})
    return subprocess.run(["bash", str(SCRIPT), *args], env=environment, capture_output=True, text=True,
                          timeout=timeout, cwd=ROOT)


def engine_env(tmp_path, port, command, config, **extra):
    return {"V10_TEST_MODE": "1", "V10_SKIP_GPU_CHECKS": "1", "V10_PORT": str(port), "V10_WORKER_CMD": command,
            "V10_CONFIG_OVERRIDE": str(config), "TENSORPILOT_API_KEY": KEY, "SESSION_BUDGET_MINUTES": "60", **extra}


def mini_config(tmp_path, *, api=None, requests=6, max_tokens=4, cap=1):
    config = {"schema": "tensorpilot.gpu-experiments/v1", "target": {"model": "stub", "revision": REVISION},
              "assumptions": {"status": "placeholder_unmeasured", "decode_tokens_per_s": 60, "prefill_tokens_per_s": 2000},
              "max_session_minutes": 5, "telemetry": {"interval_s": 0.2},
              "experiments": [{"id": "mini", "kind": "closed_loop", "requests": requests, "prompt_repeats": 1,
                               "max_tokens": max_tokens, "warmup_requests": 1, "timeout_s": 30, "max_minutes": cap}]}
    if api:
        config["api"], config["model_name"] = "openai", "stub-model"
    path = tmp_path / "mini.json"
    path.write_text(json.dumps(config))
    return path


def ledger(tmp_path):
    path = tmp_path / "session" / "ledger.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def stage_dirs(tmp_path):
    session = tmp_path / "session"
    return sorted(p for p in session.iterdir() if p.is_dir()) if session.exists() else []


def stub_alive():
    return subprocess.run(["pgrep", "-f", "stub_worker.py|stub_vllm.py"], capture_output=True).returncode == 0


# ----- plan, list and refusals: nothing starts -------------------------------------------------------------

def test_list_shows_every_stage_with_its_expected_minutes_and_hard_cap(tmp_path):
    result = run(["--list"], tmp_path)
    assert result.returncode == 0
    for stage in ("0.5b-batch", "7b-bf16-batch", "14b-nf4-batch", "vllm-7b-bf16", "quality-14b", "faults-7b", "TOTAL"):
        assert stage in result.stdout
    assert "hard cap" in result.stdout


def test_plan_prints_the_cap_and_needs_no_secret(tmp_path):
    result = run(["--plan", "14b-nf4-batch"], tmp_path)
    assert result.returncode == 0
    assert "Qwen/Qwen2.5-14B-Instruct @ cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8" in result.stdout
    assert "quantization nf4" in result.stdout and "HARD CAP:        220 minutes" in result.stdout
    assert KEY not in result.stdout and not (tmp_path / "session" / "ledger.jsonl").exists()


def test_plan_says_when_a_stage_would_exceed_the_budget(tmp_path):
    over = run(["--plan", "14b-nf4-batch"], tmp_path, env={"SESSION_BUDGET_MINUTES": "100"})
    assert over.returncode == 3 and "OVER budget" in over.stdout
    within = run(["--plan", "0.5b-serial"], tmp_path, env={"SESSION_BUDGET_MINUTES": "100"})
    assert within.returncode == 0 and "WITHIN budget" in within.stdout


def test_unknown_stages_options_and_missing_arguments_are_refused(tmp_path):
    assert run(["no-such-stage"], tmp_path, env={"TENSORPILOT_API_KEY": KEY, "SESSION_BUDGET_MINUTES": "9"}).returncode == 2
    assert run(["--plan"], tmp_path).returncode == 2
    assert run(["--bogus"], tmp_path).returncode == 2
    usage = run([], tmp_path)
    assert usage.returncode == 0 and "scale_session.sh --plan STAGE" in usage.stdout


def test_a_run_needs_the_key_and_a_budget_before_it_does_anything(tmp_path):
    missing_key = run(["0.5b-serial"], tmp_path, env={"SESSION_BUDGET_MINUTES": "60"})
    assert missing_key.returncode == 2 and "TENSORPILOT_API_KEY is required" in missing_key.stderr
    bad_budget = run(["0.5b-serial"], tmp_path, env={"TENSORPILOT_API_KEY": KEY, "SESSION_BUDGET_MINUTES": "soon"})
    assert bad_budget.returncode == 2 and "SESSION_BUDGET_MINUTES" in bad_budget.stderr
    assert not stage_dirs(tmp_path)


def test_a_host_without_a_gpu_is_refused_before_any_work(tmp_path):
    result = run(["0.5b-serial"], tmp_path, env={"TENSORPILOT_API_KEY": KEY, "SESSION_BUDGET_MINUTES": "60"})
    assert result.returncode == 2 and "nvidia-smi not found" in result.stderr
    assert not stage_dirs(tmp_path) and not ledger(tmp_path)


def test_a_stage_that_would_exceed_the_budget_is_refused_without_starting_anything(tmp_path):
    session = tmp_path / "session"
    session.mkdir()
    (session / "ledger.jsonl").write_text(json.dumps({"stage": "earlier", "elapsed_min": 130.0}) + "\n")
    result = run(["0.5b-serial"], tmp_path, env={"TENSORPILOT_API_KEY": KEY, "SESSION_BUDGET_MINUTES": "140",
                                                  "V10_TEST_MODE": "1", "V10_SKIP_GPU_CHECKS": "1"})
    assert result.returncode == 3 and "exceeds the 140 minute session budget" in result.stderr
    assert not stage_dirs(tmp_path) and len(ledger(tmp_path)) == 1


# ----- a whole stage through a stub engine -----------------------------------------------------------------

def stage_with_stub(tmp_path, stage="0.5b-serial", **overrides):
    port = free_port()
    command = f"{sys.executable} {STUB_WORKER} --port {port}"
    env = engine_env(tmp_path, port, command, mini_config(tmp_path), **overrides)
    return run([stage], tmp_path, env=env), port


def test_a_stage_runs_end_to_end_and_leaves_checksummed_evidence_and_a_ledger_entry(tmp_path):
    result, _ = stage_with_stub(tmp_path)
    assert result.returncode == 0, result.stderr[-1500:]
    (directory,) = stage_dirs(tmp_path)
    for name in ("smoke.json", "engine.log", "bench/manifest.json", "bench-run.log"):
        assert (directory / name).exists(), name
    manifest = json.loads((directory / "bench" / "manifest.json").read_text())
    assert manifest["experiments"]["mini"]["status"] == "completed"
    archive = Path(str(directory) + ".tar.gz")
    checksum = Path(str(archive) + ".sha256").read_text().split()[0]
    import hashlib
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == checksum
    with tarfile.open(archive) as tar:
        assert any(name.endswith("bench/manifest.json") for name in tar.getnames())
        assert KEY not in b"".join(tar.extractfile(m).read() for m in tar if m.isfile()).decode(errors="ignore")
    (entry,) = ledger(tmp_path)
    assert entry["stage"] == "0.5b-serial" and entry["exit_status"] == 0 and entry["sha256"] == checksum
    assert entry["elapsed_min"] > 0 and entry["cap_min"] == 21
    assert "THE POD IS STILL BILLING" in result.stderr and "TEST MODE" in result.stderr
    assert not stub_alive(), "the engine process must be stopped when the stage ends"


def test_a_batching_stage_measures_batch_invariance_before_the_benchmark(tmp_path):
    result, _ = stage_with_stub(tmp_path, stage="0.5b-batch", SESSION_BUDGET_MINUTES="500")
    assert result.returncode == 0, result.stderr[-1500:]
    (directory,) = stage_dirs(tmp_path)
    report = json.loads((directory / "invariance.json").read_text())
    assert report["schema"] == "tensorpilot.batch-invariance/v1" and report["prompts"] == 32
    assert report["together_vs_solo"]["exact_matches"] == 32, "the deterministic stub must be batch-invariant"
    assert "batch-invariance" in result.stderr
    (entry,) = ledger(tmp_path)
    assert entry["cap_min"] == 124, "the invariance check's ten minutes are inside the stage's hard cap"


def test_serial_and_vllm_stages_do_not_run_the_invariance_check(tmp_path):
    result, _ = stage_with_stub(tmp_path)
    assert result.returncode == 0
    (directory,) = stage_dirs(tmp_path)
    assert not (directory / "invariance.json").exists()


def test_the_ledger_accumulates_and_prices_the_session_when_a_price_is_given(tmp_path):
    first, _ = stage_with_stub(tmp_path, PRICE_PER_HOUR_USD="0.90")
    assert first.returncode == 0 and "at $0.90/h, stage time only" in first.stderr
    second, _ = stage_with_stub(tmp_path)
    assert second.returncode == 0 and len(ledger(tmp_path)) == 2


def test_a_worker_that_never_starts_fails_the_stage_but_still_writes_evidence(tmp_path):
    port = free_port()
    env = engine_env(tmp_path, port, f"{sys.executable} -c exit(3)", mini_config(tmp_path))
    result = run(["0.5b-serial"], tmp_path, env=env)
    assert result.returncode == 1 and "exited before" in result.stderr
    (entry,) = ledger(tmp_path)
    assert entry["exit_status"] == 1 and Path(entry["archive"]).exists()
    assert not stub_alive()


def test_the_hard_cap_stops_a_runaway_stage_and_its_evidence_is_kept(tmp_path):
    port = free_port()
    config = mini_config(tmp_path, requests=2000, max_tokens=64, cap=5)  # about 4 minutes of work
    env = engine_env(tmp_path, port, f"{sys.executable} {STUB_WORKER} --port {port}", config, V10_CAP_MINUTES_OVERRIDE="0.15")
    result = run(["0.5b-serial"], tmp_path, env=env, timeout=180)
    (entry,) = ledger(tmp_path)
    assert entry["elapsed_min"] < 1.5, "the cap, not the workload, decided when the stage ended"
    assert Path(entry["archive"]).exists() and entry["cap_min"] == 0.15
    manifest = tmp_path / "session"
    found = list(manifest.glob("*/bench/manifest.json"))
    if found and result.returncode == 0:  # the runner's own soft stop won the race
        assert json.loads(found[0].read_text())["experiments"]["mini"]["status"] in ("aborted", "completed")
    else:
        assert result.returncode in (124, 137, 143)
    assert not stub_alive()


def test_an_archive_containing_the_api_key_is_refused(tmp_path):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "engine.log").write_text(f"oops {KEY}\n")
    snippet = f'source "{SCRIPT}"; archive_stage "{evidence}"'
    result = subprocess.run(["bash", "-c", snippet], env={**os.environ, "TENSORPILOT_API_KEY": KEY},
                            capture_output=True, text=True)
    assert result.returncode == 1 and "refusing to archive" in result.stderr
    assert not (tmp_path / "evidence.tar.gz").exists()


def test_the_vllm_path_passes_the_key_by_environment_and_uses_the_openai_client(tmp_path):
    port = free_port()
    config = mini_config(tmp_path, api="openai")
    env = engine_env(tmp_path, port, "unused", config, V10_VLLM_CMD=f"{sys.executable} {STUB_VLLM} --port {port}",
                     SESSION_BUDGET_MINUTES="500")  # this stage's real hard cap is 115 minutes
    result = run(["vllm-7b-bf16"], tmp_path, env=env)
    assert result.returncode == 0, result.stderr[-1500:]
    (directory,) = stage_dirs(tmp_path)
    smoke = json.loads((directory / "smoke.json").read_text())
    assert smoke["summary"]["successful_requests"] == 5 and smoke["metadata"]["server"]["service"] == "openai-compatible"
    manifest = json.loads((directory / "bench" / "manifest.json").read_text())
    assert manifest["api"] == "openai" and manifest["experiments"]["mini"]["status"] == "completed"
    assert not stub_alive()


def test_a_crash_stage_kills_the_worker_and_audits_the_journal(tmp_path):
    port = free_port()
    env = engine_env(tmp_path, port, f"{sys.executable} {STUB_WORKER} --port {port}", mini_config(tmp_path))
    result = run(["faults-0.5b"], tmp_path, env=env, timeout=300)
    assert result.returncode == 0, result.stderr[-2000:]
    (directory,) = stage_dirs(tmp_path)
    report = json.loads((directory / "fault" / "fault-report.json").read_text())
    assert report["passed"] is True and report["kill"]["pid"]
    assert KEY not in (directory / "fault" / "fault-report.json").read_text()
    assert not stub_alive()



# ----- the Pod's lifetime backstop: judged against the real remaining time ---------------------------------

def write_deadline(tmp_path, minutes_from_now):
    import time
    path = tmp_path / "deadline"
    path.write_text(str(time.time() + minutes_from_now * 60))
    return path


def lifetime_check(tmp_path, remaining_min, *, stage="0.5b-serial", env=None):
    environment = {**os.environ, "TENSORPILOT_POD_DEADLINE_FILE": str(tmp_path / "deadline")}
    environment.pop("TENSORPILOT_POD_MAX_MINUTES", None)
    environment.update(env or {})
    if remaining_min is not None:
        write_deadline(tmp_path, remaining_min)
    snippet = f'source "{SCRIPT}"; eval "$(python3 -m tensorpilot.suite --stage-env {stage})"; pod_lifetime_check'
    return subprocess.run(["bash", "-c", snippet], env=environment, capture_output=True, text=True, cwd=ROOT)


def test_a_stage_is_refused_unless_the_real_remaining_lifetime_covers_its_cap_one_load_limit_and_the_archive(tmp_path):
    # 0.5b-serial: cap 21 + 15 (the ready timeout is 900 s) + 12 to archive = 48 minutes.
    short = lifetime_check(tmp_path, 47)
    assert short.returncode == 1 and "needs about 48" in short.stderr and "exit mid-stage" in short.stderr
    ok = lifetime_check(tmp_path, 50)
    assert ok.returncode == 0 and "needs about 48" in ok.stderr
    assert lifetime_check(tmp_path, -5).returncode == 1


def test_elapsed_time_counts_against_the_lifetime_so_a_late_stage_is_refused_even_if_the_pod_was_long_enough_at_boot(tmp_path):
    # 0.5b-batch needs 124 cap + 15 + 12 = 151 minutes.
    late = lifetime_check(tmp_path, 130, stage="0.5b-batch")
    assert late.returncode == 1 and "needs about 151" in late.stderr
    assert lifetime_check(tmp_path, 155, stage="0.5b-batch").returncode == 0


def test_no_deadline_file_is_fine_outside_the_experiment_image_but_fails_closed_inside_it(tmp_path):
    outside = lifetime_check(tmp_path, None)
    assert outside.returncode == 0
    inside = lifetime_check(tmp_path, None, env={"TENSORPILOT_POD_MAX_MINUTES": "120"})
    assert inside.returncode == 1 and "cannot be verified" in inside.stderr
    (tmp_path / "deadline").write_text("not a number")
    garbled = lifetime_check(tmp_path, None, env={"TENSORPILOT_POD_MAX_MINUTES": "120"})
    assert garbled.returncode == 1 and "unreadable" in garbled.stderr


def test_the_lifetime_guard_stops_a_run_and_a_plan_before_any_work(tmp_path):
    deadline = write_deadline(tmp_path, 30)
    env = {"TENSORPILOT_POD_DEADLINE_FILE": str(deadline), "SESSION_BUDGET_MINUTES": "360", "TENSORPILOT_API_KEY": KEY,
           "V10_TEST_MODE": "1", "V10_SKIP_GPU_CHECKS": "1"}
    run_result = run(["0.5b-serial"], tmp_path, env=env)
    assert run_result.returncode == 3 and "lifetime backstop" in run_result.stderr and not stage_dirs(tmp_path)
    plan = run(["--plan", "0.5b-serial"], tmp_path, env=env)
    assert plan.returncode == 3 and "lifetime backstop" in plan.stderr


# ----- interruption: the partial evidence is archived and the engine is stopped ----------------------------

def start_long_stage(tmp_path, *, worker=None, cap="10"):
    port = free_port()
    command = worker or f"{sys.executable} {STUB_WORKER} --port {port}"
    config = mini_config(tmp_path, requests=2000, max_tokens=64, cap=9)  # minutes of work: far longer than any test waits
    environment = {**os.environ, "TENSORPILOT_V10_SESSION": str(tmp_path / "session"),
                   "TENSORPILOT_CACHE_DIR": str(tmp_path / "cache"), "PYTHONDONTWRITEBYTECODE": "1",
                   **engine_env(tmp_path, port, command, config, V10_CAP_MINUTES_OVERRIDE=cap, SESSION_BUDGET_MINUTES="500")}
    environment.pop("TENSORPILOT_POD_MAX_MINUTES", None)
    return subprocess.Popen(["bash", str(SCRIPT), "0.5b-serial"], env=environment, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, cwd=ROOT)


def wait_for(predicate, seconds=90):
    import time
    end = time.time() + seconds
    while time.time() < end:
        if predicate():
            return True
        time.sleep(0.5)
    return False


def benchmark_started(tmp_path):
    return any(tmp_path.glob("session/*/bench/*/*.requests.jsonl"))


@pytest.mark.parametrize("name,code", [("SIGTERM", 143), ("SIGINT", 130), ("SIGHUP", 129)])
def test_an_interrupted_stage_stops_its_children_archives_the_partial_evidence_and_records_it(tmp_path, name, code):
    import signal
    process = start_long_stage(tmp_path)
    try:
        assert wait_for(lambda: benchmark_started(tmp_path)), "the benchmark never started"
        process.send_signal(getattr(signal, name))
        stdout, stderr = process.communicate(timeout=150)
    finally:
        if process.poll() is None:
            process.kill()
    assert process.returncode == code, stderr[-1500:]
    assert f"received {name}" in stderr and "partial evidence" in stderr and "interrupted" in stderr
    (entry,) = ledger(tmp_path)
    assert entry["exit_status"] == code and entry["sha256"] and Path(entry["archive"]).exists()
    assert entry["elapsed_min"] < 2, "an interrupt ended the stage; the cap and the workload did not"
    with tarfile.open(entry["archive"]) as tar:
        members = {m.name.split("/", 1)[1]: m for m in tar if m.isfile() and "/" in m.name}
        manifest = json.loads(tar.extractfile(members["bench/manifest.json"]).read())
        contents = b"".join(tar.extractfile(m).read() for m in members.values()).decode(errors="ignore")
    assert manifest["interrupted"] is True and "finished_at_utc" in manifest
    assert manifest["experiments"]["mini"]["status"] == "aborted" and manifest["experiments"]["mini"]["abort_reason"] == "interrupted"
    assert manifest["experiments"]["mini"]["partial_journals"], "the unfinished repeat's request journal is kept"
    assert "engine.log" in members and KEY not in contents
    assert not stub_alive(), "neither the engine nor the benchmark may outlive the stage"
    assert subprocess.run(["pgrep", "-f", "tensorpilot.gpu_bench|tensorpilot.load"], capture_output=True).returncode != 0


def test_a_stage_interrupted_while_the_engine_is_still_starting_is_archived_too(tmp_path):
    import signal
    process = start_long_stage(tmp_path, worker="sleep 300")
    try:
        assert wait_for(lambda: any(tmp_path.glob("session/*/engine.log")), 30)
        import time
        time.sleep(2)
        process.send_signal(signal.SIGTERM)
        _, stderr = process.communicate(timeout=60)
    finally:
        if process.poll() is None:
            process.kill()
    assert process.returncode == 143, stderr[-800:]
    (entry,) = ledger(tmp_path)
    assert entry["exit_status"] == 143 and Path(entry["archive"]).exists()
    assert subprocess.run(["pgrep", "-f", "sleep 300"], capture_output=True).returncode != 0


def test_the_memory_gate_failing_closed_stops_a_batching_stage_before_the_benchmark(tmp_path):
    port = free_port()
    env = engine_env(tmp_path, port, f"{sys.executable} {STUB_WORKER} --port {port}", mini_config(tmp_path),
                     V10_FORCE_MEMORY_GATE="1", SESSION_BUDGET_MINUTES="500")
    result = run(["0.5b-batch"], tmp_path, env=env)
    assert result.returncode == 1 and "memory gate failed" in result.stderr, result.stderr[-1200:]
    (directory,) = stage_dirs(tmp_path)
    gate = json.loads((directory / "memory-gate.json").read_text())
    assert gate["ok"] is False and gate["checks"][0]["name"] == "memory_information_present"
    assert not (directory / "bench").exists(), "no benchmark may run when the margin cannot be verified"
    (entry,) = ledger(tmp_path)
    assert entry["exit_status"] == 1 and Path(entry["archive"]).exists() and not stub_alive()


def test_the_lifetime_watchdog_stops_a_running_stage_and_archives_it_before_the_container_would_exit(tmp_path):
    import time
    deadline = write_deadline(tmp_path, 0.5)                    # the "container" ends 30 s from now
    port = free_port()
    config = mini_config(tmp_path, requests=2000, max_tokens=64, cap=9)
    env = {**os.environ, "TENSORPILOT_V10_SESSION": str(tmp_path / "session"), "TENSORPILOT_CACHE_DIR": str(tmp_path / "cache"),
           "PYTHONDONTWRITEBYTECODE": "1", "TENSORPILOT_POD_DEADLINE_FILE": str(deadline),
           "V10_LIFETIME_MARGIN_OVERRIDE_MIN": "0.1", "V10_SKIP_LIFETIME_PRECHECK": "1",
           **engine_env(tmp_path, port, f"{sys.executable} {STUB_WORKER} --port {port}", config, V10_CAP_MINUTES_OVERRIDE="10",
                        SESSION_BUDGET_MINUTES="500")}
    env.pop("TENSORPILOT_POD_MAX_MINUTES", None)
    started = time.time()
    result = subprocess.run(["bash", str(SCRIPT), "0.5b-serial"], env=env, capture_output=True, text=True, timeout=120, cwd=ROOT)
    assert result.returncode == 125, result.stderr[-1500:]
    assert "lifetime backstop is 0.1 minutes away" in result.stderr and "lifetime watchdog" in result.stderr
    assert time.time() - started < 40, "the watchdog, not the cap or the workload, ended the stage"
    (entry,) = ledger(tmp_path)
    assert entry["exit_status"] == 125 and Path(entry["archive"]).exists()
    with tarfile.open(entry["archive"]) as tar:
        manifests = [m for m in tar if m.name.endswith("bench/manifest.json")]
        assert manifests and json.loads(tar.extractfile(manifests[0]).read())["interrupted"] is True
    assert not stub_alive()


def test_the_watchdog_is_cancelled_when_a_stage_finishes_so_it_cannot_signal_a_later_command(tmp_path):
    deadline = write_deadline(tmp_path, 60)
    result, _ = stage_with_stub(tmp_path, TENSORPILOT_POD_DEADLINE_FILE=str(deadline), V10_SKIP_LIFETIME_PRECHECK="1")
    assert result.returncode == 0, result.stderr[-1200:]
    assert "lifetime watchdog" in result.stderr
    assert subprocess.run(["pgrep", "-f", "sleep 2[0-9][0-9][0-9]"], capture_output=True).returncode != 0, \
        "the watchdog's sleep must not outlive the stage"


def test_the_real_quick_serial_stage_produces_evidence_that_quick_compare_can_read(tmp_path):
    """The real stage configuration against the stub worker: the verdict tool must parse what the stage actually writes."""
    from tensorpilot import quick_compare
    port = free_port()
    env = {**engine_env(tmp_path, port, f"{sys.executable} {STUB_WORKER} --port {port}", "unused"), "SESSION_BUDGET_MINUTES": "60"}
    env.pop("V10_CONFIG_OVERRIDE")
    result = run(["0.5b-quick-serial"], tmp_path, env=env, timeout=300)
    assert result.returncode == 0, result.stderr[-1500:]
    (directory,) = stage_dirs(tmp_path)
    facts = quick_compare.stage_facts(directory)
    assert set(facts["statuses"]) == {"cold-start", "quick-c1", "quick-c8"} and set(facts["statuses"].values()) == {"completed"}
    c1, c8 = facts["experiments"]["quick-c1"], facts["experiments"]["quick-c8"]
    assert (c1["requests"], c1["failed"]) == (24, 0) and (c8["requests"], c8["failed"]) == (64, 0)
    assert c1["tok_per_s"] > 0 and c8["tok_per_s"] > 0 and facts["interrupted"] is False
    verdict = quick_compare.judge(facts, facts)
    assert "evidence_present" not in {c["name"] for c in verdict["checks"]}, "both throughput checks were computed from real files"
