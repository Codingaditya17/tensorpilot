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
    assert "quantization nf4" in result.stdout and "HARD CAP:        214 minutes" in result.stdout
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



def lifetime_check(life, budget):
    environment = {**os.environ, "SESSION_BUDGET_MINUTES": str(budget)}
    environment.pop("TENSORPILOT_POD_MAX_MINUTES", None)
    if life is not None:
        environment["TENSORPILOT_POD_MAX_MINUTES"] = str(life)
    return subprocess.run(["bash", "-c", f'source "{SCRIPT}"; pod_lifetime_check'], env=environment,
                          capture_output=True, text=True)


def test_a_pod_whose_lifetime_backstop_would_end_the_session_early_is_refused():
    short = lifetime_check(120, 360)
    assert short.returncode == 1 and "container would exit mid-stage" in short.stderr and "at least 420" in short.stderr
    assert lifetime_check(419, 360).returncode == 1
    assert lifetime_check(420, 360).returncode == 0
    assert lifetime_check(None, 360).returncode == 0, "outside the experiment image there is no backstop to check"


def test_the_lifetime_guard_stops_a_run_and_a_plan_before_any_work(tmp_path):
    env = {"TENSORPILOT_POD_MAX_MINUTES": "120", "SESSION_BUDGET_MINUTES": "360", "TENSORPILOT_API_KEY": KEY,
           "V10_TEST_MODE": "1", "V10_SKIP_GPU_CHECKS": "1"}
    run_result = run(["0.5b-serial"], tmp_path, env=env)
    assert run_result.returncode == 3 and "backstop" in run_result.stderr and not stage_dirs(tmp_path)
    plan = run(["--plan", "0.5b-serial"], tmp_path, env=env)
    assert plan.returncode == 3 and "backstop" in plan.stderr
