"""deploy/runpod/run_7b_session.sh: the order, the gates and the stop-on-anything behavior, against stand-in stage commands.

The stand-ins record each call and fail on request, so what is tested is the script's own logic: what runs, in what order, and
that nothing runs after a failure. The real stages are tested elsewhere (test_scale_session.py). Runs on macOS and Linux.
"""
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "deploy" / "runpod" / "run_7b_session.sh"
DIGEST = "sha256:7e65d7ca7cdd4fa7f636a74092550b9353ffa5923796b6a993f625858c446e56"
STAGES = ["probe-7b", "7b-bf16-prod-serial", "7b-bf16-prod", "faults-7b", "vllm-7b-bf16-prod", "quality-7b"]

FAKE_SCALE = r'''#!/usr/bin/env bash
set -eu
echo "scale $1 pack=${PREFILL_PACK_TOKENS:-unset}" >> "$CALLS"
[[ "${FAIL_AT:-}" == "$1" ]] && { echo "stand-in failure of $1" >&2; exit 1; }
mkdir -p "$TENSORPILOT_V10_SESSION"
printf '{"stage": "%s", "elapsed_min": %s, "cap_min": 30, "exit_status": 0}\n' "$1" "${ELAPSED:-10}" >> "$TENSORPILOT_V10_SESSION/ledger.jsonl"
if [[ "$1" == probe-7b ]]; then
  d="$TENSORPILOT_V10_SESSION/probe-7b-20261101T000000Z"; mkdir -p "$d"
  cp "$PROBE_JSON" "$d/divergence-probe.json"
fi
touch "$TENSORPILOT_V10_SESSION/$1-20261101T000000Z.tar.gz"
'''
FAKE_INSTALL = r'''#!/usr/bin/env bash
echo "install ${1:-run}" >> "$CALLS"
[[ "${FAIL_AT:-}" == "install ${1:-run}" ]] && { echo "stand-in vLLM failure" >&2; exit 1; }
exit 0
'''
FAKE_SMI = r'''#!/usr/bin/env bash
echo "NVIDIA GeForce RTX 4090, ${DRIVER:-595.91.07}"
'''


def probe(**over):
    base = {"passed": True, "prompts": 32, "exact_matches": 31, "near_tie": 1, "unexplained": 0,
            "settings": {"capacity": 32, "granted_capacity": 32, "prefill_pack_tokens": 512}}
    for key, value in over.items():
        (base["settings"] if key in base["settings"] else base)[key] = value
    return base


@pytest.fixture
def pod(tmp_path):
    binary = tmp_path / "bin"
    binary.mkdir()
    for name, body in (("scale_session.sh", FAKE_SCALE), ("install_vllm.sh", FAKE_INSTALL), ("nvidia-smi", FAKE_SMI)):
        (binary / name).write_text(body)
        (binary / name).chmod(0o755)
    (tmp_path / "probe.json").write_text(json.dumps(probe()))
    session = tmp_path / "session"
    env = {"PATH": f"{binary}:/usr/bin:/bin:/usr/local/bin", "HOME": str(tmp_path), "CALLS": str(tmp_path / "calls.txt"),
           "PROBE_JSON": str(tmp_path / "probe.json"), "TENSORPILOT_V10_SESSION": str(session),
           # exactly the variables the script documents, and nothing else
           "TENSORPILOT_API_KEY": "k" * 32, "TENSORPILOT_IMAGE_REF": f"ghcr.io/x/cuda-experiment@{DIGEST}",
           "RUNPOD_POD_ID": "pod-1", "SESSION_BUDGET_MINUTES": "250", "PREFILL_PACK_TOKENS": "512",
           "TENSORPILOT_POD_MAX_MINUTES": "330"}
    return tmp_path, env


def run(pod, **changes):
    tmp, env = pod
    env = {**env, **{k: v for k, v in changes.items() if v is not None}}
    for key, value in changes.items():
        if value is None:
            env.pop(key, None)
    return subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True, cwd=tmp)


def calls(pod):
    path = pod[0] / "calls.txt"
    return path.read_text().splitlines() if path.exists() else []


def test_the_successful_path_runs_the_six_stages_in_order_and_the_optional_probe_unpacked(pod):
    done = run(pod)
    assert done.returncode == 0, done.stderr
    assert calls(pod) == ["scale probe-7b pack=512", "scale 7b-bf16-prod-serial pack=512", "scale 7b-bf16-prod pack=512",
                          "scale faults-7b pack=512", "install --plan", "install run", "scale vllm-7b-bf16-prod pack=512",
                          "scale quality-7b pack=512", "scale probe-0.5b pack=0"]
    assert "ALL STAGES DONE" in done.stdout and "TERMINATE THE POD" in done.stdout


@pytest.mark.parametrize("failing", STAGES + ["install --plan", "install run"])
def test_nothing_runs_after_any_failure_and_the_exit_status_survives(pod, failing):
    done = run(pod, FAIL_AT=failing)
    assert done.returncode != 0
    order = ["scale " + s for s in STAGES[:4]] + ["install --plan", "install run", "scale vllm-7b-bf16-prod", "scale quality-7b"]
    wanted = order[:order.index(failing if failing.startswith("install") else "scale " + failing) + 1]
    assert [c.rsplit(" pack=", 1)[0] for c in calls(pod)] == wanted
    assert "STOPPED at:" in done.stderr and "STILL BILLING" in done.stderr
    assert "probe-0.5b" not in " ".join(calls(pod))


def test_a_failing_install_plan_stops_the_chain_even_though_the_install_itself_would_pass(pod):
    """The bug in the first draft: `plan && install` swallows the plan's failure under set -e."""
    done = run(pod, FAIL_AT="install --plan")
    assert done.returncode != 0 and "install run" not in calls(pod) and "scale vllm-7b-bf16-prod pack=512" not in calls(pod)


@pytest.mark.parametrize("bad", [probe(unexplained=1, passed=False), probe(granted_capacity=31), probe(prefill_pack_tokens=2048),
                                 probe(prompts=16), probe(passed=False)])
def test_the_probe_gate_stops_the_session_before_the_serial_baseline(pod, bad):
    (pod[0] / "probe.json").write_text(json.dumps(bad))
    done = run(pod)
    assert done.returncode != 0 and "GATE FAIL" in done.stdout
    assert calls(pod) == ["scale probe-7b pack=512"] and "STOPPED at: probe-7b gate" in done.stderr


def test_the_probe_gate_prints_what_it_measured(pod):
    done = run(pod)
    assert "probe-7b: 31/32 identical, 1 near-tie, 0 unexplained, 32/32 slots, packing 512: GATE PASS" in done.stdout


@pytest.mark.parametrize("change", [{"TENSORPILOT_IMAGE_REF": "ghcr.io/x/cuda-experiment:v1.0.0-rc2-2"},
                                    {"TENSORPILOT_IMAGE_REF": "ghcr.io/x/c@sha256:" + "0" * 64},
                                    {"RUNPOD_POD_ID": None}, {"TENSORPILOT_API_KEY": None}, {"PREFILL_PACK_TOKENS": "2048"},
                                    {"PREFILL_PACK_TOKENS": None}, {"SESSION_BUDGET_MINUTES": "225"},
                                    {"SESSION_BUDGET_MINUTES": "300"}, {"SESSION_BUDGET_MINUTES": None},
                                    {"TENSORPILOT_POD_MAX_MINUTES": "120"}, {"TENSORPILOT_POD_MAX_MINUTES": None},
                                    {"DRIVER": "570.1"}, {"V10_TEST_MODE": "1"}])
def test_a_wrong_environment_is_refused_before_any_stage_runs(pod, change):
    done = run(pod, **change)
    assert done.returncode != 0 and calls(pod) == [] and "refused:" in done.stderr


def test_missing_tools_are_refused_up_front(pod):
    (pod[0] / "bin" / "install_vllm.sh").unlink()
    done = run(pod)
    assert done.returncode != 0 and calls(pod) == [] and "install_vllm.sh is not on PATH" in done.stderr


def test_the_optional_probe_runs_only_if_the_ledger_leaves_room_for_its_cap(pod):
    done = run(pod, ELAPSED="40")          # six stages at 40 minutes: 240 used of 250, no room for a 12-minute stage
    assert done.returncode == 0 and "probe-0.5b" not in " ".join(calls(pod))
    assert "optional unpacked 0.5B probe skipped: 240 of 250 ledger minutes used" in done.stdout


def test_the_documented_invocation_preserves_the_scripts_failure_through_tee(pod):
    tmp, env = pod
    command = f'set -o pipefail; bash "{SCRIPT}" 2>&1 | tee "{tmp}/run.log"; exit ${{PIPESTATUS[0]}}'
    failed = subprocess.run(["bash", "-c", command], env={**env, "FAIL_AT": "7b-bf16-prod"}, capture_output=True, text=True, cwd=tmp)
    assert failed.returncode == 1 and "stand-in failure of 7b-bf16-prod" in (tmp / "run.log").read_text()
    passed = subprocess.run(["bash", "-c", command], env={**env, "FAIL_AT": ""}, capture_output=True, text=True, cwd=tmp)
    assert passed.returncode == 0
    # and the trap: without pipefail the pipeline would have hidden the failure, which is why the invocation sets it
    hidden = subprocess.run(["bash", "-c", f'bash "{SCRIPT}" 2>&1 | tee "{tmp}/h.log"'], env={**env, "FAIL_AT": "faults-7b"},
                            capture_output=True, text=True, cwd=tmp)
    assert hidden.returncode == 0


def test_the_script_needs_no_variables_beyond_the_documented_ones(pod):
    """`pod` builds the environment from scratch (no os.environ), so passing means nothing undocumented is read."""
    tmp, env = pod
    allowed = {"PATH", "HOME", "CALLS", "PROBE_JSON", "TENSORPILOT_V10_SESSION", "TENSORPILOT_API_KEY", "TENSORPILOT_IMAGE_REF",
               "RUNPOD_POD_ID", "SESSION_BUDGET_MINUTES", "PREFILL_PACK_TOKENS", "TENSORPILOT_POD_MAX_MINUTES"}
    assert set(env) == allowed and "PRICE_PER_HOUR_USD" not in env
    assert run(pod).returncode == 0
    header = SCRIPT.read_text().split("set -Eeuo pipefail")[0]
    for name in ("TENSORPILOT_API_KEY", "TENSORPILOT_IMAGE_REF", "RUNPOD_POD_ID", "SESSION_BUDGET_MINUTES", "PREFILL_PACK_TOKENS",
                 "TENSORPILOT_POD_MAX_MINUTES", "TENSORPILOT_RESULTS_ROOT"):
        assert name in header, f"{name} is read by the script but not documented in its header"


def test_the_pinned_digest_is_the_published_rc2_3_digest_and_the_session_budget_is_the_planned_one():
    from tensorpilot import suite
    assert DIGEST in SCRIPT.read_text()
    plan = suite.budget_plan(0.91, 8.57, {"7b": suite.SESSIONS["7b"]})["sessions"]["7b"]
    assert plan["session_budget_minutes"] == 250 and plan["ledger_refuses_on_the_expected_path"] is None
    assert suite.SESSIONS["7b"]["stages"] == STAGES


def test_the_copy_in_the_document_is_the_script():
    text = (ROOT / "docs" / "v10-production-evaluation.md").read_text()
    assert SCRIPT.read_text().strip() in text
