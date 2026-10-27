"""deploy/runpod/run_vllm_rerun.sh: the order, the up-front refusals and the stop-on-anything behavior, against stand-in commands.

The script re-runs only what the 7B session lost (the vLLM comparison and the quality evaluation). The stand-ins record each call and
fail on request, so what is tested is the script's own logic. The real stages are tested in test_scale_session.py."""
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "deploy" / "runpod" / "run_vllm_rerun.sh"
DIGEST = "sha256:" + "c" * 64
CALLS_ORDER = ["check_toolchain", "install --plan", "install run", "scale vllm-startup-0.5b", "scale vllm-7b-bf16-cmp", "scale 7b-bf16-cmp",
               "scale quality-7b"]

FAKE_SCALE = r'''#!/usr/bin/env bash
set -eu
echo "scale $1 pack=${PREFILL_PACK_TOKENS:-unset}" >> "$CALLS"
[[ "${FAIL_AT:-}" == "scale $1" ]] && { echo "stand-in failure of $1" >&2; exit 1; }
mkdir -p "$TENSORPILOT_V10_SESSION"
printf '{"stage": "%s", "elapsed_min": 10, "cap_min": 30, "exit_status": 0}\n' "$1" >> "$TENSORPILOT_V10_SESSION/ledger.jsonl"
touch "$TENSORPILOT_V10_SESSION/$1-20261101T000000Z.tar.gz"
'''
FAKE_INSTALL = r'''#!/usr/bin/env bash
echo "install ${1:-run}" >> "$CALLS"
[[ "${FAIL_AT:-}" == "install ${1:-run}" ]] && { echo "stand-in vLLM failure" >&2; exit 1; }
exit 0
'''
FAKE_TOOLCHAIN = r'''#!/usr/bin/env bash
echo "check_toolchain" >> "$CALLS"
[[ "${FAIL_AT:-}" == "check_toolchain" ]] && { echo "check_toolchain: no C compiler" >&2; exit 2; }
exit 0
'''
FAKE_SMI = r'''#!/usr/bin/env bash
echo "NVIDIA GeForce RTX 4090, ${DRIVER:-595.91.07}"
'''


@pytest.fixture
def pod(tmp_path):
    binary = tmp_path / "bin"
    binary.mkdir()
    for name, body in (("scale_session.sh", FAKE_SCALE), ("install_vllm.sh", FAKE_INSTALL), ("check_toolchain.sh", FAKE_TOOLCHAIN),
                       ("nvidia-smi", FAKE_SMI)):
        (binary / name).write_text(body)
        (binary / name).chmod(0o755)
    pinned = tmp_path / "run_vllm_rerun.sh"          # the committed script, with a test digest pinned
    pinned.write_text(re.sub(r'EXPECTED_DIGEST="[^"]*"', f'EXPECTED_DIGEST="{DIGEST}"', SCRIPT.read_text(), count=1))
    env = {"PATH": f"{binary}:/usr/bin:/bin:/usr/local/bin", "HOME": str(tmp_path), "CALLS": str(tmp_path / "calls.txt"),
           "TENSORPILOT_V10_SESSION": str(tmp_path / "session"), "TENSORPILOT_API_KEY": "k" * 32,
           "TENSORPILOT_IMAGE_REF": f"ghcr.io/x/cuda-experiment@{DIGEST}", "RUNPOD_POD_ID": "pod-1",
           "SESSION_BUDGET_MINUTES": "97", "PREFILL_PACK_TOKENS": "512", "TENSORPILOT_POD_MAX_MINUTES": "165"}
    return tmp_path, env, pinned


def run(pod, script=None, **changes):
    tmp, env, pinned = pod
    env = {**env, **{k: v for k, v in changes.items() if v is not None}}
    for key, value in changes.items():
        if value is None:
            env.pop(key, None)
    return subprocess.run(["bash", str(script or pinned)], env=env, capture_output=True, text=True, cwd=tmp)


def calls(pod):
    path = pod[0] / "calls.txt"
    return [c.rsplit(" pack=", 1)[0] for c in path.read_text().splitlines()] if path.exists() else []


def test_the_toolchain_is_proved_first_then_vllm_is_installed_then_the_three_stages_run_in_order(pod):
    done = run(pod)
    assert done.returncode == 0, done.stderr
    assert calls(pod) == CALLS_ORDER
    assert "ALL STAGES DONE" in done.stdout and "TERMINATE THE POD" in done.stdout


def test_every_stage_runs_with_the_archived_runs_prefill_packing(pod):
    run(pod)
    text = (pod[0] / "calls.txt").read_text()
    assert all("pack=512" in line for line in text.splitlines() if line.startswith("scale"))


@pytest.mark.parametrize("failing", CALLS_ORDER)
def test_nothing_runs_after_any_failure(pod, failing):
    done = run(pod, FAIL_AT=failing)
    assert done.returncode != 0
    assert calls(pod) == CALLS_ORDER[:CALLS_ORDER.index(failing) + 1]
    assert "STOPPED at:" in done.stderr and "STILL BILLING" in done.stderr


def test_a_failed_early_startup_stage_stops_the_session_before_any_7b_stage(pod):
    """The point of the 0.5B stage: a third start-up failure must cost minutes, not the 7B download and benchmark."""
    done = run(pod, FAIL_AT="scale vllm-startup-0.5b")
    assert calls(pod)[-1] == "scale vllm-startup-0.5b" and "vllm-7b-bf16-cmp" not in " ".join(calls(pod))
    assert "STOPPED at: vllm-startup-0.5b" in done.stderr


def test_a_missing_compiler_stops_the_session_before_the_install_or_any_download(pod):
    """The failure this script exists for: it must show in the first seconds, with nothing billed beyond boot."""
    done = run(pod, FAIL_AT="check_toolchain")
    assert calls(pod) == ["check_toolchain"] and "STOPPED at: toolchain check" in done.stderr


def test_the_committed_script_refuses_until_an_image_digest_is_pinned(pod):
    done = run(pod, script=SCRIPT)
    assert done.returncode == 2 and "no image digest pinned" in done.stderr and calls(pod) == []


@pytest.mark.parametrize("change", [{"TENSORPILOT_IMAGE_REF": "ghcr.io/x/cuda-experiment:v1.0.0-rc2-3"},
                                    {"TENSORPILOT_IMAGE_REF": "ghcr.io/x/c@sha256:" + "0" * 64},
                                    {"RUNPOD_POD_ID": None}, {"TENSORPILOT_API_KEY": None}, {"PREFILL_PACK_TOKENS": "0"},
                                    {"PREFILL_PACK_TOKENS": None}, {"SESSION_BUDGET_MINUTES": "84"},
                                    {"SESSION_BUDGET_MINUTES": "250"}, {"SESSION_BUDGET_MINUTES": "106"}, {"SESSION_BUDGET_MINUTES": None},
                                    {"TENSORPILOT_POD_MAX_MINUTES": "150"}, {"TENSORPILOT_POD_MAX_MINUTES": None},
                                    {"DRIVER": "570.1"}, {"V10_TEST_MODE": "1"}])
def test_a_wrong_environment_is_refused_before_anything_runs(pod, change):
    done = run(pod, **change)
    assert done.returncode != 0 and calls(pod) == [] and "refused:" in done.stderr


@pytest.mark.parametrize("tool", ["scale_session.sh", "install_vllm.sh", "check_toolchain.sh"])
def test_missing_tools_are_refused_up_front(pod, tool):
    (pod[0] / "bin" / tool).unlink()
    done = run(pod)
    assert done.returncode != 0 and calls(pod) == [] and f"{tool} is not on PATH" in done.stderr


def test_the_documented_invocation_preserves_the_failure_through_tee(pod):
    tmp, env, pinned = pod
    command = f'set -o pipefail; bash "{pinned}" 2>&1 | tee "{tmp}/rerun.log"; exit ${{PIPESTATUS[0]}}'
    failed = subprocess.run(["bash", "-c", command], env={**env, "FAIL_AT": "scale vllm-7b-bf16-cmp"}, capture_output=True,
                            text=True, cwd=tmp)
    assert failed.returncode == 1 and "stand-in failure of vllm-7b-bf16-cmp" in (tmp / "rerun.log").read_text()


def test_the_script_documents_every_variable_it_reads():
    header = SCRIPT.read_text().split("set -Eeuo pipefail")[0]
    for name in ("TENSORPILOT_API_KEY", "TENSORPILOT_IMAGE_REF", "RUNPOD_POD_ID", "SESSION_BUDGET_MINUTES", "PREFILL_PACK_TOKENS",
                 "TENSORPILOT_POD_MAX_MINUTES", "TENSORPILOT_RESULTS_ROOT"):
        assert name in header, f"{name} is read by the script but not documented in its header"


def test_the_stage_order_and_budget_match_the_plan():
    from tensorpilot import suite
    plan = suite.budget_plan(0.91, 3.0, suite.RERUN_SESSIONS)["sessions"]["7b-vllm-rerun"]
    assert plan["stages"] == ["vllm-startup-0.5b", "vllm-7b-bf16-cmp", "7b-bf16-cmp", "quality-7b"]
    assert plan["session_budget_minutes"] == 97 and plan["ledger_refuses_on_the_expected_path"] is None
    text = SCRIPT.read_text()
    assert [m for m in re.findall(r"^run (\S+)$", text, re.M)] == plan["stages"]


DOC = ROOT / "docs" / "v10-vllm-startup-failures.md"


@pytest.mark.skipif(not DOC.exists(), reason="the image test job mounts tests, deploy and benchmarks, not docs")
def test_the_copy_in_the_document_is_the_script():
    assert SCRIPT.read_text().strip() in DOC.read_text()
