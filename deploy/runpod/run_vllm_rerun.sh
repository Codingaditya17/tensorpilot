#!/usr/bin/env bash
# Re-run only what the 7B session lost, on a fresh Pod, as user tensorpilot, inside tmux:
#   the vLLM comparison (vllm-7b-bf16-cmp), a TensorPilot control on the same card (7b-bf16-cmp), and quality-7b.
#
#   set -o pipefail
#   bash run_vllm_rerun.sh 2>&1 | tee ~/rerun.log ; echo "exit status: ${PIPESTATUS[0]}"
#
# Order: toolchain check -> vLLM install (plan, install, GPU kernel smoke) -> vllm-7b-bf16-cmp -> 7b-bf16-cmp -> quality-7b.
# vLLM goes first because it is what failed; everything before it takes seconds or is the install itself. It stops at the first
# failure of anything; every stage archives its evidence whether it passes or fails. THE POD KEEPS BILLING until you terminate it.
#
# Environment it needs. You export: TENSORPILOT_API_KEY, TENSORPILOT_IMAGE_REF, RUNPOD_POD_ID, SESSION_BUDGET_MINUTES (84 to 95),
# PREFILL_PACK_TOKENS (512); PRICE_PER_HOUR_USD is optional. The login profile gives it: PATH (scale_session.sh, install_vllm.sh,
# check_toolchain.sh), TENSORPILOT_POD_MAX_MINUTES (150 or more) and TENSORPILOT_RESULTS_ROOT. Nothing else is read.
set -Eeuo pipefail

# The digest of the published image that carries gcc and libc6-dev (v1.0.0-rc2-4). The script refuses to run until it is pinned.
EXPECTED_DIGEST="sha256:92721fab66c72deb382cf21248ca19cb43e18dc5147fc9a25deacaddf888612f"
STAGE="startup"
trap 'echo >&2; echo "*** STOPPED at: ${STAGE}. Evidence for every stage that ran, including this one, is archived under ${SESSION:-the session directory}." >&2; echo "*** THE POD IS STILL BILLING: copy the evidence, then TERMINATE the Pod." >&2' ERR

fail() { echo "refused: $*" >&2; exit 2; }
[[ -z "${V10_TEST_MODE:-}" ]] || fail "V10_TEST_MODE is set; that mode skips GPU checks and its results are not evidence"
[[ "$EXPECTED_DIGEST" =~ ^sha256:[0-9a-f]{64}$ ]] || fail "this script has no image digest pinned yet (EXPECTED_DIGEST); publish the image and pin it"
for name in TENSORPILOT_API_KEY RUNPOD_POD_ID; do [[ -n "${!name:-}" ]] || fail "$name is not exported"; done
[[ "${TENSORPILOT_IMAGE_REF:-}" == *"@${EXPECTED_DIGEST}" ]] || fail "TENSORPILOT_IMAGE_REF must end in @${EXPECTED_DIGEST}"
[[ "${PREFILL_PACK_TOKENS:-}" == 512 ]] || fail "PREFILL_PACK_TOKENS must be 512: the control must run TensorPilot as the archived 7b-bf16-prod did"
budget="${SESSION_BUDGET_MINUTES:-}"; pod_max="${TENSORPILOT_POD_MAX_MINUTES:-}"
[[ "$budget" =~ ^[0-9]+$ ]] && (( budget >= 84 && budget <= 95 )) ||
  fail "SESSION_BUDGET_MINUTES must be 84 to 95 (less and the ledger refuses a stage; more spends past the plan)"
[[ "$pod_max" =~ ^[0-9]+$ ]] && (( pod_max >= 150 )) ||
  fail "TENSORPILOT_POD_MAX_MINUTES must be at least 150 in the Pod's environment variables"
for tool in scale_session.sh install_vllm.sh check_toolchain.sh; do
  command -v "$tool" >/dev/null || fail "$tool is not on PATH (log in again so the profile is read)"
done
driver="$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
echo "GPU: ${driver}"
major="${driver##*, }"; major="${major%%.*}"
[[ "$major" =~ ^[0-9]+$ ]] && (( major >= 580 )) || fail "the driver must be 580 or newer for vLLM (this host: ${driver})"

export SESSION_BUDGET_MINUTES PREFILL_PACK_TOKENS TENSORPILOT_API_KEY
SESSION="${TENSORPILOT_V10_SESSION:-${TENSORPILOT_RESULTS_ROOT:-/workspace/results}/v10-session}"
export TENSORPILOT_V10_SESSION="$SESSION"

run() { STAGE="$1"; echo; echo "=== $(date -u +%H:%M:%S) stage $1"; scale_session.sh "$1"; }

# ----- 1. the thing that failed last time, proven in seconds, before any billed download -----
STAGE="toolchain check"
check_toolchain.sh

# ----- 2. vLLM: its own virtualenv; the installer repeats the toolchain check and launches a real Triton kernel on the GPU -----
STAGE="vLLM plan"
install_vllm.sh --plan
STAGE="vLLM installation"
install_vllm.sh

# ----- 3. the comparison: vLLM first, then TensorPilot on the same card, then the skipped quality evaluation -----
run vllm-7b-bf16-cmp
run 7b-bf16-cmp
run quality-7b

STAGE="finished"
echo; echo "=== ALL STAGES DONE ($(date -u +%H:%M:%S))"
cat "$SESSION/ledger.jsonl"
ls -lh "$SESSION"/*.tar.gz
echo "*** COPY THE EVIDENCE (the archives, ledger.jsonl and rerun.log) TO YOUR MAC, THEN TERMINATE THE POD (terminating, not stopping, ends billing)."
