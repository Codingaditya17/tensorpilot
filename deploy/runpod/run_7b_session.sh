#!/usr/bin/env bash
# Run the whole 7B BF16 production session as one gated chain, on the Pod, as user tensorpilot, inside tmux.
#
#   set -o pipefail
#   bash run_7b_session.sh 2>&1 | tee ~/run7b.log ; echo "exit status: ${PIPESTATUS[0]}"
#
# Order: probe-7b (mandatory correctness gate) -> 7b-bf16-prod-serial -> 7b-bf16-prod -> faults-7b -> vLLM install -> vllm-7b-bf16-prod
# -> quality-7b -> (optional) the unpacked 0.5B divergence probe, only if the ledger budget still allows it.
#
# It stops at the first failure of anything: a stage, the probe gate, the vLLM installation or a check below. Every stage archives its own
# evidence whether it passes or fails, so a stop loses nothing. THE POD KEEPS BILLING until you terminate it.
#
# Environment it needs. You export: TENSORPILOT_API_KEY, TENSORPILOT_IMAGE_REF, RUNPOD_POD_ID, SESSION_BUDGET_MINUTES (250 to 260),
# PREFILL_PACK_TOKENS (512); PRICE_PER_HOUR_USD is optional (cost lines). The login profile gives it: PATH (scale_session.sh,
# install_vllm.sh), TENSORPILOT_POD_MAX_MINUTES (330 or more) and TENSORPILOT_RESULTS_ROOT. Nothing else is read.
set -Eeuo pipefail

EXPECTED_DIGEST="sha256:7e65d7ca7cdd4fa7f636a74092550b9353ffa5923796b6a993f625858c446e56"
STAGE="startup"
trap 'echo >&2; echo "*** STOPPED at: ${STAGE}. Evidence for every stage that ran, including this one, is archived under ${SESSION:-the session directory}." >&2; echo "*** THE POD IS STILL BILLING: copy the evidence, then TERMINATE the Pod." >&2' ERR

# ----- checks the stage runner does not make (it checks the GPU, disk, lifetime and budget per stage, and the vLLM driver only
# ----- when the vLLM stage starts, about two hours in; this fails early instead) -----
fail() { echo "refused: $*" >&2; exit 2; }
[[ -z "${V10_TEST_MODE:-}" ]] || fail "V10_TEST_MODE is set; that mode skips GPU checks and its results are not evidence"
for name in TENSORPILOT_API_KEY RUNPOD_POD_ID; do [[ -n "${!name:-}" ]] || fail "$name is not exported"; done
[[ "${TENSORPILOT_IMAGE_REF:-}" == *"@${EXPECTED_DIGEST}" ]] || fail "TENSORPILOT_IMAGE_REF must end in @${EXPECTED_DIGEST}"
[[ "${PREFILL_PACK_TOKENS:-}" == 512 ]] || fail "PREFILL_PACK_TOKENS must be 512 for this session"
budget="${SESSION_BUDGET_MINUTES:-}"; pod_max="${TENSORPILOT_POD_MAX_MINUTES:-}"
[[ "$budget" =~ ^[0-9]+$ ]] && (( budget >= 250 && budget <= 260 )) ||
  fail "SESSION_BUDGET_MINUTES must be 250 to 260 (225 would have the ledger refuse the vLLM stage; more than 260 spends past the plan)"
[[ "$pod_max" =~ ^[0-9]+$ ]] && (( pod_max >= 330 )) ||
  fail "TENSORPILOT_POD_MAX_MINUTES must be at least 330 in the Pod's environment variables"
command -v scale_session.sh >/dev/null || fail "scale_session.sh is not on PATH (log in again so the profile is read)"
command -v install_vllm.sh >/dev/null || fail "install_vllm.sh is not on PATH (log in again so the profile is read)"
driver="$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
echo "GPU: ${driver}"
major="${driver##*, }"; major="${major%%.*}"
[[ "$major" =~ ^[0-9]+$ ]] && (( major >= 580 )) || fail "the driver must be 580 or newer for vLLM (this host: ${driver})"

export SESSION_BUDGET_MINUTES PREFILL_PACK_TOKENS TENSORPILOT_API_KEY
SESSION="${TENSORPILOT_V10_SESSION:-${TENSORPILOT_RESULTS_ROOT:-/workspace/results}/v10-session}"
export TENSORPILOT_V10_SESSION="$SESSION"

run() { STAGE="$1"; echo; echo "=== $(date -u +%H:%M:%S) stage $1"; scale_session.sh "$1"; }

ledger_used() {
  python3 - "$SESSION/ledger.jsonl" <<'PY'
import json, sys
print(round(sum(json.loads(l)["elapsed_min"] for l in open(sys.argv[1]) if l.strip()), 1))
PY
}

# ----- 1. the mandatory correctness gate -----
run probe-7b
STAGE="probe-7b gate"
python3 - "$SESSION" <<'PY'
import glob, json, sys
r = json.load(open(sorted(glob.glob(sys.argv[1] + "/probe-7b-2*/divergence-probe.json"))[-1]))
s = r["settings"]
ok = (r["passed"] and r["unexplained"] == 0 and r["prompts"] == 32 and s["granted_capacity"] == s["capacity"] == 32
      and s["prefill_pack_tokens"] == 512)
print(f"probe-7b: {r['exact_matches']}/32 identical, {r['near_tie']} near-tie, {r['unexplained']} unexplained, "
      f"{s['granted_capacity']}/{s['capacity']} slots, packing {s['prefill_pack_tokens']}: GATE {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
PY

# ----- 2-4. serial baseline; continuous batching (its memory gate must grant all 48 slots); crash recovery -----
run 7b-bf16-prod-serial
run 7b-bf16-prod
run faults-7b

# ----- 5. vLLM: installed into its own virtualenv (the 7B weights are already in the shared cache, so nothing is downloaded twice) -----
STAGE="vLLM plan"
install_vllm.sh --plan
STAGE="vLLM installation"
install_vllm.sh
run vllm-7b-bf16-prod

# ----- 6. quality -----
run quality-7b

# ----- 7. optional: the unpacked 0.5B probe that explains the gate-session divergence (about 2 minutes), only if the ledger allows -----
used="$(ledger_used)"
if python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) + 12 <= float(sys.argv[2]) else 1)" "$used" "$SESSION_BUDGET_MINUTES"; then
  STAGE="probe-0.5b (optional)"
  echo; echo "=== optional unpacked 0.5B probe (ledger ${used} of ${SESSION_BUDGET_MINUTES} minutes used)"
  PREFILL_PACK_TOKENS=0 scale_session.sh probe-0.5b
else
  echo; echo "optional unpacked 0.5B probe skipped: ${used} of ${SESSION_BUDGET_MINUTES} ledger minutes used, its 12-minute cap would not fit"
fi

STAGE="finished"
echo; echo "=== ALL STAGES DONE ($(date -u +%H:%M:%S))"
cat "$SESSION/ledger.jsonl"
ls -lh "$SESSION"/*.tar.gz
echo "*** COPY THE EVIDENCE TO YOUR MAC, THEN TERMINATE THE POD (terminating, not stopping, ends billing)."
