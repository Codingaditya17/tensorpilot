#!/usr/bin/env bash
# Run ONE stage of the v1.0 GPU evaluation under a hard cap, a budget ledger and an always-written evidence archive.
#
#   scale_session.sh --list                       # stages, expected minutes and hard caps; starts nothing
#   scale_session.sh --plan STAGE                 # validate the environment and print what would run; starts nothing
#   scale_session.sh STAGE                        # run the stage
#
# Required environment (never put these in files or Runpod's plain settings):
#   TENSORPILOT_API_KEY      bearer token for the engine under test (TensorPilot worker or vLLM)
#   SESSION_BUDGET_MINUTES   hard budget for the sum of stage hard caps; a stage that would exceed it is refused
# Optional:
#   PRICE_PER_HOUR_USD       your Pod's hourly price, only to print a cost estimate (no price is recorded in the repo)
#   TENSORPILOT_CACHE_DIR    model cache (the Pod login banner sets it)
#   TENSORPILOT_V10_SESSION  session directory holding the ledger and every stage's evidence
#   VLLM_VENV                virtualenv created by install_vllm.sh (default /workspace/vllm-venv)
#
# Billing: stopping the engine does not stop billing. Terminate the Pod when the evidence is downloaded.
#
# Test hooks (CI dry runs on a CPU host only; they are inert unless V10_TEST_MODE=1 and print a banner):
#   V10_WORKER_CMD, V10_VLLM_CMD, V10_CONFIG_OVERRIDE, V10_SKIP_GPU_CHECKS, V10_CAP_MINUTES_OVERRIDE, V10_PORT,
#   V10_FORCE_MEMORY_GATE, V10_LIFETIME_MARGIN_OVERRIDE_MIN, V10_SKIP_LIFETIME_PRECHECK
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

log() { echo "[$(date -u +%H:%M:%S)] $*" >&2; }
die() { echo "scale_session: $*" >&2; exit "${2:-2}"; }

TEST_MODE=0
if [[ "${V10_TEST_MODE:-}" == "1" ]]; then
  TEST_MODE=1
fi
# True only for a CI dry run that explicitly switched the GPU checks off.
skip_gpu() { [[ $TEST_MODE -eq 1 && "${V10_SKIP_GPU_CHECKS:-}" == "1" ]]; }

sha256_of() {
  if command -v sha256sum >/dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi
}

load_stage() {
  local exports
  exports="$(python3 -m tensorpilot.suite --stage-env "$1")" || return 2
  eval "$exports"
  if [[ $TEST_MODE -eq 1 && -n "${V10_CAP_MINUTES_OVERRIDE:-}" ]]; then
    V10_CAP_MINUTES="$V10_CAP_MINUTES_OVERRIDE"
  fi
  if [[ $TEST_MODE -eq 1 && -n "${V10_CONFIG_OVERRIDE:-}" ]]; then
    V10_CONFIG="$V10_CONFIG_OVERRIDE"
  fi
}

session_dir() {
  local root="${TENSORPILOT_RESULTS_ROOT:-/workspace/results}"
  echo "${TENSORPILOT_V10_SESSION:-$root/v10-session}"
}

ledger_used_minutes() {
  python3 - "$1" <<'PY'
import json, sys
total = 0.0
try:
    for line in open(sys.argv[1]):
        if line.strip():
            total += json.loads(line)["elapsed_min"]
except FileNotFoundError:
    pass
print(round(total, 2))
PY
}

ledger_append() {  # ledger stage started_utc exit_status archive sha256 started_epoch
  python3 - "$@" <<'PY'
import json, sys, time
ledger, stage, started_utc, status, archive, sha, started_epoch, cap = sys.argv[1:9]
entry = {"stage": stage, "started_utc": started_utc,
         "ended_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
         "elapsed_min": round((time.time() - float(started_epoch)) / 60, 2), "exit_status": int(status),
         "cap_min": float(cap), "archive": archive, "sha256": sha}
open(ledger, "a").write(json.dumps(entry) + "\n")
PY
}

cost_line() {
  python3 - "$1" "${PRICE_PER_HOUR_USD:-}" <<'PY'
import sys
minutes, price = float(sys.argv[1]), sys.argv[2]
print(f"{minutes:.1f} stage minutes" + (f" (about ${minutes / 60 * float(price):.2f} at ${price}/h, stage time only)" if price else
      " (set PRICE_PER_HOUR_USD to see a cost estimate)"))
PY
}

list_stages() {
  python3 -m tensorpilot.suite --plan
}

require_environment() {
  [[ -n "${TENSORPILOT_API_KEY:-}" ]] || die "TENSORPILOT_API_KEY is required"
  [[ "${SESSION_BUDGET_MINUTES:-}" =~ ^[0-9]+(\.[0-9]+)?$ ]] || die "SESSION_BUDGET_MINUTES must be a positive number"
  export TENSORPILOT_API_KEY  # inherited by the engine; never placed on a command line
}

gpu_checks() {  # stage_engine
  if skip_gpu; then
    log "TEST MODE: GPU checks skipped"
    return 0
  fi
  command -v nvidia-smi >/dev/null || die "nvidia-smi not found: this is not a GPU Pod"
  local line memory driver
  line="$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits | head -1)"
  memory="$(echo "$line" | awk -F', ' '{print $2}')"
  driver="$(echo "$line" | awk -F', ' '{print $3}')"
  log "GPU: $line"
  (( ${memory%.*} >= 22000 )) || die "the stage plans assume a 24 GB card; this GPU has ${memory} MiB"
  if [[ "$1" == "vllm" ]]; then
    local major="${driver%%.*}"
    (( major >= 580 )) || die "vLLM's CUDA 13.0 runtime needs driver 580 or newer; this host has $driver"
  fi
}

# The experiment image exits its container when its lifetime backstop elapses, and pod-entrypoint.sh records that deadline in a file.
# Two defenses keep a stage's evidence from dying with the container:
#   1. pod_lifetime_check refuses to START a stage unless the real remaining time covers its hard cap, one model-load limit and
#      the archive margin, so a stage that is already doomed to be cut short is not begun.
#   2. a watchdog sends this runner SIGUSR1 LIFETIME_ARCHIVE_MARGIN_MIN minutes before the deadline wherever the stage is: the
#      running step is stopped and the partial evidence is archived while the container is still up. This holds even when a model
#      download or load takes longer than planned, so the check above can use a realistic bound instead of the worst case.
# Outside the experiment image there is no backstop and neither defense does anything.
LIFETIME_ARCHIVE_MARGIN_MIN=12
pod_deadline_file() { echo "${TENSORPILOT_POD_DEADLINE_FILE:-/run/tensorpilot/pod-deadline-epoch}"; }
lifetime_margin_min() {
  if [[ $TEST_MODE -eq 1 && -n "${V10_LIFETIME_MARGIN_OVERRIDE_MIN:-}" ]]; then echo "$V10_LIFETIME_MARGIN_OVERRIDE_MIN"; else echo "$LIFETIME_ARCHIVE_MARGIN_MIN"; fi
}

pod_lifetime_check() {
  local file
  file="$(pod_deadline_file)"
  if [[ ! -f "$file" ]]; then
    [[ -z "${TENSORPILOT_POD_MAX_MINUTES:-}" ]] && return 0
    echo "refused: TENSORPILOT_POD_MAX_MINUTES is set but the Pod deadline file $file is missing, so the remaining lifetime cannot be" \
         "verified" >&2
    return 1
  fi
  [[ $TEST_MODE -eq 1 && "${V10_SKIP_LIFETIME_PRECHECK:-}" == "1" ]] && return 0
  python3 - "$file" "$V10_CAP_MINUTES" "$V10_READY_TIMEOUT_S" "$(lifetime_margin_min)" <<'PY'
import sys, time
path, cap, ready_s, margin = sys.argv[1:5]
try:
    deadline = float(open(path).read().strip())
except (OSError, ValueError):
    print(f"refused: the Pod deadline file {path} is unreadable, so the remaining lifetime cannot be verified", file=sys.stderr)
    sys.exit(1)
remaining = (deadline - time.time()) / 60
need = float(cap) + float(ready_s) / 60 + float(margin)
if remaining < need:
    print(f"refused: this Pod's lifetime backstop leaves {remaining:.0f} minutes but this stage needs about {need:.0f} "
          f"({float(cap):.0f} hard cap + {float(ready_s) / 60:.0f} model-load limit + {float(margin):.0f} to archive). Starting it "
          "would have the container exit mid-stage. Terminate this Pod and start one with a larger "
          "TENSORPILOT_POD_MAX_MINUTES, or choose a shorter stage.", file=sys.stderr)
    sys.exit(1)
print(f"pod lifetime: {remaining:.0f} minutes remain, this stage needs about {need:.0f}", file=sys.stderr)
PY
}

WATCHDOG_PID=""
start_lifetime_watchdog() {
  local file seconds parent=$$
  file="$(pod_deadline_file)"
  [[ -f "$file" ]] || return 0
  seconds="$(python3 -c "import sys, time; print(max(0.0, float(open(sys.argv[1]).read()) - time.time() - float(sys.argv[2]) * 60))" \
             "$file" "$(lifetime_margin_min)")" || return 0
  log "lifetime watchdog: this stage will be stopped and archived $(lifetime_margin_min) minutes before the Pod's deadline, in ${seconds%.*} s if still running"
  ( sleep "$seconds"; kill -USR1 "$parent" ) >/dev/null 2>&1 &
  WATCHDOG_PID=$!
}

stop_lifetime_watchdog() {
  [[ -n "$WATCHDOG_PID" ]] || return 0
  pkill -P "$WATCHDOG_PID" 2>/dev/null || true
  kill "$WATCHDOG_PID" 2>/dev/null || true
  WATCHDOG_PID=""
}

check_disk() {
  local cache="$1" need_gib="$2" free_kib free_gib
  mkdir -p "$cache"
  free_kib="$(df -Pk "$cache" | awk 'NR==2 {print $4}')"
  free_gib=$(( free_kib / 1024 / 1024 ))
  local need
  need="$(python3 -c "import math; print(math.ceil($need_gib * 2 + 6))")"
  (( free_gib >= need )) || die "only ${free_gib} GiB free at $cache; this stage needs about ${need} GiB"
}

wait_ready() {  # url pid timeout_s [bearer]
  local url="$1" pid="$2" timeout_s="$3" key="${4:-}" elapsed=0
  local -a auth=()
  [[ -n "$key" ]] && auth=(-H "Authorization: Bearer $key")
  until curl -fsS --max-time 3 ${auth[@]+"${auth[@]}"} "$url" >/dev/null 2>&1; do
    kill -0 "$pid" 2>/dev/null || { echo "process $pid exited before $url became ready" >&2; return 1; }
    (( elapsed < timeout_s )) || { echo "timed out after ${timeout_s}s waiting for $url" >&2; return 1; }
    sleep 3
    elapsed=$((elapsed + 3))
  done
}

stop_process() {  # pid [signal]
  local pid="$1" signal="${2:-TERM}" waited=0
  kill -0 "$pid" 2>/dev/null || return 0
  kill "-$signal" "$pid" 2>/dev/null || true
  while kill -0 "$pid" 2>/dev/null && (( waited < 45 )); do sleep 1; waited=$((waited + 1)); done
  if kill -0 "$pid" 2>/dev/null; then
    log "pid $pid ignored SIG$signal for 45 s; killing"
    kill -KILL "$pid" 2>/dev/null || true
  fi
  wait "$pid" 2>/dev/null || true
}

wait_gpu_idle() {
  skip_gpu && return 0
  local used
  for _ in $(seq 1 30); do
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)"
    (( ${used%.*} < 2000 )) && return 0
    sleep 2
  done
  log "WARNING: ${used} MiB of GPU memory is still in use 60 s after the engine stopped"
}

archive_stage() {  # stage_dir ; refuses to archive a directory that contains the API key
  local dir="$1"
  if [[ -n "${TENSORPILOT_API_KEY:-}" ]] && grep -rqF -- "$TENSORPILOT_API_KEY" "$dir" 2>/dev/null; then
    echo "the API key appears in $dir; refusing to archive it" >&2
    return 1
  fi
  local archive="$dir.tar.gz"
  tar -C "$(dirname "$dir")" -czf "$archive" "$(basename "$dir")"
  tar -tzf "$archive" >/dev/null
  sha256_of "$archive" | tee "$archive.sha256" >&2
  echo "$archive"
}

engine_command_tensorpilot() {  # prints the worker command, one word per line
  if [[ $TEST_MODE -eq 1 && -n "${V10_WORKER_CMD:-}" ]]; then
    # shellcheck disable=SC2086
    printf '%s\n' $V10_WORKER_CMD
    return
  fi
  printf '%s\n' python3 -m tensorpilot.server --backend cuda --model "$V10_MODEL_REPO" --revision "$V10_REVISION" \
    --cache-dir "$CACHE" --host 127.0.0.1 --port "$PORT" --dtype "$V10_DTYPE" --quantization "$V10_QUANT" \
    --execution "$V10_EXECUTION" --batch-size "$V10_BATCH_SIZE" --queue-size "$V10_QUEUE_SIZE" \
    --max-prompt-tokens "$V10_MAX_PROMPT_TOKENS" --max-output-tokens 512 --gpu-telemetry-interval 0
}

vllm_python() { echo "${VLLM_VENV:-/workspace/vllm-venv}/bin/python"; }

vllm_guards() {
  if [[ $TEST_MODE -eq 1 && -n "${V10_VLLM_CMD:-}" ]]; then
    log "TEST MODE: vLLM environment guards skipped"
    return 0
  fi
  local python help
  python="$(vllm_python)"
  [[ -x "$python" ]] || die "vLLM is not installed: run deploy/runpod/install_vllm.sh first"
  help="$("$python" -m vllm.entrypoints.openai.api_server --help=all 2>&1)" || die "vLLM could not print its help"
  echo "$help" >"$STAGE_DIR/vllm-help.txt"
  grep -q -- "--generation-config" <<<"$help" ||
    die "this vLLM build has no --generation-config; without it the model's own sampling defaults would leak in"
  if [[ "$V10_QUANT" == "nf4" ]]; then
    grep -q "bitsandbytes" <<<"$help" || die "this vLLM build does not list bitsandbytes quantization"
  fi
  "$python" -m pip freeze >"$STAGE_DIR/vllm-env.txt" 2>&1 || true
}

engine_command_vllm() {
  if [[ $TEST_MODE -eq 1 && -n "${V10_VLLM_CMD:-}" ]]; then
    # shellcheck disable=SC2086
    printf '%s\n' $V10_VLLM_CMD
    return
  fi
  local -a command=("$(vllm_python)" -m vllm.entrypoints.openai.api_server --model "$V10_MODEL_REPO"
                    --revision "$V10_REVISION" --served-model-name "$V10_MODEL_REPO" --download-dir "$CACHE"
                    --dtype "$V10_DTYPE" --max-model-len "$((V10_MAX_PROMPT_TOKENS + 512))"
                    --max-num-seqs "$V10_BATCH_SIZE" --gpu-memory-utilization 0.90 --host 127.0.0.1
                    --port "$PORT" --generation-config vllm)
  [[ "$V10_QUANT" == "nf4" ]] && command+=(--quantization bitsandbytes --load-format bitsandbytes)
  printf '%s\n' "${command[@]}"
}

# Every long step runs as a background child that the shell waits on. A signal then reaches the trap at once (bash defers traps
# while a foreground command runs), so an interrupt stops the step and still archives what it produced.
CHILD_PID=""
TAIL_PID=""
# GUARD_OUT=file run_guarded ... sends only the child's output to that file; the trap and finalize logs stay on the terminal.
run_guarded() {
  if [[ -n "${GUARD_OUT:-}" ]]; then
    "$@" >>"$GUARD_OUT" 2>&1 &
  else
    "$@" &
  fi
  CHILD_PID=$!
  local rc=0
  wait "$CHILD_PID" || rc=$?
  CHILD_PID=""
  return $rc
}

stop_child() {
  [[ -n "$CHILD_PID" ]] || return 0
  local pid="$CHILD_PID" waited=0
  kill -TERM "$pid" 2>/dev/null || true
  while kill -0 "$pid" 2>/dev/null && (( waited < 75 )); do sleep 1; waited=$((waited + 1)); done
  if kill -0 "$pid" 2>/dev/null; then
    log "step $pid ignored SIGTERM for 75 s; killing it and its children"
    pkill -KILL -P "$pid" 2>/dev/null || true
    kill -KILL "$pid" 2>/dev/null || true
  fi
  wait "$pid" 2>/dev/null || true
  CHILD_PID=""
}

ENGINE_PID=""
start_engine() {  # engine
  local -a command
  mapfile -t command < <("engine_command_$1")
  # The key travels by environment only, so it never appears in `ps`. vLLM reads VLLM_API_KEY.
  VLLM_API_KEY="$TENSORPILOT_API_KEY" "${command[@]}" >"$STAGE_DIR/engine.log" 2>&1 &
  ENGINE_PID=$!
}

cleanup_engine() {
  if [[ -n "$ENGINE_PID" ]]; then
    stop_process "$ENGINE_PID" TERM
    ENGINE_PID=""
    wait_gpu_idle
  fi
}

run_smoke() {  # engine url
  local extra=()
  [[ "$1" == "vllm" ]] && extra=(--api openai --model-name "$V10_MODEL_REPO")
  run_guarded python3 -m tensorpilot.load --url "$2" --requests 5 --concurrency 1 --max-tokens 16 --timeout 120 \
    "${extra[@]}" --output "$STAGE_DIR/smoke.json" --journal "$STAGE_DIR/smoke.requests.jsonl" || true
  run_guarded python3 -m tensorpilot.smoke_check "$STAGE_DIR/smoke.json" --expect 5 --log "$STAGE_DIR/engine.log"
}

# The gate reads a real GPU's allocator numbers, so a CPU dry run skips it unless a test forces it (a stub then fails it closed).
memory_gate_applies() { ! skip_gpu || [[ $TEST_MODE -eq 1 && "${V10_FORCE_MEMORY_GATE:-}" == "1" ]]; }

stage_benchmark() {  # engine
  local engine="$1" url="http://127.0.0.1:$PORT" status=0
  if [[ "$engine" == "tensorpilot" ]] && ! skip_gpu; then
    log "preflight (downloads the model if it is not cached)"
    run_guarded timeout --signal=TERM --kill-after=30 "${V10_READY_TIMEOUT_S}s" python3 -m tensorpilot.gpu_preflight \
      --model "$V10_MODEL_REPO" --revision "$V10_REVISION" --cache-dir "$CACHE" --dtype "$V10_DTYPE" \
      --quantization "$V10_QUANT" --max-prompt-tokens "$V10_MAX_PROMPT_TOKENS" \
      --output "$STAGE_DIR/preflight.json" --traceback-file "$STAGE_DIR/preflight.traceback.txt" || return $?
  fi
  [[ "$engine" == "vllm" ]] && vllm_guards
  log "starting $engine on 127.0.0.1:$PORT (ready timeout ${V10_READY_TIMEOUT_S}s)"
  start_engine "$engine"
  local probe="$url/readyz" key=""
  if [[ "$engine" == "vllm" ]]; then probe="$url/v1/models"; key="$TENSORPILOT_API_KEY"; fi
  wait_ready "$probe" "$ENGINE_PID" "$V10_READY_TIMEOUT_S" "$key" || { tail -20 "$STAGE_DIR/engine.log" >&2; return 1; }
  run_smoke "$engine" "$url" || return 1
  if [[ "$engine" == "tensorpilot" && "$V10_EXECUTION" == "batch" ]] && memory_gate_applies; then
    # Measured, not planned: the real free memory inside the allocator cap and the slots actually granted.
    log "memory gate: measured margin and granted slots"
    run_guarded python3 -m tensorpilot.memory_gate --url "$url" --expect-rows "$V10_BATCH_SIZE" \
      --min-margin-gib "${V10_MIN_MARGIN_GIB:-1.5}" --output "$STAGE_DIR/memory-gate.json" ||
      { echo "the memory gate failed: the stage's pool or margin is not what the plan assumed; not benchmarking" >&2; return 1; }
  fi
  if [[ "${V10_INVARIANCE:-0}" == "1" && "$engine" == "tensorpilot" ]]; then
    log "batch-invariance: does a request's output depend on what else is in the batch?"
    run_guarded timeout --signal=TERM --kill-after=30 10m python3 -m tensorpilot.batch_invariance --url "$url" --prompts 32 \
      --output "$STAGE_DIR/invariance.json" || log "the invariance check did not complete; the benchmark continues"
  fi
  # The runner stops starting experiments one minute before the hard kill, so a normal overrun ends cleanly.
  local soft
  soft="$(python3 -c "cap = float('$V10_CAP_MINUTES'); print(cap - 1 if cap > 2 else cap)")"
  log "benchmark: $V10_CONFIG, soft stop at ${soft} minutes, hard cap ${V10_CAP_MINUTES}"
  : >"$STAGE_DIR/bench-run.log"
  tail -n +1 -F "$STAGE_DIR/bench-run.log" 2>/dev/null &
  TAIL_PID=$!
  GUARD_OUT="$STAGE_DIR/bench-run.log" run_guarded timeout --signal=TERM --kill-after=60 "${V10_CAP_MINUTES}m" \
    python3 -m tensorpilot.gpu_bench run "$V10_CONFIG" --url "$url" --output-dir "$STAGE_DIR/bench" \
    --session-minutes "$soft" || status=$?
  sleep 1
  kill "$TAIL_PID" 2>/dev/null || true
  TAIL_PID=""
  return $status
}

stage_quality() {
  local common=(--model "$V10_MODEL_REPO" --revision "$V10_REVISION" --cache-dir "$CACHE" --dtype "$V10_DTYPE")
  local cap="${V10_CAP_MINUTES}m"
  if [[ "$V10_MODEL_KEY" != "14b" ]]; then
    log "capturing the bf16 reference"
    run_guarded timeout --signal=TERM --kill-after=60 "$cap" python3 -m tensorpilot.quality capture "${common[@]}" \
      --output "$STAGE_DIR/reference-bf16.json" || return $?
    log "capturing NF4 against the reference"
    run_guarded timeout --signal=TERM --kill-after=60 "$cap" python3 -m tensorpilot.quality capture "${common[@]}" \
      --quantization nf4 --reference "$STAGE_DIR/reference-bf16.json" --output "$STAGE_DIR/nf4.json" || return $?
    GUARD_OUT="$STAGE_DIR/comparison.txt" run_guarded python3 -m tensorpilot.quality compare \
      "$STAGE_DIR/reference-bf16.json" "$STAGE_DIR/nf4.json" --output "$STAGE_DIR/comparison.json" || return $?
    cat "$STAGE_DIR/comparison.txt"
  else
    log "capturing 14B NF4 (absolute metrics only: no bf16 reference fits on this card)"
    run_guarded timeout --signal=TERM --kill-after=60 "$cap" python3 -m tensorpilot.quality capture "${common[@]}" \
      --quantization nf4 --output "$STAGE_DIR/nf4-absolute.json" || return $?
  fi
}

stage_faults() {
  python3 -m tensorpilot.suite --dump-spec short >"$STAGE_DIR/workload-spec.json"
  local -a command=(python3 -m tensorpilot.fault_run --port "$PORT" --url "http://127.0.0.1:$PORT"
                    --output-dir "$STAGE_DIR/fault" --workload-spec "$STAGE_DIR/workload-spec.json"
                    --requests "$V10_FAULT_REQUESTS" --rate "$V10_FAULT_RATE" --kill-after "$V10_FAULT_KILL_AFTER"
                    --execution "$V10_EXECUTION" --batch-size "$V10_BATCH_SIZE" --queue-size "$V10_QUEUE_SIZE"
                    --ready-timeout "$V10_READY_TIMEOUT_S" --max-seconds "$((V10_CAP_MINUTES * 60))")
  if [[ $TEST_MODE -eq 1 && -n "${V10_WORKER_CMD:-}" ]]; then
    command+=(--worker-command-json "$(python3 -c "import json,sys; print(json.dumps(sys.argv[1:]))" $V10_WORKER_CMD)")
  else
    local flag
    for flag in --backend cuda --model "$V10_MODEL_REPO" --revision "$V10_REVISION" --cache-dir "$CACHE" \
                --dtype "$V10_DTYPE" --quantization "$V10_QUANT" --max-prompt-tokens "$V10_MAX_PROMPT_TOKENS" \
                --gpu-telemetry-interval 0; do
      command+=("--worker-arg=$flag")
    done
  fi
  run_guarded timeout --signal=TERM --kill-after=60 "${V10_CAP_MINUTES}m" "${command[@]}"
}

run_stage() {
  local stage="$1"
  load_stage "$stage" || die "unknown stage: $stage (try --list)"
  require_environment
  CACHE="${TENSORPILOT_CACHE_DIR:-/workspace/huggingface}"
  PORT="$([[ "$V10_ENGINE" == "vllm" ]] && echo 8081 || echo 8080)"
  [[ $TEST_MODE -eq 1 && -n "${V10_PORT:-}" ]] && PORT="$V10_PORT"
  [[ $TEST_MODE -eq 1 ]] && log "TEST MODE: hooks active, results are not GPU evidence"
  SESSION="$(session_dir)"
  mkdir -p "$SESSION"
  local ledger="$SESSION/ledger.jsonl" used
  used="$(ledger_used_minutes "$ledger")"
  python3 - "$used" "$V10_CAP_MINUTES" "$SESSION_BUDGET_MINUTES" <<'PY' || exit 3
import sys
used, cap, budget = map(float, sys.argv[1:4])
if used + cap > budget:
    print(f"refused: {used:.1f} minutes already used + this stage's {cap:.0f} minute hard cap exceeds the "
          f"{budget:.0f} minute session budget. Raise SESSION_BUDGET_MINUTES only if you accept the cost.", file=sys.stderr)
    sys.exit(1)
PY
  gpu_checks "$V10_ENGINE"
  pod_lifetime_check || exit 3
  skip_gpu || check_disk "$CACHE" "$V10_DOWNLOAD_GIB"
  STAGE_DIR="$SESSION/$stage-$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "$STAGE_DIR"
  local started_epoch started_utc
  started_epoch="$(date +%s)"
  started_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  STAGE_STATUS=""
  STAGE_STARTED_EPOCH="$started_epoch" STAGE_STARTED_UTC="$started_utc" STAGE_LEDGER="$ledger" STAGE_NAME="$stage"
  FINALIZED=0
  trap 'finalize_stage $?' EXIT
  trap 'on_signal TERM 143' TERM
  trap 'on_signal INT 130' INT
  trap 'on_signal HUP 129' HUP
  trap 'on_signal USR1 125' USR1
  start_lifetime_watchdog
  case "$V10_ENGINE" in
    tensorpilot|vllm) stage_benchmark "$V10_ENGINE" || STAGE_STATUS=$? ;;
    quality) stage_quality || STAGE_STATUS=$? ;;
    faults) stage_faults || STAGE_STATUS=$? ;;
  esac
  STAGE_STATUS="${STAGE_STATUS:-0}"
  finalize_stage "$STAGE_STATUS"
  return "$STAGE_STATUS"
}

# Runs exactly once, on a normal finish, a failure, or an interrupt: stop the step and the engine, archive whatever exists, write the ledger.
finalize_stage() {
  (( FINALIZED == 0 )) || return 0
  FINALIZED=1
  trap '' TERM INT HUP USR1  # nothing may interrupt the archive
  stop_lifetime_watchdog
  local status="${STAGE_STATUS:-${1:-1}}" archive="" sha="" used
  stop_child
  [[ -n "$TAIL_PID" ]] && { kill "$TAIL_PID" 2>/dev/null || true; TAIL_PID=""; }
  cleanup_engine
  case "$status" in
    143|130|129) log "stage $STAGE_NAME was interrupted (exit $status); archiving the partial evidence" ;;
    125) log "stage $STAGE_NAME was stopped by the lifetime watchdog (exit 125); archiving the partial evidence" ;;
    *) log "stage $STAGE_NAME finished with exit status $status; archiving evidence" ;;
  esac
  if archive="$(archive_stage "$STAGE_DIR")"; then
    sha="$(cut -d' ' -f1 "$archive.sha256")"
  else
    archive=""
    (( status != 0 )) || status=1  # an unarchivable stage is a failed stage
  fi
  ledger_append "$STAGE_LEDGER" "$STAGE_NAME" "$STAGE_STARTED_UTC" "$status" "$archive" "$sha" "$STAGE_STARTED_EPOCH" "$V10_CAP_MINUTES" ||
    log "WARNING: the ledger entry could not be written"
  used="$(ledger_used_minutes "$STAGE_LEDGER")"
  log "session ledger: $(cost_line "$used"); budget ${SESSION_BUDGET_MINUTES} minutes"
  log "evidence: ${archive:-NOT ARCHIVED}"
  log "THE POD IS STILL BILLING. Download the archive, then TERMINATE the Pod in the Runpod console."
  STAGE_STATUS="$status"
}

on_signal() {  # signal_name exit_code
  if [[ "$1" == USR1 ]]; then
    log "the Pod's lifetime backstop is $(lifetime_margin_min) minutes away: stopping the running step; partial evidence will be archived"
  else
    log "received SIG$1: stopping the running step; partial evidence will be archived"
  fi
  STAGE_STATUS="$2"
  exit "$2"
}

plan_stage() {
  local stage="$1"
  load_stage "$stage" || die "unknown stage: $stage (try --list)"
  CACHE="${TENSORPILOT_CACHE_DIR:-/workspace/huggingface}"
  cat <<PLAN
stage:           $stage  ($V10_ENGINE)
model:           $V10_MODEL_REPO @ $V10_REVISION
precision:       dtype $V10_DTYPE, quantization $V10_QUANT, execution $V10_EXECUTION
slots / queue:   $V10_BATCH_SIZE / $V10_QUEUE_SIZE   (the pool may grant fewer: see docs/v10-evaluation-plan.md)
prompt limit:    $V10_MAX_PROMPT_TOKENS tokens
config:          ${V10_CONFIG:-n/a}
expected:        $V10_EXPECTED_MINUTES minutes of experiments (an assumption, not a measurement)
HARD CAP:        $V10_CAP_MINUTES minutes (the stage is stopped there and its evidence kept)
model download:  about $V10_DOWNLOAD_GIB GiB into $CACHE (not billed separately, but the time is)
PLAN
  pod_lifetime_check || return 3
  if [[ -n "${SESSION_BUDGET_MINUTES:-}" ]]; then
    local used
    used="$(ledger_used_minutes "$(session_dir)/ledger.jsonl")"
    python3 - "$used" "$V10_CAP_MINUTES" "$SESSION_BUDGET_MINUTES" <<'PY' || return 3
import sys
used, cap, budget = map(float, sys.argv[1:4])
verdict = "WITHIN" if used + cap <= budget else "OVER"
print(f"budget:          {used:.1f} used + {cap:.0f} cap vs {budget:.0f} budget: {verdict} budget")
sys.exit(0 if verdict == "WITHIN" else 1)
PY
  else
    echo "budget:          SESSION_BUDGET_MINUTES is not set (required to run)"
  fi
}

main() {
  case "${1:-}" in
    --list) list_stages ;;
    --plan) [[ -n "${2:-}" ]] || die "usage: $0 --plan STAGE"; plan_stage "$2" ;;
    ""|-h|--help) sed -n '2,24p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' ;;
    -*) die "unknown option $1" ;;
    *) run_stage "$1" ;;
  esac
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
