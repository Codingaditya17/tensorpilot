#!/usr/bin/env bash
# One bounded Runpod GPU session: diagnostics, preflight, worker, gateway, short benchmark, evidence.
#
# Required environment (set in the Pod's secret store, never in files committed to git):
#   TENSORPILOT_API_KEY           bearer token for the CUDA worker
#   TENSORPILOT_CONTROL_API_KEY   bearer token for the control plane (must differ from the others)
#   TENSORPILOT_DATA_API_KEY      bearer token for the data plane (must differ from the others)
#   TENSORPILOT_MODEL             Transformers checkpoint, for example Qwen/Qwen2.5-0.5B-Instruct
#   TENSORPILOT_REVISION          40-character Hugging Face commit SHA
# Optional:
#   TENSORPILOT_RESULTS           evidence directory (default /workspace/results/v09-<UTC stamp>)
#   TENSORPILOT_BENCH_CONFIG      experiment file (default benchmarks/gpu/v09-first-30min.json)
#   TENSORPILOT_BENCH_TIMEOUT     hard cap on the benchmark step, GNU timeout syntax (default 35m)
#   TENSORPILOT_CACHE_DIR         Hugging Face cache directory (default /workspace/huggingface)
#
# Network exposure: every process binds 127.0.0.1. Nothing in this script opens a public port.
# To look at the dashboard from your laptop, use an SSH tunnel, for example
#   ssh -L 8080:127.0.0.1:8080 <pod>
#
# Usage:
#   deploy/runpod/session.sh --plan   # validate the environment and print the steps; starts nothing
#   deploy/runpod/session.sh          # run the session; every started process is stopped on exit
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

if [[ "${1:-}" != "" && "${1:-}" != "--plan" ]]; then
  echo "usage: $0 [--plan]" >&2
  exit 2
fi
PLAN_ONLY=0
[[ "${1:-}" == "--plan" ]] && PLAN_ONLY=1

for name in TENSORPILOT_API_KEY TENSORPILOT_CONTROL_API_KEY TENSORPILOT_DATA_API_KEY TENSORPILOT_MODEL TENSORPILOT_REVISION; do
  if [[ -z "${!name:-}" ]]; then
    echo "missing required environment variable: $name" >&2
    exit 2
  fi
done
if [[ "$TENSORPILOT_API_KEY" == "$TENSORPILOT_CONTROL_API_KEY" || "$TENSORPILOT_API_KEY" == "$TENSORPILOT_DATA_API_KEY" \
   || "$TENSORPILOT_CONTROL_API_KEY" == "$TENSORPILOT_DATA_API_KEY" ]]; then
  echo "worker, control and data-plane keys must all be different" >&2
  exit 2
fi
if [[ ! "$TENSORPILOT_REVISION" =~ ^[0-9a-f]{40}$ ]]; then
  echo "TENSORPILOT_REVISION must be a 40-character lowercase commit SHA" >&2
  exit 2
fi

WORKER_PORT="${TENSORPILOT_PORT:-8080}"
CONTROL_PORT="${TENSORPILOT_CONTROL_PORT:-8090}"
DATA_PORT="${TENSORPILOT_DATA_PORT:-8091}"
BENCH_CONFIG="${TENSORPILOT_BENCH_CONFIG:-benchmarks/gpu/v09-first-30min.json}"
BENCH_TIMEOUT="${TENSORPILOT_BENCH_TIMEOUT:-35m}"
CACHE_DIR="${TENSORPILOT_CACHE_DIR:-/workspace/huggingface}"
RESULTS="${TENSORPILOT_RESULTS:-/workspace/results/v09-$(date -u +%Y%m%dT%H%M%SZ)}"

if [[ $PLAN_ONLY -eq 1 ]]; then
  cat <<PLAN
TensorPilot v0.9 Runpod session (nothing started)
  model:      $TENSORPILOT_MODEL @ $TENSORPILOT_REVISION
  results:    $RESULTS
  cache:      $CACHE_DIR
  bench:      $BENCH_CONFIG (hard cap $BENCH_TIMEOUT)
  bind:       127.0.0.1 only (worker :$WORKER_PORT, control :$CONTROL_PORT, data :$DATA_PORT)
  1. diagnostics  nvidia-smi, driver, CUDA runtime, package versions -> diagnostics/
  2. preflight    tensorpilot.gpu_preflight: driver policy, pinned revision, storage, VRAM fit,
                  smoke inference; the session stops unless the gate is ready
  3. worker       tensorpilot.server --backend cuda, readiness on /readyz
  4. gateway      control and data planes; one streamed smoke request validated by tensorpilot.load
  5. benchmark    tensorpilot.gpu_bench plan, then run with per-experiment caps and abort rules
  6. evidence     scan for secrets, archive results, write a SHA-256 checksum
  7. teardown     stop started processes; then terminate the Pod (runbook checklist)
PLAN
  exit 0
fi

mkdir -p "$RESULTS"
PIDS=()
cleanup() {
  local status=$?
  for pid in ${PIDS[@]+"${PIDS[@]}"}; do
    kill -TERM "$pid" 2>/dev/null || true
  done
  for pid in ${PIDS[@]+"${PIDS[@]}"}; do
    wait "$pid" 2>/dev/null || true
  done
  echo "teardown complete (exit $status). Download the evidence, then terminate the Pod." >&2
  exit "$status"
}
trap cleanup EXIT

wait_for_ready() {
  local url="$1" pid="$2" timeout_s="$3" elapsed=0
  until curl -fsS --max-time 2 "$url" >/dev/null 2>&1; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "process $pid exited before $url became ready" >&2
      return 1
    fi
    if (( elapsed >= timeout_s )); then
      echo "timed out waiting for $url" >&2
      return 1
    fi
    sleep 2
    elapsed=$((elapsed + 2))
  done
}

echo "[1/7] diagnostics -> $RESULTS/diagnostics"
bash "$REPO_ROOT/deploy/runpod/diagnostics.sh" "$RESULTS/diagnostics"

echo "[2/7] preflight"
python3 -m tensorpilot.gpu_preflight --model "$TENSORPILOT_MODEL" --revision "$TENSORPILOT_REVISION" \
  --cache-dir "$CACHE_DIR" --output "$RESULTS/preflight.json"

echo "[3/7] worker on 127.0.0.1:$WORKER_PORT"
TENSORPILOT_API_KEY="$TENSORPILOT_API_KEY" python3 -m tensorpilot.server --backend cuda \
  --model "$TENSORPILOT_MODEL" --revision "$TENSORPILOT_REVISION" --cache-dir "$CACHE_DIR" \
  --host 127.0.0.1 --port "$WORKER_PORT" >"$RESULTS/worker.log" 2>&1 &
PIDS+=($!)
wait_for_ready "http://127.0.0.1:$WORKER_PORT/readyz" "${PIDS[-1]}" 900

echo "[4/7] gateway: control $CONTROL_PORT, data $DATA_PORT"
TENSORPILOT_WORKER_API_KEY="$TENSORPILOT_API_KEY" TENSORPILOT_CONTROL_API_KEY="$TENSORPILOT_CONTROL_API_KEY" \
  python3 -m tensorpilot.control_plane --host 127.0.0.1 --port "$CONTROL_PORT" \
  --worker "worker-a=http://127.0.0.1:$WORKER_PORT" >"$RESULTS/control.log" 2>&1 &
PIDS+=($!)
wait_for_ready "http://127.0.0.1:$CONTROL_PORT/readyz" "${PIDS[-1]}" 120
TENSORPILOT_CONTROL_API_KEY="$TENSORPILOT_CONTROL_API_KEY" TENSORPILOT_DATA_API_KEY="$TENSORPILOT_DATA_API_KEY" \
  python3 -m tensorpilot.data_plane --host 127.0.0.1 --port "$DATA_PORT" \
  --control-url "http://127.0.0.1:$CONTROL_PORT" >"$RESULTS/data.log" 2>&1 &
PIDS+=($!)
wait_for_ready "http://127.0.0.1:$DATA_PORT/readyz" "${PIDS[-1]}" 120
# The integration check uses the same stream validator as the benchmark: any protocol violation fails it.
TENSORPILOT_API_KEY="$TENSORPILOT_DATA_API_KEY" python3 -m tensorpilot.load \
  --url "http://127.0.0.1:$DATA_PORT" --requests 5 --concurrency 1 --max-tokens 16 --timeout 60 \
  --output "$RESULTS/gateway-smoke.json" --journal "$RESULTS/gateway-smoke.requests.jsonl"
python3 - "$RESULTS/gateway-smoke.json" <<'PY'
import json, sys
summary = json.load(open(sys.argv[1]))["summary"]
if summary["failed_requests"] or summary["successful_requests"] != 5:
    sys.exit("gateway smoke failed: " + json.dumps(summary))
print("gateway smoke: 5 of 5 streams validated through the data plane")
PY

echo "[5/7] benchmark plan and run (caps and abort rules from the config)"
python3 -m tensorpilot.gpu_bench plan "$BENCH_CONFIG" | tee "$RESULTS/plan.txt"
TENSORPILOT_API_KEY="$TENSORPILOT_API_KEY" timeout "$BENCH_TIMEOUT" python3 -m tensorpilot.gpu_bench run \
  "$BENCH_CONFIG" --url "http://127.0.0.1:$WORKER_PORT" --output-dir "$RESULTS/bench" | tee "$RESULTS/bench-run.log"

echo "[6/7] secret scan and evidence archive"
for key in "$TENSORPILOT_API_KEY" "$TENSORPILOT_CONTROL_API_KEY" "$TENSORPILOT_DATA_API_KEY"; do
  if grep -rqF -- "$key" "$RESULTS"; then
    echo "a credential value appears in the evidence directory; refusing to archive" >&2
    exit 1
  fi
done
ARCHIVE="$RESULTS.tar.gz"
tar -C "$(dirname "$RESULTS")" -czf "$ARCHIVE" "$(basename "$RESULTS")"
sha256sum "$ARCHIVE" | tee "$ARCHIVE.sha256"
tar -tzf "$ARCHIVE" >/dev/null && echo "archive readable: $ARCHIVE"

echo "[7/7] stopping started processes"
