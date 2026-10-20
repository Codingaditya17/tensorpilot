#!/usr/bin/env bash
# One bounded Runpod GPU session: diagnostics, preflight, worker, gateway, benchmark, evidence.
#
# Required environment (set in the Pod's secret store, never in files committed to git):
#   TENSORPILOT_API_KEY    bearer token for every TensorPilot endpoint
#   TENSORPILOT_MODEL      Transformers checkpoint, for example Qwen/Qwen2.5-0.5B-Instruct
#   TENSORPILOT_REVISION   40-character Hugging Face commit SHA
# Optional:
#   TENSORPILOT_RESULTS    evidence directory (default /workspace/results/v09-<UTC stamp>)
#   TENSORPILOT_BENCH_CONFIG  experiment file (default benchmarks/gpu/v09-single-gpu.json)
#   TENSORPILOT_BENCH_TIMEOUT hard cap on the benchmark step, GNU timeout syntax (default 3h)
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

for name in TENSORPILOT_API_KEY TENSORPILOT_MODEL TENSORPILOT_REVISION; do
  if [[ -z "${!name:-}" ]]; then
    echo "missing required environment variable: $name" >&2
    exit 2
  fi
done
if [[ ! "$TENSORPILOT_REVISION" =~ ^[0-9a-f]{40}$ ]]; then
  echo "TENSORPILOT_REVISION must be a 40-character lowercase commit SHA" >&2
  exit 2
fi

WORKER_PORT="${TENSORPILOT_PORT:-8080}"
CONTROL_PORT="${TENSORPILOT_CONTROL_PORT:-8090}"
DATA_PORT="${TENSORPILOT_DATA_PORT:-8091}"
BENCH_CONFIG="${TENSORPILOT_BENCH_CONFIG:-benchmarks/gpu/v09-single-gpu.json}"
BENCH_TIMEOUT="${TENSORPILOT_BENCH_TIMEOUT:-3h}"
RESULTS="${TENSORPILOT_RESULTS:-/workspace/results/v09-$(date -u +%Y%m%dT%H%M%SZ)}"

if [[ $PLAN_ONLY -eq 1 ]]; then
  cat <<PLAN
TensorPilot v0.9 Runpod session (nothing started)
  model:      $TENSORPILOT_MODEL @ $TENSORPILOT_REVISION
  results:    $RESULTS
  bench:      $BENCH_CONFIG (hard cap $BENCH_TIMEOUT)
  1. diagnostics  nvidia-smi, driver, CUDA runtime, package versions -> diagnostics/
  2. preflight    tensorpilot.gpu_preflight; the session stops unless every check passes
  3. worker       tensorpilot.server --backend cuda on 127.0.0.1:$WORKER_PORT
  4. gateway      control plane :$CONTROL_PORT, data plane :$DATA_PORT, one streamed smoke request
  5. benchmark    tensorpilot.gpu_bench plan, then run against the worker
  6. evidence     archive results and write a SHA-256 checksum
  7. teardown     stop started processes; then terminate the Pod (see the runbook checklist)
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
  echo "teardown complete (exit $status). Terminate the Pod now; see the billable-resource checklist." >&2
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
  --output "$RESULTS/preflight.json"

echo "[3/7] worker on 127.0.0.1:$WORKER_PORT"
python3 -m tensorpilot.server --backend cuda --model "$TENSORPILOT_MODEL" \
  --revision "$TENSORPILOT_REVISION" --host 127.0.0.1 --port "$WORKER_PORT" \
  >"$RESULTS/worker.log" 2>&1 &
PIDS+=($!)
wait_for_ready "http://127.0.0.1:$WORKER_PORT/readyz" "${PIDS[-1]}" 900

echo "[4/7] gateway: control $CONTROL_PORT, data $DATA_PORT"
TENSORPILOT_WORKER_API_KEY="$TENSORPILOT_API_KEY" TENSORPILOT_CONTROL_API_KEY="$TENSORPILOT_API_KEY" \
  python3 -m tensorpilot.control_plane --port "$CONTROL_PORT" --worker "worker-a=http://127.0.0.1:$WORKER_PORT" \
  >"$RESULTS/control.log" 2>&1 &
PIDS+=($!)
wait_for_ready "http://127.0.0.1:$CONTROL_PORT/readyz" "${PIDS[-1]}" 120
TENSORPILOT_CONTROL_API_KEY="$TENSORPILOT_API_KEY" TENSORPILOT_DATA_API_KEY="$TENSORPILOT_API_KEY" \
  python3 -m tensorpilot.data_plane --port "$DATA_PORT" --control-url "http://127.0.0.1:$CONTROL_PORT" \
  >"$RESULTS/data.log" 2>&1 &
PIDS+=($!)
wait_for_ready "http://127.0.0.1:$DATA_PORT/readyz" "${PIDS[-1]}" 120
curl -fsS --max-time 60 -H "Authorization: Bearer $TENSORPILOT_API_KEY" -H "Content-Type: application/json" \
  -d '{"prompt":"Explain GPU memory bandwidth in one sentence.","max_tokens":16}' \
  "http://127.0.0.1:$DATA_PORT/v1/generate" >"$RESULTS/smoke.ndjson"
if ! grep -Eq '"type": ?"done"' "$RESULTS/smoke.ndjson"; then
  echo "smoke request through the gateway did not finish with a done event" >&2
  exit 1
fi
echo "smoke request through the gateway completed"

echo "[5/7] benchmark plan and run"
python3 -m tensorpilot.gpu_bench plan "$BENCH_CONFIG" | tee "$RESULTS/plan.txt"
timeout "$BENCH_TIMEOUT" python3 -m tensorpilot.gpu_bench run "$BENCH_CONFIG" \
  --url "http://127.0.0.1:$WORKER_PORT" --output-dir "$RESULTS/bench" | tee "$RESULTS/bench-run.log"

echo "[6/7] evidence archive"
ARCHIVE="$RESULTS.tar.gz"
tar -C "$(dirname "$RESULTS")" -czf "$ARCHIVE" "$(basename "$RESULTS")"
sha256sum "$ARCHIVE" | tee "$ARCHIVE.sha256"

echo "[7/7] stopping started processes"
