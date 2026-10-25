#!/usr/bin/env bash
# PREPARED, NOT EXECUTED. Two independent worker processes, one per GPU, behind one gateway.
# Run only after the single-GPU session (deploy/runpod/session.sh) passed and you chose to rent
# a Pod with at least two GPUs. See docs/v09-two-gpu-experiment.md.
#
# Same environment as session.sh (three distinct keys, TENSORPILOT_MODEL, TENSORPILOT_REVISION).
# Optional: TENSORPILOT_RESULTS, TENSORPILOT_CACHE_DIR, TENSORPILOT_LOAD_REQUESTS (default 60),
#           TENSORPILOT_LOAD_CONCURRENCY (default 8), TENSORPILOT_PHASE_TIMEOUT (default 10m).
#
# What this proves: two GPU worker processes on one host register, receive routed load, and one
# survives the other's failure. What it does NOT prove: multi-node networking. Placement evidence
# is recorded in placement.json; the report must quote it before using the words "multi-node".
#
#   deploy/runpod/two_gpu_session.sh --plan   # validate and print; starts nothing
#   deploy/runpod/two_gpu_session.sh          # run; every started process is stopped on exit
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

PORT_A=8081; PORT_B=8082; CONTROL_PORT=8090; DATA_PORT=8091
REQUESTS="${TENSORPILOT_LOAD_REQUESTS:-60}"
CONCURRENCY="${TENSORPILOT_LOAD_CONCURRENCY:-8}"
PHASE_TIMEOUT="${TENSORPILOT_PHASE_TIMEOUT:-10m}"
CACHE_DIR="${TENSORPILOT_CACHE_DIR:-/workspace/huggingface}"
RESULTS="${TENSORPILOT_RESULTS:-/workspace/results/v09-two-gpu-$(date -u +%Y%m%dT%H%M%SZ)}"

if [[ $PLAN_ONLY -eq 1 ]]; then
  cat <<PLAN
TensorPilot v0.9 two-GPU session (nothing started)
  model:    $TENSORPILOT_MODEL @ $TENSORPILOT_REVISION
  results:  $RESULTS
  load:     $REQUESTS requests, concurrency $CONCURRENCY per phase, each phase capped at $PHASE_TIMEOUT
  bind:     127.0.0.1 only (workers :$PORT_A :$PORT_B, control :$CONTROL_PORT, data :$DATA_PORT)
  1. placement   refuse unless >= 2 GPUs; record hostname, machine-id, GPU UUIDs and PCI bus ids
  2. workers     worker-a on GPU 0 and worker-b on GPU 1 via CUDA_VISIBLE_DEVICES, own processes
  3. gateway     control plane registers both (capacity-aware); data plane in front
  4. baseline    phase A: same load straight to worker-a (single-GPU reference, same session)
  5. two-GPU     phase B: same load through the gateway; per-worker request counts, nvidia-smi
                 samples for both GPUs, /v1/workers snapshots
  6. failure     phase C: kill -9 worker-b during load; record detection time, request outcomes
  7. recovery    restart worker-b, wait for registry to mark it available, short load
  8. evidence    secret scan, archive, checksum; then terminate the Pod
PLAN
  exit 0
fi

mkdir -p "$RESULTS"
PIDS=()
SAMPLER=""
cleanup() {
  local status=$?
  [[ -n "$SAMPLER" ]] && kill "$SAMPLER" 2>/dev/null || true
  for pid in ${PIDS[@]+"${PIDS[@]}"}; do kill -TERM "$pid" 2>/dev/null || true; done
  for pid in ${PIDS[@]+"${PIDS[@]}"}; do wait "$pid" 2>/dev/null || true; done
  echo "teardown complete (exit $status). Download the evidence, then terminate the Pod." >&2
  exit "$status"
}
trap cleanup EXIT

wait_for_ready() {
  local url="$1" pid="$2" timeout_s="$3" elapsed=0
  until curl -fsS --max-time 2 "$url" >/dev/null 2>&1; do
    kill -0 "$pid" 2>/dev/null || { echo "process $pid exited before $url became ready" >&2; return 1; }
    (( elapsed >= timeout_s )) && { echo "timed out waiting for $url" >&2; return 1; }
    sleep 2; elapsed=$((elapsed + 2))
  done
}

echo "[1/8] placement evidence -> $RESULTS/placement.json"
GPU_COUNT="$(nvidia-smi -L | wc -l)"
if (( GPU_COUNT < 2 )); then
  echo "found $GPU_COUNT GPU(s); this experiment needs at least 2. Aborting before any worker starts." >&2
  exit 3
fi
bash "$REPO_ROOT/deploy/runpod/diagnostics.sh" "$RESULTS/diagnostics"
python3 - "$RESULTS/placement.json" <<'PY'
import json, platform, subprocess, sys
def run(*cmd):
    return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout.strip()
def read(path):
    try:
        return open(path).read().strip()
    except OSError:
        return None
gpus = [dict(zip(("index", "uuid", "pci_bus_id", "name", "memory_total_mib"), [c.strip() for c in line.split(",")]))
        for line in run("nvidia-smi", "--query-gpu=index,uuid,pci.bus_id,name,memory.total",
                        "--format=csv,noheader,nounits").splitlines()]
report = {"hostname": platform.node(), "machine_id": read("/etc/machine-id"),
          "gpus": gpus, "distinct_uuids": len({g["uuid"] for g in gpus}) == len(gpus),
          "same_host": True,
          "note": "Both workers run on this one host. This is two physical GPUs, not two nodes."}
json.dump(report, open(sys.argv[1], "w"), indent=2)
print(json.dumps(report, indent=2))
if not report["distinct_uuids"]:
    sys.exit("GPU UUIDs are not distinct; refusing to continue")
PY

echo "[2/8] workers"
start_worker() {  # name gpu port
  CUDA_VISIBLE_DEVICES="$2" TENSORPILOT_API_KEY="$TENSORPILOT_API_KEY" python3 -m tensorpilot.server --backend cuda \
    --model "$TENSORPILOT_MODEL" --revision "$TENSORPILOT_REVISION" --cache-dir "$CACHE_DIR" \
    --host 127.0.0.1 --port "$3" >>"$RESULTS/$1.log" 2>&1 &
  echo $!
}
PID_A="$(start_worker worker-a 0 "$PORT_A")"; PIDS+=("$PID_A")
PID_B="$(start_worker worker-b 1 "$PORT_B")"; PIDS+=("$PID_B")
wait_for_ready "http://127.0.0.1:$PORT_A/readyz" "$PID_A" 900
wait_for_ready "http://127.0.0.1:$PORT_B/readyz" "$PID_B" 900

echo "[3/8] gateway"
TENSORPILOT_WORKER_API_KEY="$TENSORPILOT_API_KEY" TENSORPILOT_CONTROL_API_KEY="$TENSORPILOT_CONTROL_API_KEY" \
  python3 -m tensorpilot.control_plane --host 127.0.0.1 --port "$CONTROL_PORT" --policy capacity-aware \
  --worker "worker-a=http://127.0.0.1:$PORT_A" --worker "worker-b=http://127.0.0.1:$PORT_B" \
  >"$RESULTS/control.log" 2>&1 &
PIDS+=($!)
wait_for_ready "http://127.0.0.1:$CONTROL_PORT/readyz" "${PIDS[-1]}" 120
TENSORPILOT_CONTROL_API_KEY="$TENSORPILOT_CONTROL_API_KEY" TENSORPILOT_DATA_API_KEY="$TENSORPILOT_DATA_API_KEY" \
  python3 -m tensorpilot.data_plane --host 127.0.0.1 --port "$DATA_PORT" \
  --control-url "http://127.0.0.1:$CONTROL_PORT" >"$RESULTS/data.log" 2>&1 &
PIDS+=($!)
wait_for_ready "http://127.0.0.1:$DATA_PORT/readyz" "${PIDS[-1]}" 120

ctl() { curl -fsS --max-time 5 -H "Authorization: Bearer $TENSORPILOT_CONTROL_API_KEY" "http://127.0.0.1:$CONTROL_PORT$1"; }
snapshot() { ctl /v1/workers >"$RESULTS/workers.$1.json" || echo '{"error":"control plane unreachable"}' >"$RESULTS/workers.$1.json"; }
worker_metrics() { for p in "$PORT_A:a" "$PORT_B:b"; do
  curl -fsS --max-time 5 -H "Authorization: Bearer $TENSORPILOT_API_KEY" "http://127.0.0.1:${p%%:*}/metrics" \
    >"$RESULTS/worker-${p##*:}.metrics.$1.txt" 2>/dev/null || true; done; }

# Continuous per-GPU samples for the whole load section (both GPUs, 1 s).
nvidia-smi --query-gpu=timestamp,index,uuid,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
  --format=csv -l 1 >"$RESULTS/gpu-samples.csv" 2>&1 &
SAMPLER=$!

load() {  # label url key extra-args...
  local label="$1" url="$2" key="$3"; shift 3
  TENSORPILOT_API_KEY="$key" timeout "$PHASE_TIMEOUT" python3 -m tensorpilot.load --url "$url" \
    --requests "$REQUESTS" --concurrency "$CONCURRENCY" --max-tokens 64 --timeout 120 --seed 7 \
    --output "$RESULTS/$label.json" --journal "$RESULTS/$label.requests.jsonl" "$@" || echo "phase $label exited $?" >&2
}

echo "[4/8] phase A: single-GPU reference (worker-a only)"
worker_metrics before-a
load phase-a-single "http://127.0.0.1:$PORT_A" "$TENSORPILOT_API_KEY"
worker_metrics after-a

echo "[5/8] phase B: two GPUs through the gateway"
snapshot before-b; worker_metrics before-b
load phase-b-two-gpu "http://127.0.0.1:$DATA_PORT" "$TENSORPILOT_DATA_API_KEY"
snapshot after-b; worker_metrics after-b

echo "[6/8] phase C: kill -9 worker-b during load"
load phase-c-failure "http://127.0.0.1:$DATA_PORT" "$TENSORPILOT_DATA_API_KEY" &
LOAD_PID=$!
sleep 10
KILL_AT="$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)"
kill -9 "$PID_B" 2>/dev/null || true
echo "{\"killed\": \"worker-b\", \"pid\": $PID_B, \"at\": \"$KILL_AT\"}" >"$RESULTS/failure-injection.json"
for i in $(seq 1 30); do
  snapshot "failure-$i"; sleep 1
done
wait "$LOAD_PID" || true

echo "[7/8] recovery: restart worker-b"
PID_B="$(start_worker worker-b 1 "$PORT_B")"; PIDS+=("$PID_B")
wait_for_ready "http://127.0.0.1:$PORT_B/readyz" "$PID_B" 900
for i in $(seq 1 30); do
  snapshot "recovery-$i"; sleep 2
done
load phase-d-recovery "http://127.0.0.1:$DATA_PORT" "$TENSORPILOT_DATA_API_KEY"
worker_metrics final
kill "$SAMPLER" 2>/dev/null || true; SAMPLER=""

echo "[8/8] secret scan and evidence archive"
for key in "$TENSORPILOT_API_KEY" "$TENSORPILOT_CONTROL_API_KEY" "$TENSORPILOT_DATA_API_KEY"; do
  if grep -rqF -- "$key" "$RESULTS"; then
    echo "a credential value appears in the evidence directory; refusing to archive" >&2
    exit 1
  fi
done
ARCHIVE="$RESULTS.tar.gz"
tar -C "$(dirname "$RESULTS")" -czf "$ARCHIVE" "$(basename "$RESULTS")"
sha256sum "$ARCHIVE" | tee "$ARCHIVE.sha256"
echo "archive: $ARCHIVE"
