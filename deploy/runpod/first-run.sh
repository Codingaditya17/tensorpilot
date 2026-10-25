#!/usr/bin/env bash
# The first experiment, hard-capped. Generates the three API keys in this shell if they are not set,
# then runs session.sh under `timeout`. session.sh archives partial evidence if the cap fires.
#   first-run.sh --plan   # validate and print; starts nothing
#   first-run.sh          # run
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
if [[ -z "${TENSORPILOT_API_KEY:-}" && -z "${TENSORPILOT_CONTROL_API_KEY:-}" && -z "${TENSORPILOT_DATA_API_KEY:-}" ]]; then
  eval "$("$ROOT/deploy/runpod/pod-keys.sh")"
fi
if [[ "${1:-}" == "--plan" ]]; then
  exec "$ROOT/deploy/runpod/session.sh" --plan
fi
echo "evidence root: ${TENSORPILOT_RESULTS_ROOT:-/workspace/results}; cache: ${TENSORPILOT_CACHE_DIR:-/workspace/huggingface}" >&2
echo "session cap ${TENSORPILOT_SESSION_CAP:-45m}; benchmark cap ${TENSORPILOT_BENCH_TIMEOUT:-25m}" >&2
exec timeout --signal=TERM --kill-after=60s "${TENSORPILOT_SESSION_CAP:-45m}" "$ROOT/deploy/runpod/session.sh"
