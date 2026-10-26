#!/usr/bin/env bash
# Install vLLM into an isolated virtualenv from the hash-locked requirements-vllm.txt.
#
#   install_vllm.sh --plan     # print what would happen; installs nothing
#   install_vllm.sh            # install, then verify the versions and that CUDA is visible
#
# vLLM 0.31.0 pins torch 2.13.0 with the CUDA 13.0 runtime and transformers 5.17, which conflict with TensorPilot's
# torch 2.9.1 (CUDA 12.8) and transformers 5.20.1. That is why it gets its own virtualenv and is never installed into
# the system Python. The CUDA 13.0 runtime needs a Linux driver of 580 or newer.
#
# This downloads several gigabytes and is billed Pod time; it is bounded by INSTALL_TIMEOUT (default 40m).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="${VLLM_VENV:-/workspace/vllm-venv}"
LOCK="$REPO_ROOT/requirements-vllm.txt"
TIMEOUT="${INSTALL_TIMEOUT:-40m}"
EXPECT_VLLM="0.31.0"
EXPECT_TORCH="2.13.0"
NEED_FREE_GIB=40

die() { echo "install_vllm: $*" >&2; exit 2; }

[[ -f "$LOCK" ]] || die "missing $LOCK"
PINS="$(grep -cE '^[A-Za-z0-9_.-]+==' "$LOCK")"

if [[ "${1:-}" == "--plan" ]]; then
  cat <<PLAN
venv:        $VENV (isolated; the system Python is not touched)
lock:        $LOCK ($PINS pinned packages, every one hash-verified, installed with --no-deps)
expects:     vllm $EXPECT_VLLM with torch $EXPECT_TORCH (CUDA 13.0 runtime, driver 580 or newer)
disk:        about 15 GiB for the venv; refuses below $NEED_FREE_GIB GiB free
time bound:  $TIMEOUT (billed Pod time)
PLAN
  exit 0
fi

if [[ "${V10_SKIP_GPU_CHECKS:-}" != "1" ]]; then
  command -v nvidia-smi >/dev/null || die "nvidia-smi not found: this is not a GPU Pod"
  driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
  (( ${driver%%.*} >= 580 )) || die "the CUDA 13.0 runtime needs driver 580 or newer; this host has $driver"
fi
mkdir -p "$(dirname "$VENV")"
free_gib=$(( $(df -Pk "$(dirname "$VENV")" | awk 'NR==2 {print $4}') / 1024 / 1024 ))
(( free_gib >= NEED_FREE_GIB )) || die "only ${free_gib} GiB free at $(dirname "$VENV"); need $NEED_FREE_GIB"
[[ ! -e "$VENV" ]] || die "$VENV already exists; remove it deliberately to reinstall"

python3 -m venv "$VENV"
timeout --signal=TERM --kill-after=60 "$TIMEOUT" \
  "$VENV/bin/python" -m pip install --quiet --disable-pip-version-check --no-cache-dir --require-hashes --no-deps -r "$LOCK"

read -r vllm torch < <("$VENV/bin/python" - <<'PY'
from importlib.metadata import version
print(version("vllm"), version("torch").split("+")[0])
PY
)
[[ "$vllm" == "$EXPECT_VLLM" && "$torch" == "$EXPECT_TORCH" ]] || die "installed vllm $vllm / torch $torch, expected $EXPECT_VLLM / $EXPECT_TORCH"
if [[ "${V10_SKIP_GPU_CHECKS:-}" != "1" ]]; then
  "$VENV/bin/python" -c "import torch; assert torch.cuda.is_available(), 'CUDA is not visible to the vLLM environment'"
fi
echo "vllm $vllm with torch $torch installed in $VENV" | tee "$VENV/INSTALL-OK"
