#!/usr/bin/env bash
# Capture host and GPU diagnostics. Each probe is independent so one failure does not hide the rest.
set -uo pipefail

OUT="${1:?usage: diagnostics.sh <output-directory>}"
mkdir -p "$OUT"

{ date -u; uname -a; grep -E '^(PRETTY_NAME|VERSION_ID)=' /etc/os-release 2>/dev/null; } >"$OUT/system.txt"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi >"$OUT/nvidia-smi.txt" 2>&1
  nvidia-smi -q >"$OUT/nvidia-smi-query.txt" 2>&1
  nvidia-smi --query-gpu=index,name,driver_version,compute_cap,memory.total,clocks.max.sm,power.limit \
    --format=csv >"$OUT/gpu-inventory.csv" 2>&1
else
  echo "nvidia-smi not found" >"$OUT/nvidia-smi.txt"
fi
python3 -c 'import platform, sys; print(sys.version); print(platform.platform())' >"$OUT/python.txt" 2>&1
python3 -c 'import torch; print("torch", torch.__version__, "cuda_runtime", torch.version.cuda, "available", torch.cuda.is_available())' \
  >"$OUT/torch.txt" 2>&1
python3 -m pip freeze >"$OUT/pip-freeze.txt" 2>&1
df -h >"$OUT/disk.txt" 2>&1
free -h >"$OUT/memory.txt" 2>&1 || true
exit 0
