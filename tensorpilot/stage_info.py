"""Who ran this stage, on what: written into every stage's evidence so a comparison can prove it used one physical card.

It records the GPU's UUID (nvidia-smi), the Pod id, the image build information baked in at build time, and the image
reference the operator declared. Environment variables are read from an allow-list by name; nothing else is copied, so no
credential can end up here.

    python -m tensorpilot.stage_info --stage 7b-bf16-prod --output STAGE_DIR/stage-info.json
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "tensorpilot.stage-info/v1"
BUILD_INFO = Path(os.environ.get("TENSORPILOT_BUILD_INFO", "/opt/tensorpilot/BUILD_INFO.json"))
ENV_ALLOWLIST = ("RUNPOD_POD_ID", "RUNPOD_GPU_COUNT", "TENSORPILOT_IMAGE_REF", "TENSORPILOT_POD_MAX_MINUTES",
                 "SESSION_BUDGET_MINUTES", "PRICE_PER_HOUR_USD", "PREFILL_PACK_TOKENS")


def gpus():
    """GPU name, UUID, driver and memory per device, or [] when nvidia-smi is unavailable."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    devices = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 4:
            devices.append({"name": parts[0], "uuid": parts[1], "driver_version": parts[2], "memory_total_mib": parts[3]})
    return devices


def build_info():
    try:
        return json.loads(BUILD_INFO.read_text())
    except (OSError, ValueError):
        return None


def collect(stage):
    return {"schema": SCHEMA, "stage": stage, "at_utc": datetime.now(timezone.utc).isoformat(), "gpus": gpus(),
            "build_info": build_info(), "environment": {name: os.environ[name] for name in ENV_ALLOWLIST if name in os.environ}}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    Path(args.output).write_text(json.dumps(collect(args.stage), indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
