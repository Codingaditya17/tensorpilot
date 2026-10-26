"""The vLLM comparison environment: its hashed lock and the installer's guards. Nothing is installed."""
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_the_vllm_lock_hashes_every_package_and_pins_the_expected_versions():
    text = (ROOT / "requirements-vllm.txt").read_text()
    pins = re.findall(r"^([A-Za-z0-9_.-]+)==([^\s\\]+) \\\n((?:    --hash=sha256:[0-9a-f]{64}(?: \\)?\n)+)", text, re.M)
    names = {name: version for name, version, _ in pins}
    assert len(pins) == len(re.findall(r"^[A-Za-z0-9_.-]+==", text, re.M)) > 150, "every pin must carry hashes"
    assert names["vllm"] == "0.31.0" and names["torch"] == "2.13.0" and names["bitsandbytes"] == "0.50.2"
    assert names["transformers"].startswith("5.17"), "vLLM's own transformers, isolated from TensorPilot's 5.20.1"


def test_the_installer_plans_without_installing_and_refuses_to_overwrite(tmp_path):
    script = ROOT / "deploy" / "runpod" / "install_vllm.sh"
    plan = subprocess.run(["bash", str(script), "--plan"], capture_output=True, text=True,
                          env={**os.environ, "VLLM_VENV": str(tmp_path / "venv")})
    assert plan.returncode == 0 and "isolated" in plan.stdout and "driver 580" in plan.stdout
    assert not (tmp_path / "venv").exists()
    (tmp_path / "venv").mkdir()
    refused = subprocess.run(["bash", str(script)], capture_output=True, text=True,
                             env={**os.environ, "VLLM_VENV": str(tmp_path / "venv"), "V10_SKIP_GPU_CHECKS": "1"})
    assert refused.returncode == 2 and "already exists" in refused.stderr
