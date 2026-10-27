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


def lock_versions():
    text = (ROOT / "requirements-vllm.txt").read_text()
    return {name: version for name, version in re.findall(r"^([A-Za-z0-9_.-]+)==([^\s\\]+)", text, re.M)}


def test_the_cuda_compiler_set_is_one_release_and_matches_the_runtime_headers():
    """The third vLLM failure class: nvcc 13.4.92 beside CUDA 13.0.96 headers. CCCL, which FlashInfer's JIT build includes, refuses
    that mix ("CUDA compiler and CUDA toolkit headers are incompatible"). nvcc, crt and nvvm ship together, and their major.minor
    must equal the runtime's (the headers FlashInfer compiles against) and the cuda-toolkit pin torch brought in."""
    pins = lock_versions()
    compiler = {pins[n] for n in ("nvidia-cuda-nvcc", "nvidia-cuda-crt", "nvidia-nvvm")}
    assert len(compiler) == 1, f"nvcc, crt and nvvm must be the same release, found {compiler}"
    major_minor = lambda v: tuple(v.split(".")[:2])  # noqa: E731
    assert major_minor(compiler.pop()) == major_minor(pins["nvidia-cuda-runtime"]) == major_minor(pins["cuda-toolkit"])
    assert pins["nvidia-cuda-nvcc"] == "13.0.88", "the 13.0.3 toolkit's compiler release"


def test_the_nvcc_pins_are_declared_in_the_input_file_so_a_relock_cannot_drift_them():
    text = (ROOT / "requirements-vllm.in").read_text()
    for pin in ("nvidia-cuda-nvcc==13.0.88", "nvidia-cuda-crt==13.0.88", "nvidia-nvvm==13.0.88"):
        assert pin in text


def test_the_lock_carries_what_flashinfer_and_triton_build_with():
    pins = lock_versions()
    for needed in ("ninja", "flashinfer-python", "nvidia-cuda-nvcc", "nvidia-cuda-runtime", "triton"):
        assert needed in pins, f"{needed} is required by the vLLM start-up path and is missing from the lock"
