"""The vLLM stage needs a C compiler (Triton builds a helper module with it). The 7B session lost its vLLM stage, after two hours of
billed time, to a base image that had none. These tests pin the fix: the image installs one, and every vLLM entry point proves it
works in seconds, before the expensive steps."""
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNPOD = ROOT / "deploy" / "runpod"
CHECK = RUNPOD / "check_toolchain.sh"
needs_cc = pytest.mark.skipif(shutil.which("cc") is None, reason="needs a C compiler on the test host")


def run(args, tmp_path, **extra):
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path), "TMPDIR": str(tmp_path), **extra}
    return subprocess.run(args, capture_output=True, text=True, env=env, cwd=tmp_path)


def script(tmp_path, body, name):
    path = tmp_path / name
    path.write_text("#!/usr/bin/env bash\n" + body)
    path.chmod(0o755)
    return str(path)


def test_the_experiment_image_installs_a_c_compiler_and_libc_headers():
    """The root cause: python:3.11-slim has no gcc and no libc headers, so Triton could not build its CUDA helper module."""
    text = (ROOT / "deploy" / "docker" / "Dockerfile.experiment").read_text()
    apt = re.search(r"apt-get install[^\n]*(?:\\\n[^\n]*)*", text).group(0)
    assert re.search(r"\bgcc\b", apt) and re.search(r"\blibc6-dev\b", apt), apt
    assert re.search(r"(?<!\S)g\+\+(?!\S)", apt), "nvcc and FlashInfer's JIT need a C++ compiler; Debian's gcc package does not include g++"
    assert "--no-install-recommends" in apt


@needs_cc
def test_a_working_toolchain_passes(tmp_path):
    done = run(["bash", str(CHECK), sys.executable], tmp_path)
    assert done.returncode == 0, done.stderr
    assert "shared object builds" in done.stdout


def test_a_missing_compiler_is_refused_with_the_reason(tmp_path):
    done = run(["bash", str(CHECK), sys.executable], tmp_path, CC=str(tmp_path / "no-such-compiler"))
    assert done.returncode == 2
    assert "no C compiler" in done.stderr and "gcc and libc6-dev" in done.stderr


@needs_cc
def test_a_missing_cxx_compiler_is_refused_even_though_the_c_compiler_works(tmp_path):
    """Debian's `gcc` has no g++: the C check passes and nvcc's host compile would still fail."""
    done = run(["bash", str(CHECK), sys.executable], tmp_path, CXX=str(tmp_path / "no-such-c++"))
    assert done.returncode == 2 and "no C++ compiler" in done.stderr


@needs_cc
def test_a_cxx_compiler_without_its_standard_library_is_refused(tmp_path):
    cxx = script(tmp_path, 'echo "fatal error: vector: No such file or directory" >&2\nexit 1\n', "fake-c++")
    done = run(["bash", str(CHECK), sys.executable], tmp_path, CXX=cxx)
    assert done.returncode == 2 and "vector: No such file" in done.stderr


def test_a_compiler_that_cannot_build_is_refused_with_its_own_message(tmp_path):
    """A compiler without libc headers, as in the slim image after only `gcc` is added: it exists but cannot build."""
    cc = script(tmp_path, 'echo "fatal error: stdio.h: No such file or directory" >&2\nexit 1\n', "fake-cc")
    done = run(["bash", str(CHECK), sys.executable], tmp_path, CC=cc)
    assert done.returncode == 2 and "stdio.h: No such file" in done.stderr


def fake_venv(tmp_path, smoke_status=None, with_nvcc=True):
    """A virtualenv-shaped directory whose python records the smoke scripts it is asked to run, with the CUDA 13 wheel tree."""
    venv = tmp_path / "venv"
    site = venv / "lib" / "site-packages"
    root = site / "nvidia" / "cu13"
    for d in ("bin", "include", "lib", "nvvm/libdevice", "nvvm/bin"):
        (root / d).mkdir(parents=True)
    if with_nvcc:
        script(root / "bin", 'echo "Cuda compilation tools, release 13.4, V13.4.92"\n', "nvcc")
    (root / "include" / "cuda_runtime.h").write_text("")
    (root / "nvvm" / "bin" / "cicc").write_text("")
    (root / "lib" / "libcudart.so.13").write_text("")
    (venv / "bin").mkdir()
    seen = tmp_path / "seen.txt"
    status = smoke_status or {}
    body = (f'if [[ "$1" == "-c" ]]; then if [[ "$2" == *purelib* ]]; then echo "{site}"; exit 0; fi; exec "{sys.executable}" "$@"; fi\n'
            f'name="$(basename "$1")"; echo "$name CUDA_HOME=${{CUDA_HOME:-}} PATH=$PATH" >> "{seen}"\n'
            'case "$name" in\n'
            + "".join(f'  {n}) exit {c};;\n' for n, c in status.items())
            + "esac\nexit 0\n")
    script(venv / "bin", body, "python")
    libcuda = tmp_path / "libcuda.so.1"
    libcuda.write_text("")
    return venv, seen, str(libcuda)


@needs_cc
def test_the_gpu_option_builds_a_cuda_home_and_runs_both_smokes_with_it(tmp_path):
    venv, seen, libcuda = fake_venv(tmp_path)
    done = run(["bash", str(CHECK), "--gpu", str(venv / "bin" / "python")], tmp_path, CUDA_HOME_LIBCUDA=libcuda)
    assert done.returncode == 0, done.stderr
    lines = seen.read_text().splitlines()
    assert [line.split()[0] for line in lines] == ["triton_smoke.py", "flashinfer_smoke.py"]
    assert "nvcc: builds a CUDA shared object" in done.stdout
    home = str(venv / "cuda-home")
    assert all(f"CUDA_HOME={home}" in line for line in lines)
    assert all(f"PATH={venv}/bin:{home}/bin:" in line for line in lines), "ninja lives in the venv; nvcc in the CUDA home"


@needs_cc
@pytest.mark.parametrize("failing, message", [("triton_smoke.py", "Triton kernel could not be compiled"),
                                              ("flashinfer_smoke.py", "FlashInfer's sampling kernel could not be compiled")])
def test_a_failing_gpu_smoke_fails_the_check_and_names_it(tmp_path, failing, message):
    venv, _, libcuda = fake_venv(tmp_path, {failing: 1})
    done = run(["bash", str(CHECK), "--gpu", str(venv / "bin" / "python")], tmp_path, CUDA_HOME_LIBCUDA=libcuda)
    assert done.returncode == 2 and message in done.stderr


@needs_cc
def test_the_gpu_option_refuses_a_venv_without_nvcc_before_any_smoke(tmp_path):
    """The second vLLM failure: FlashInfer found no nvcc. The check must fail on the missing compiler, not later in the engine."""
    venv, seen, libcuda = fake_venv(tmp_path, with_nvcc=False)
    done = run(["bash", str(CHECK), "--gpu", str(venv / "bin" / "python")], tmp_path, CUDA_HOME_LIBCUDA=libcuda)
    assert done.returncode == 2 and "bin/nvcc is missing" in done.stderr and not seen.exists()


def test_cuda_home_assembles_the_conventional_layout_from_the_wheel_tree(tmp_path):
    venv, _, libcuda = fake_venv(tmp_path)
    done = run(["bash", str(RUNPOD / "cuda_home.sh"), str(venv)], tmp_path, CUDA_HOME_LIBCUDA=libcuda)
    assert done.returncode == 0, done.stderr
    home = Path(done.stdout.strip())
    assert home == venv / "cuda-home"
    assert (home / "bin" / "nvcc").exists() and (home / "include" / "cuda_runtime.h").exists()
    assert (home / "nvvm" / "libdevice").is_dir()
    assert (home / "lib64" / "libcudart.so").resolve().name == "libcudart.so.13", "-lcudart needs the unversioned name"
    assert (home / "lib64" / "stubs" / "libcuda.so").resolve() == Path(libcuda).resolve(), "-lcuda needs a libcuda.so"


def test_cuda_home_is_rebuilt_not_trusted_when_it_already_exists(tmp_path):
    venv, _, libcuda = fake_venv(tmp_path)
    (venv / "cuda-home").mkdir()
    (venv / "cuda-home" / "stale").write_text("")
    run(["bash", str(RUNPOD / "cuda_home.sh"), str(venv)], tmp_path, CUDA_HOME_LIBCUDA=libcuda)
    assert not (venv / "cuda-home" / "stale").exists()


@pytest.mark.parametrize("missing", ["bin/nvcc", "include/cuda_runtime.h", "nvvm/bin/cicc", "lib/libcudart.so.13"])
def test_cuda_home_names_the_missing_component(tmp_path, missing):
    venv, _, libcuda = fake_venv(tmp_path)
    (venv / "lib" / "site-packages" / "nvidia" / "cu13" / missing).unlink()
    done = run(["bash", str(RUNPOD / "cuda_home.sh"), str(venv)], tmp_path, CUDA_HOME_LIBCUDA=libcuda)
    assert done.returncode == 2 and ("is missing" in done.stderr or "no libcudart" in done.stderr)


def test_cuda_home_requires_the_driver_library(tmp_path):
    venv, _, _ = fake_venv(tmp_path)
    done = run(["bash", str(RUNPOD / "cuda_home.sh"), str(venv)], tmp_path, CUDA_HOME_LIBCUDA=str(tmp_path / "absent.so"))
    assert done.returncode == 2 and "libcuda.so.1 was not found" in done.stderr


def test_the_triton_smoke_checks_cuda_and_the_result():
    source = (RUNPOD / "triton_smoke.py").read_text()
    compile(source, "triton_smoke.py", "exec")
    assert "torch.cuda.is_available()" in source and "torch.equal" in source


def test_the_installer_plan_and_install_refuse_before_downloading_anything(tmp_path):
    install = RUNPOD / "install_vllm.sh"
    venv = tmp_path / "venv"
    bad = {"CC": str(tmp_path / "no-such-compiler"), "VLLM_VENV": str(venv), "V10_SKIP_GPU_CHECKS": "1"}
    plan = run(["bash", str(install), "--plan"], tmp_path, **bad)
    assert plan.returncode == 2 and "no C compiler" in plan.stdout
    real = run(["bash", str(install)], tmp_path, **bad)
    assert real.returncode == 2 and "no C compiler" in real.stderr
    assert not venv.exists(), "nothing may be created or downloaded when the toolchain is missing"


STUB_PYTHON = '''if [[ "$*" == *--help=all* ]]; then echo "usage: --generation-config --no-enable-prefix-caching --quantization bitsandbytes"; exit 0; fi
if [[ "$1" == "-m" && "$2" == "pip" ]]; then echo "vllm==0.31.0"; exit 0; fi
exec "%s" "$@"
'''


def guards(tmp_path, **extra):
    venv = tmp_path / "vllm-venv" / "bin"
    venv.mkdir(parents=True)
    script(venv, STUB_PYTHON % sys.executable, "python")
    stage = tmp_path / "stage"
    stage.mkdir()
    call = f'source "{RUNPOD / "scale_session.sh"}"; STAGE_DIR="{stage}"; V10_QUANT=none; vllm_guards'
    done = run(["bash", "-c", call], tmp_path, VLLM_VENV=str(tmp_path / "vllm-venv"), V10_TEST_MODE="1",
               V10_SKIP_GPU_CHECKS="1", **extra)
    return done, stage


def test_the_vllm_stage_guard_stops_on_a_missing_compiler_before_the_engine_starts(tmp_path):
    done, stage = guards(tmp_path, CC=str(tmp_path / "no-such-compiler"))
    assert done.returncode != 0
    assert "toolchain check failed" in done.stderr and "no C compiler" in done.stderr
    assert "no C compiler" in (stage / "toolchain.txt").read_text()
    assert not (stage / "vllm-help.txt").exists(), "the guard must stop before it asks vLLM anything"


@needs_cc
def test_the_vllm_stage_guard_passes_with_a_working_toolchain(tmp_path):
    done, stage = guards(tmp_path)
    assert done.returncode == 0, done.stderr
    assert "shared object builds" in (stage / "toolchain.txt").read_text()


def test_the_vllm_command_turns_prefix_caching_off_so_replayed_prompts_do_not_inflate_the_comparison(tmp_path):
    """Every repeat of an experiment replays the same prompts; with vLLM's default prefix cache the second repeat measures cache hits."""
    call = (f'source "{RUNPOD / "scale_session.sh"}"; V10_MODEL_REPO=m V10_REVISION=r CACHE=/c V10_DTYPE=bfloat16 '
            'V10_MAX_PROMPT_TOKENS=1024 V10_BATCH_SIZE=48 PORT=8081 V10_QUANT=none; engine_command_vllm')
    done = run(["bash", "-c", call], tmp_path, VLLM_VENV=str(tmp_path))
    words = done.stdout.split()
    assert "--no-enable-prefix-caching" in words and "--enable-prefix-caching" not in words
    assert words[words.index("--max-num-seqs") + 1] == "48" and words[words.index("--max-model-len") + 1] == "1536"


def test_a_vllm_build_without_the_prefix_caching_switch_is_refused_by_the_guard(tmp_path):
    venv = tmp_path / "vllm-venv" / "bin"
    venv.mkdir(parents=True)
    script(venv, 'if [[ "$*" == *--help=all* ]]; then echo "usage: --generation-config"; exit 0; fi\nexec "%s" "$@"\n' % sys.executable,
           "python")
    stage = tmp_path / "stage"
    stage.mkdir()
    call = f'source "{RUNPOD / "scale_session.sh"}"; STAGE_DIR="{stage}"; V10_QUANT=none; vllm_guards'
    done = run(["bash", "-c", call], tmp_path, VLLM_VENV=str(tmp_path / "vllm-venv"), V10_TEST_MODE="1", V10_SKIP_GPU_CHECKS="1")
    assert done.returncode != 0 and "no --no-enable-prefix-caching" in done.stderr


def test_the_flashinfer_smoke_runs_the_sampler_the_engine_warmup_runs():
    source = (RUNPOD / "flashinfer_smoke.py").read_text()
    compile(source, "flashinfer_smoke.py", "exec")
    assert "top_k_top_p_sampling_from_logits" in source and "torch.cuda.is_available()" in source


@pytest.mark.skipif(subprocess.run(["bash", "-c", "type mapfile"], capture_output=True).returncode != 0,
                    reason="scale_session.sh needs bash 4 (mapfile); the Pod and CI have it, the macOS system bash does not")
def test_the_engine_is_started_with_the_cuda_home_and_the_venv_on_path(tmp_path):
    """FlashInfer reads CUDA_HOME (or `which nvcc`) and runs `ninja` by name; the engine process must see both."""
    log = tmp_path / "stage"
    log.mkdir()
    call = (f'source "{RUNPOD / "scale_session.sh"}"; STAGE_DIR="{log}"; VLLM_CUDA_HOME=/opt/ch; '
            f'VLLM_VENV="{tmp_path}/v"; TENSORPILOT_API_KEY=sekrit; start_engine vllm; wait "$ENGINE_PID"')
    done = run(["bash", "-c", call], tmp_path, V10_TEST_MODE="1", V10_VLLM_CMD="/usr/bin/env")
    assert done.returncode == 0, done.stderr
    environment = (log / "engine.log").read_text().splitlines()
    assert "CUDA_HOME=/opt/ch" in environment and "VLLM_API_KEY=sekrit" in environment
    path = next(line for line in environment if line.startswith("PATH="))[5:].split(":")
    assert path[:2] == [f"{tmp_path}/v/bin", "/opt/ch/bin"]
