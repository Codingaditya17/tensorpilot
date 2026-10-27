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


def test_a_compiler_that_cannot_build_is_refused_with_its_own_message(tmp_path):
    """A compiler without libc headers, as in the slim image after only `gcc` is added: it exists but cannot build."""
    cc = script(tmp_path, 'echo "fatal error: stdio.h: No such file or directory" >&2\nexit 1\n', "fake-cc")
    done = run(["bash", str(CHECK), sys.executable], tmp_path, CC=cc)
    assert done.returncode == 2 and "stdio.h: No such file" in done.stderr


@needs_cc
def test_the_gpu_option_launches_the_triton_smoke_and_fails_when_it_fails(tmp_path):
    seen = tmp_path / "seen.txt"
    for status in (0, 1):
        python = script(tmp_path, f'if [[ "$1" == *triton_smoke.py ]]; then echo "$1" >> "{seen}"; exit {status}; fi\n'
                                  f'exec "{sys.executable}" "$@"\n', f"python{status}")
        done = run(["bash", str(CHECK), "--gpu", python], tmp_path)
        assert done.returncode == status * 2, done.stderr
    assert seen.read_text().count("triton_smoke.py") == 2
    assert "could not be compiled and run on the GPU" in done.stderr


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
