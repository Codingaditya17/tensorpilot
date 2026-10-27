#!/usr/bin/env bash
# Prove that Triton can build its CUDA helper module on this host, before any billed vLLM work depends on it.
#
#   check_toolchain.sh                       # a C compiler, libc headers and Python.h: compile and link a shared object
#   check_toolchain.sh --gpu /path/to/venv/bin/python
#                                            # the same, then on the GPU: build a CUDA_HOME from the venv (cuda_home.sh), launch one
#                                            # Triton kernel, and compile and run FlashInfer's sampling kernel with nvcc
#
# vLLM 0.31.0 compiles the model with torch.compile (Inductor), which generates Triton kernels. Triton builds a small C module
# (cuda_utils) with the system C compiler the first time it touches the driver. The base image is python:3.11-slim, which has
# neither a compiler nor libc headers, so the 7B session's vLLM stage died at that point after two hours of billed time:
# "RuntimeError: Failed to find C compiler. Please specify via CC environment variable". This check fails in seconds instead.
# The second attempt got past that and died in the engine's kernel warm-up, where FlashInfer needs nvcc to compile its sampling kernel:
# "Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist". --gpu now runs that exact kernel too.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
want_gpu=0
if [[ "${1:-}" == "--gpu" ]]; then want_gpu=1; shift; fi
python="${1:-python3}"

die() { echo "check_toolchain: $*" >&2; exit 2; }

cc_bin="${CC:-cc}"
command -v "$cc_bin" >/dev/null 2>&1 || die "no C compiler: '$cc_bin' is not on PATH. Triton cannot build its CUDA helper without one (the image needs gcc and libc6-dev, or set CC)"
include="$("$python" -c 'import sysconfig; print(sysconfig.get_paths()["include"])')" || die "cannot run $python"
[[ -f "$include/Python.h" ]] || die "Python.h is missing from $include"

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
cat >"$work/probe.c" <<'C'
#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
int tensorpilot_toolchain_probe(void) { return (int)sizeof(PyObject) + (int)strlen("x") + (printf("") == 0); }
C
# -shared -fPIC -O3 -I<python include> mirrors how Triton builds its module, so a libc-headers or linker gap shows up here.
"$cc_bin" -shared -fPIC -O3 -I"$include" "$work/probe.c" -o "$work/probe.so" 2>"$work/cc.err" ||
  die "the C compiler '$cc_bin' cannot build a Python-header shared object: $(head -c 400 "$work/cc.err")"
echo "toolchain: $("$cc_bin" --version | head -1); Python.h and libc headers present; shared object builds"

# nvcc compiles every .cu file's host half with a C++ compiler, and FlashInfer builds its .cpp files with `c++`. Debian's `gcc` package
# does not include g++, so the C compiler passing says nothing about this.
cxx_bin="${CXX:-c++}"
command -v "$cxx_bin" >/dev/null 2>&1 || die "no C++ compiler: '$cxx_bin' is not on PATH. nvcc and FlashInfer's JIT build need one (the image needs g++, or set CXX)"
cat >"$work/probe.cpp" <<'CPP'
#include <cstdio>
#include <string>
#include <vector>
extern "C" int tensorpilot_cxx_probe() { std::vector<std::string> v(2, "x"); return (int)v.size() + (std::printf("") == 0); }
CPP
"$cxx_bin" -shared -fPIC -O2 "$work/probe.cpp" -o "$work/probe_cxx.so" 2>"$work/cxx.err" ||
  die "the C++ compiler '$cxx_bin' cannot build a shared object with the standard library: $(head -c 400 "$work/cxx.err")"
echo "toolchain: $("$cxx_bin" --version | head -1); C++ standard library present"

if (( want_gpu )); then
  venv="$(cd "$(dirname "$python")/.." && pwd)"
  CUDA_HOME="$("$HERE/cuda_home.sh" "$venv")" || die "no usable CUDA_HOME could be assembled from $venv (FlashInfer needs nvcc)"
  export CUDA_HOME PATH="$venv/bin:$CUDA_HOME/bin:$PATH"   # ninja lives in the venv; FlashInfer calls it by name
  echo "cuda_home: $CUDA_HOME ($("$CUDA_HOME/bin/nvcc" --version | tail -n 1))"
  # The whole nvcc pipeline (front end, cicc, ptxas, host compiler, linker) on a kernel, for the card's architecture. No GPU is involved.
  cat >"$work/probe.cu" <<'CU'
#include <cuda_runtime.h>
__global__ void tensorpilot_probe_kernel(float *x) { x[threadIdx.x] += 1.0f; }
extern "C" void tensorpilot_probe(float *x) { tensorpilot_probe_kernel<<<1, 32>>>(x); }
CU
  "$CUDA_HOME/bin/nvcc" -shared -Xcompiler -fPIC -gencode=arch=compute_89,code=sm_89 -I"$CUDA_HOME/include" \
    -L"$CUDA_HOME/lib64" -lcudart "$work/probe.cu" -o "$work/probe_cu.so" 2>"$work/nvcc.err" ||
    die "nvcc cannot build a CUDA shared object (the step FlashInfer's JIT runs): $(head -c 600 "$work/nvcc.err")"
  echo "nvcc: builds a CUDA shared object for sm_89"
  "$python" "$HERE/triton_smoke.py" || die "a Triton kernel could not be compiled and run on the GPU with $python"
  "$python" "$HERE/flashinfer_smoke.py" || die "FlashInfer's sampling kernel could not be compiled and run on the GPU with $python"
fi
