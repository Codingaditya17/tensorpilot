#!/usr/bin/env bash
# Prove that Triton can build its CUDA helper module on this host, before any billed vLLM work depends on it.
#
#   check_toolchain.sh                       # a C compiler, libc headers and Python.h: compile and link a shared object
#   check_toolchain.sh --gpu /path/to/python # the same, then JIT-compile and launch one Triton kernel on the GPU
#
# vLLM 0.31.0 compiles the model with torch.compile (Inductor), which generates Triton kernels. Triton builds a small C module
# (cuda_utils) with the system C compiler the first time it touches the driver. The base image is python:3.11-slim, which has
# neither a compiler nor libc headers, so the 7B session's vLLM stage died at that point after two hours of billed time:
# "RuntimeError: Failed to find C compiler. Please specify via CC environment variable". This check fails in seconds instead.
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

if (( want_gpu )); then
  "$python" "$HERE/triton_smoke.py" || die "a Triton kernel could not be compiled and run on the GPU with $python"
fi
