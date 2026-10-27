#!/usr/bin/env bash
# Print a CUDA_HOME for the vLLM environment, built from the CUDA components already in its hash-locked virtualenv.
#
#   cuda_home.sh /workspace/vllm-venv        # prints /workspace/vllm-venv/cuda-home
#
# FlashInfer, which vLLM uses for top-k/top-p sampling, compiles its kernels with nvcc the first time they are needed. It finds nvcc
# through CUDA_HOME, `which nvcc` or /usr/local/cuda. The Pod image has no CUDA toolkit (the runtime comes from pip wheels), so the 7B
# session's second vLLM attempt died in the engine's kernel warm-up with "Could not find nvcc and default cuda_home='/usr/local/cuda'
# doesn't exist". The lock already pins nvidia-cuda-nvcc, -crt, -nvvm, -runtime and -cccl, which all install into one tree,
# site-packages/nvidia/cu13/{bin,include,lib,nvvm}. FlashInfer's build wants the conventional layout (lib64, an unversioned libcudart.so,
# lib64/stubs/libcuda.so), so this assembles that layout from symlinks. Nothing is downloaded and nothing outside the venv is touched.
set -euo pipefail

die() { echo "cuda_home: $*" >&2; exit 2; }
venv="${1:-}"
[[ -x "$venv/bin/python" ]] || die "usage: cuda_home.sh VENV (no python at '$venv/bin/python')"
site="$("$venv/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
root="$site/nvidia/cu13"
for need in bin/nvcc include/cuda_runtime.h nvvm/libdevice nvvm/bin/cicc; do
  [[ -e "$root/$need" ]] || die "$root/$need is missing: the venv lacks the CUDA 13 component that provides it (nvidia-cuda-nvcc, -runtime, -nvvm)"
done
cudart="$(ls "$root"/lib/libcudart.so.[0-9]* 2>/dev/null | head -n 1 || true)"
[[ -n "$cudart" ]] || die "no libcudart.so.N in $root/lib"

# The driver library for -lcuda. A Pod gets libcuda.so.1 from the NVIDIA container runtime; there is no libcuda.so development symlink.
libcuda="${CUDA_HOME_LIBCUDA:-}"
if [[ -z "$libcuda" ]]; then
  libcuda="$(ldconfig -p 2>/dev/null | awk '/libcuda\.so\.1 / && /x86-64|x86_64|libc6,/ {print $NF; exit}' || true)"
fi
[[ -n "$libcuda" && -e "$libcuda" ]] || die "libcuda.so.1 was not found (is the NVIDIA driver visible in this container?); set CUDA_HOME_LIBCUDA to override"

home="${VLLM_CUDA_HOME:-$venv/cuda-home}"
rm -rf "$home"
mkdir -p "$home/lib64/stubs"
ln -s "$root/bin" "$home/bin"
ln -s "$root/include" "$home/include"
ln -s "$root/nvvm" "$home/nvvm"
for f in "$root"/lib/*; do ln -s "$f" "$home/lib64/$(basename "$f")"; done
ln -sf "$cudart" "$home/lib64/libcudart.so"
ln -s "$libcuda" "$home/lib64/stubs/libcuda.so"

"$home/bin/nvcc" --version >/dev/null 2>&1 || die "$home/bin/nvcc does not run"
echo "$home"
