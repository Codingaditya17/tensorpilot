"""Compile and run one Triton kernel on the GPU, using the interpreter given on the command line.

This is the smallest thing that exercises what vLLM's torch.compile path needs: Triton's driver module (built with the system C
compiler), the kernel compiler and a launch. It takes seconds. It is run by check_toolchain.sh --gpu with the vLLM virtualenv's
Python, which is why it imports nothing from tensorpilot.
"""
import sys

import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    tl.store(out_ptr + offsets, tl.load(x_ptr + offsets, mask=mask) + tl.load(y_ptr + offsets, mask=mask), mask=mask)


def main():
    if not torch.cuda.is_available():
        print("triton_smoke: CUDA is not visible", file=sys.stderr)
        return 1
    n = 4096
    x = torch.arange(n, device="cuda", dtype=torch.float32)
    y = torch.ones(n, device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)
    add_kernel[(triton.cdiv(n, 1024),)](x, y, out, n, BLOCK=1024)
    torch.cuda.synchronize()
    if not torch.equal(out, x + y):
        print("triton_smoke: the kernel ran but returned wrong values", file=sys.stderr)
        return 1
    print(f"triton_smoke: triton {triton.__version__} compiled and ran a kernel on {torch.cuda.get_device_name(0)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
