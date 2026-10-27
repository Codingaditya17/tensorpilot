"""Run FlashInfer's top-k/top-p sampling kernel once on the GPU, compiling it if it is not cached.

This is the exact operation vLLM's kernel warm-up executes when it starts, which needs nvcc (see cuda_home.sh). It runs with the
vLLM virtualenv's Python, so it imports nothing from tensorpilot. The first run compiles the sampling module (minutes on a Pod CPU),
and the compiled module is cached for the engine that follows.
"""
import sys
import time

import torch


def main():
    if not torch.cuda.is_available():
        print("flashinfer_smoke: CUDA is not visible", file=sys.stderr)
        return 1
    import flashinfer.sampling as sampling

    started = time.perf_counter()
    logits = torch.randn(4, 4096, device="cuda", dtype=torch.float32)
    tokens = sampling.top_k_top_p_sampling_from_logits(logits, 20, 0.9)
    torch.cuda.synchronize()
    if tokens.shape != (4,) or int(tokens.min()) < 0 or int(tokens.max()) >= 4096:
        print(f"flashinfer_smoke: the kernel returned invalid tokens {tokens.tolist()}", file=sys.stderr)
        return 1
    print(f"flashinfer_smoke: top-k/top-p sampling compiled and ran in {time.perf_counter() - started:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
