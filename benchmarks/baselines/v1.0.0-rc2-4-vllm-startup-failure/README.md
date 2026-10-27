# Baseline: the second vLLM start-up failure (image rc2-4, one RTX 4090)

The archive is exactly as exported from the Pod (SHA-256 in the `.sha256` file; verified against the hash reported by the operator).
`engine.log` holds the root cause (line 150: FlashInfer finds no `nvcc`); `toolchain.txt` shows the rc2-4 compiler and Triton checks passing.
Analysis, the offline reproduction and the fix: [`docs/v10-vllm-startup-failures.md`](../../../docs/v10-vllm-startup-failures.md).

Do not edit or replace these files.
