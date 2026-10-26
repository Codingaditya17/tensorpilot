# Baseline: TensorPilot v0.9.1-4 on one RTX 4090 (serial CUDA worker)

This directory is the **immutable baseline** for the v1.0 evaluation. It is the first successful real-GPU run:
Qwen2.5-0.5B-Instruct at a pinned revision, PyTorch 2.9.1+cu128, one sequence at a time, 20 of 20 benchmark requests
successful, gateway smoke 5 of 5.

- `evidence.tar.gz` is the raw archive exactly as exported from the Pod. `evidence.tar.gz.sha256` is its checksum.
- `summary.json` is generated from that archive by reading its files; no number was typed in by hand.

Do not edit or replace these files. New results are compared against them; they are not merged into them.

What this baseline does **not** show: concurrency above one overlapping, overload, cancellation, crash recovery,
continuous GPU sampling, GPU-side correctness against a reference, any model above 0.5B, or batching. Its
~57-59 output tokens/s is single-stream decode for a 0.5B model with a serial worker, not production-scale performance.
