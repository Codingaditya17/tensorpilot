# TensorPilot v0.6 — GPU runtime intelligence

TensorPilot v0.6 adds synchronized GPU phase measurement, adaptive batch width,
bounded workload-matrix planning, performance gates and Apple Metal capture. The
runtime remains a single-node MLX service. All results below were measured on the
Apple M5 host with the pinned Qwen2.5 0.5B 4-bit artifact recorded in each report.

## Measurement contract

- Model loading, warmup and execution preparation use monotonic model-thread wall
  time. They are separate in `/health`.
- Real batch steps call `mlx.core.synchronize()` immediately before and after
  `BatchGenerator.next()`. Empty response steps are labeled prefill; nonempty
  response steps are labeled decode. These are synchronized phase measurements,
  not individual kernel timings.
- Runtime counters count a shared step once. Load reports subtract before/after
  counters so shared GPU work is not multiplied by the number of sequences.
- Per-request phase profiles describe the shared steps observed while that request
  was active. They must not be summed across requests.
- TTFT includes queueing. Inter-token latency is derived from non-EOS output-token
  timestamps. The load client also records end-to-end timing.
- `estimated_kv_bytes` comes from conservative admission coefficients. MLX active
  and peak allocator values are direct allocator observations. Neither equals
  process RSS or total unified-memory pressure.
- `tensorpilot-kernel-capture` creates a real `.gputrace` after an excluded warmup.
  Capture overhead is extreme, so trace timings are diagnostic only.

## Adaptive scheduler

`--execution adaptive --batch-size N` enables a deterministic controller. At
each admission boundary it observes waiting jobs, active sequences, oldest queue
wait and estimated admission-memory pressure.

1. At or above the hard estimated-memory threshold, target width is one.
2. At or above the soft threshold, target width is at most two.
3. If a queued request exceeds the latency target, target width opens to the
   configured maximum.
4. Otherwise width follows current demand, bounded to `1..N`.

Decisions never preempt active sequences. FIFO queue order, deadline checks,
per-sequence cancellation and reservation cleanup remain in force. `/health` and
Prometheus expose the current target, reason and bounded decision labels.

The heuristic reacts to observable pressure; it does not predict GPU time. A high
width can improve aggregate throughput while worsening inter-token latency. The
memory thresholds operate on conservative reservations, not live allocator data.

## Real results

Three counterbalanced 30-request repetitions compared serial and fixed width four.
All 180 requests succeeded. Median fixed-width throughput was about 137.3% higher,
P95 client latency about 39.4% lower and P95 TTFT about 95.8% lower. P95
inter-token latency was about 375.1% higher, which is the material cost of sharing
decode work at this width. Evidence: `results/v06_serial_vs_fixed4/`.

Three matched repetitions compared fixed width four with adaptive maximum width
four. All 180 requests succeeded. Adaptive throughput differed by +0.13% and P95
latency by -0.02%. This workload kept enough demand to select width four, so it
does not support a performance-improvement claim. It verifies parity under steady
load. Evidence: `results/v06_fixed4_vs_adaptive4/`.

A changing-traffic run selected observed decode width one at concurrency one and
width eight at concurrency eight. Both windows completed without failures or
rejections. The high-load window reached 890.3 output tokens/s with measured mean
step occupancy 7.77; the low-load window reached 267.8 tokens/s with occupancy
1.0. These are different traffic windows, not a controlled speedup comparison.

The bounded context/output matrix covered target input lengths 128/512/2048 and
output caps 128/256/512. Actual tokenizer counts were 120/504/2040. All nine
requests completed their cap, producing 2,688 tokens with no failure/rejection.
The measured window contained 2 prefill and 1,574 decode steps, 0.065 seconds of
synchronized prefill work, 6.954 seconds of synchronized decode work and mean
occupancy 1.73. Maximum observed MLX active allocation was 339,040,188 bytes;
process-lifetime allocator peak reached 614,778,088 bytes.

One real Metal capture contains a warmup-excluded 16-token generation and pinned
model metadata. The `.gputrace` is about 616 MiB and remains under local ignored
results rather than release archives. Open it with Xcode Instruments to inspect
kernel names, command-buffer timing and GPU occupancy. Its captured request took
27.2 seconds because capture overhead dominates normal execution.

## Reproduction

```bash
tensorpilot-server --execution adaptive --batch-size 4 --offline
tensorpilot-matrix --safe-only --output results/v06_matrix.json
tensorpilot-experiment --comparison execution \
  --execution-baseline batch --execution-candidate adaptive \
  --batch-size 4 --requests 30 --concurrency 4 --repeat 3 --offline
MTL_CAPTURE_ENABLED=1 tensorpilot-kernel-capture --offline \
  --requests 1 --max-tokens 16 --output results/v06_kernel_profile.gputrace
```

Run matrix cases only after their conservative preflight passes. Start with low
concurrency, inspect actual prompt-token counts and allocator samples, then scale.
The generated matrix contains both fixed and adaptive cases at widths 1/2/4/8 and
arrival rates zero (closed loop) and four requests per second.

## Limits

- Results are local Apple Metal measurements for one small quantized model. They
  do not predict AMD ROCm, NVIDIA CUDA or multi-GPU behavior.
- Prefill/decode classification follows `BatchGenerator` response boundaries.
  Metal capture is required for kernel-level attribution.
- The memory preflight is deliberately conservative and is not a capacity model.
- The 2048-token target stays just below the service context cap after tokenization.
- Three small repetitions show repeatability for this workload, not statistical
  significance across models, prompts or thermal states.
- Adaptive scheduling has no learned cost model and cannot preempt an active step.
