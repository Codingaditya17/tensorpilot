# Mixed-workload scheduling study — 2026-10-17

Measured on the local **Apple M5, 16 GiB unified memory**, using real MLX inference
through TensorPilot's HTTP service. These are not CUDA/ROCm or multi-node results.

## Method

- Qwen2.5-0.5B-Instruct, 4-bit; greedy sampling and raw-text prompts.
- Thirty requests per run: 18 short (16-token budget), 6 medium (64), 6 long (128).
- Seed 42; six concurrent clients; eight waiting slots; 6 GiB MLX allocator limit.
- Three repetitions per policy/threshold; FIFO/shortest-budget order alternates.
- Each run loads/warms a fresh owned model process, then warms one request per class.
- Three aging thresholds, 18 measured runs, **540 successful requests, zero failures
  or rejections**. Warmup requests are separate and excluded from these counts.
- Workload is closed-loop and concurrency-capped. The same ordered request plan
  does not imply identical actual arrival times across policies.

## Results

Each value is the relative change in the median of three per-run metrics for
shortest-budget versus that threshold's matched FIFO baseline. Negative latency
is lower; positive throughput is higher.

| Aging threshold | Overall median latency | Overall P95 | Short P95 | Long P95 | Output throughput |
|---|---:|---:|---:|---:|---:|
| 0.25 s | -0.9% | -1.0% | -1.2% | -1.1% | +1.3% |
| 1.0 s | -28.1% | +12.2% | +11.0% | +28.1% | -0.2% |
| 2.0 s | -49.2% | +86.5% | -28.1% | +106.0% | -0.0% |

The 0.25 s threshold behaves close to FIFO under this workload. At 1 s, smaller
requests improve median latency while overall and long-request tail latency
worsen. At 2 s, short-request latency improves more, but long-request tail latency
roughly doubles and overall P95 worsens substantially. Aggregate throughput stays
similar because scheduling reorders work on a single execution worker.

**Decision: retain FIFO as the default.** The 1 s candidate failed an explicit
5% overall P95 regression gate; the CLI returned status 1 as designed. Choosing a
policy requires an explicit class/SLO tradeoff, not only a lower global median.
No statistical significance or generalization to other workloads is established.
The corpus has only three prompt templates, and the six long requests per run
make per-class tail estimates especially uncertain.

## Provenance and evidence

- Model revision: `a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3`.
- Artifact SHA-256: `909346d914889eadd92f3073fd789c19bf963886a07f88946551e975a79153a5`.
- Source SHA-256: `75386351351ffc718f6a7c7231b92e003a65987955f95874c7cb9f51bbe6778e`.
- MLX 0.32.4; mlx-lm 0.32.2; Python 3.14.6.

Each experiment directory includes an immutable source ZIP, installed-version
snapshot, exact workload JSON, manifest, raw request timing/token records, warmup
reports, per-server logs and comparison JSON. Raw files stay under ignored
`results/`; `scheduler_summary.json` preserves the comparison aggregates here.
The source archives include the README required for package installation.
