# TensorPilot production-scale acceptance

**Evaluation: FAIL**  ·  **Performance targets: INCONCLUSIVE**

PASS only if every gate criterion passes; any gate FAIL is FAIL; otherwise missing evidence is INCONCLUSIVE. Performance targets are reported separately and never change the evaluation verdict.

| # | Criterion | Group | Verdict | Evidence |
|---|---|---|---|---|
| 1 | 7B BF16 loads and generates on a real GPU | correctness | **PASS** | preflight ready=True, smoke 5 completed, 19381 benchmark requests completed, CUDA device seen=True |
| 1 | 14B NF4 loads and generates on a real GPU | correctness | **INCONCLUSIVE** | stage 14b-nf4-prod was not found in the evidence |
| 1 | At least 99.9% protocol-valid responses | correctness | **PASS** | 20037 of 20037 responses were protocol-valid (1.00000, need 0.999); cancelled, rejected and timed-out requests are not responses to validate |
| 1 | Zero silent truncation, duplicate completions or unaccounted requests | correctness | **PASS** | 20568 requests across 46 runs: 0 truncated, 0 duplicated, 0 unaccounted |
| 1 | 7B BF16 batched output equals solo output, or every divergence is explained | correctness | **PASS** | exact-match fractions {'together_vs_solo': 0.625, 'staggered_vs_solo': 0.625}; every diverged request [0, 6, 7, 11, 12, 13, 14, 18, 20, 23, 28, 31] was reproduced by the probe (packing 512) and is a rounding-level near-tie (documented numerical divergence) |
| 1 | The quick session's 31/32 divergence is explained with logits | correctness | **INCONCLUSIVE** | no invariance.json in 0.5b-quick-batch: the batch-invariance check did not run or did not complete |
| 1 | NF4 7B fidelity against the bf16 reference meets the declared threshold | correctness | **INCONCLUSIVE** | quality-7b comparison.json was not found in the evidence |
| 1 | NF4 14B is at least as good as the 7B bf16 reference on the same text | correctness | **INCONCLUSIVE** | quality-14b/nf4-absolute.json and quality-7b/reference-bf16.json was not found in the evidence |
| 2 | Throughput and latency reported at every concurrency level (7b) | completeness | **PASS** | all seven levels were measured and reported (rejections and timeouts included, not required to be absent) |
| 2 | Throughput and latency reported at every concurrency level (14b) | completeness | **INCONCLUSIVE** | stage 14b-nf4-prod was not found in the evidence |
| 2 | At least 10,000 requests executed overall | completeness | **PASS** | 20568 requests executed across all stages (need 10000; the 100000-request stretch target is not met) |
| 2 | Maximum sustainable arrival rate found by driving to saturation (7b) | completeness | **PASS** | 12 req/s is the highest rate that, with every lower rate, met the objective; a higher rate did not |
| 2 | Maximum sustainable arrival rate found by driving to saturation (14b) | completeness | **INCONCLUSIVE** | 14b-nf4-prod ladder-poisson was not found in the evidence |
| 2 | Continuous batching is at least 1.5x the serial baseline on comparable workloads | performance | **PASS** | cmp-c8: 5.98x, cmp-c32: 15.50x (need 1.5x; median of repeats, wall-clock tokens/s) |
| 3 | At sustainable load at least 99.9% of requests succeed | reliability | **PASS** | 3000 of 3000 succeeded (1.00000, need 0.999); intentionally cancelled requests and documented overload rejections are excluded, timeouts and failures are not |
| 3 | Zero CUDA out-of-memory events | reliability | **PASS** | 3095 telemetry samples: max cuda_oom_events 0; OOM text in engine logs: none |
| 3 | At overload the queue stays bounded, rejections are controlled and nothing leaks | reliability | **FAIL** | nothing was rejected: the overload never exceeded the queue, so the bound was not exercised; uncontrolled failures: {'incomplete_stream': 318} |
| 3 | A worker crash recovers within 60 seconds with request accounting intact | reliability | **PASS** | recovered in 16.3 s (limit 60 s); failed accounting checks: none |
| 4 | Workload-specific SLOs were declared before execution and used unchanged | reproducibility | **PASS** | 42 runs carry exactly the SLO committed in criteria.json |
| 4 | P95 TTFT <= 2 s and P95 inter-token <= 150 ms at the accepted-load operating point (7b) | performance | **PASS** | at 12 req/s: P95 TTFT 0.428 s (max 2.0), P95 inter-token 58.7 ms (max 150); P50/P95/P99 e2e 2.304627623001579/8.08743955940008/11.006301456529158 s |
| 4 | P95 TTFT <= 2 s and P95 inter-token <= 150 ms at the accepted-load operating point (14b) | performance | **INCONCLUSIVE** | 14b-nf4-prod ladder-poisson was not found in the evidence |
| 4 | Timeouts, rejections and failures are counted in every rate and SLO attainment | reproducibility | **PASS** | 46 runs checked |
| 5 | vLLM and TensorPilot ran on the same physical GPU, model revision and precision | completeness | **INCONCLUSIVE** | vllm-7b-bf16-prod has no benchmark manifest (it produced no measurements), so nothing was compared |
| 5 | Three measured repetitions per key configuration, on identical workloads | completeness | **INCONCLUSIVE** | cmp-c8: 3 TensorPilot / 0 vLLM repeats; cmp-c32: 3 TensorPilot / 0 vLLM repeats; cmp-c128: 3 TensorPilot / 0 vLLM repeats |
| 5 | Throughput, goodput, latency, VRAM and error rates are reported for both engines | completeness | **PASS** | throughput, goodput, latency percentiles, VRAM and error rates are present for both |
| 5 | Performance gaps to vLLM are explained with evidence | completeness | **INCONCLUSIVE** | no written report was supplied (--report) |
| 6 | Continuous GPU telemetry covers every measured experiment | reproducibility | **PASS** | 22 experiments have continuous samples |
| 6 | Configurations, model revisions and the image are pinned and reproducible | reproducibility | **PASS** | 2 stages ran the committed config, a pinned revision and a declared image |
| 6 | Evidence archives are SHA-256 verified and secret-free | reproducibility | **INCONCLUSIVE** | no .tar.gz archives were supplied (--archives) |
| 7 | Every stage stopped within its hard cap | budget | **INCONCLUSIVE** | no ledger was supplied (--ledger) |
| 7 | Total spend stayed within the approved budget | budget | **INCONCLUSIVE** | pass --spent-usd from the Runpod billing page. The ledger shows 0 stage minutes, but boot, downloads, loads and idle time are not in it |
