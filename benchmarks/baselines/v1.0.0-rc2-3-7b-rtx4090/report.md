# TensorPilot v1.0 GPU evaluation report

Generated from the stage evidence listed below. Nothing here is estimated: a figure appears only if the evidence contains it, and a comparison appears only if both sides were run. **†** marks a point built from fewer than 100 completed requests; its p95 and p99 are effectively the maximum. Hollow chart markers mean the same.

## Acceptance: evaluation FAIL, performance targets INCONCLUSIVE

PASS only if every gate criterion passes; any gate FAIL is FAIL; otherwise missing evidence is INCONCLUSIVE. Performance targets are reported separately and never change the evaluation verdict.

| # | criterion | group | verdict | evidence |
|---|---|---|---|---|
| 1 | 7B BF16 loads and generates on a real GPU | correctness | PASS | preflight ready=True, smoke 5 completed, 19381 benchmark requests completed, CUDA device seen=True |
| 1 | 14B NF4 loads and generates on a real GPU | correctness | INCONCLUSIVE | stage 14b-nf4-prod was not found in the evidence |
| 1 | At least 99.9% protocol-valid responses | correctness | PASS | 20037 of 20037 responses were protocol-valid (1.00000, need 0.999); cancelled, rejected and timed-out requests are not responses to validate |
| 1 | Zero silent truncation, duplicate completions or unaccounted requests | correctness | PASS | 20568 requests across 46 runs: 0 truncated, 0 duplicated, 0 unaccounted |
| 1 | 7B BF16 batched output equals solo output, or every divergence is explained | correctness | PASS | exact-match fractions {'together_vs_solo': 0.625, 'staggered_vs_solo': 0.625}; every diverged request [0, 6, 7, 11, 12, 13, 14, 18, 20, 23, 28, 31] was reproduced by the probe (packing 512) and is a rounding-level near-tie (documented numerical divergence) |
| 1 | The quick session's 31/32 divergence is explained with logits | correctness | INCONCLUSIVE | no invariance.json in 0.5b-quick-batch: the batch-invariance check did not run or did not complete |
| 1 | NF4 7B fidelity against the bf16 reference meets the declared threshold | correctness | INCONCLUSIVE | quality-7b comparison.json was not found in the evidence |
| 1 | NF4 14B is at least as good as the 7B bf16 reference on the same text | correctness | INCONCLUSIVE | quality-14b/nf4-absolute.json and quality-7b/reference-bf16.json was not found in the evidence |
| 2 | Throughput and latency reported at every concurrency level (7b) | completeness | PASS | all seven levels were measured and reported (rejections and timeouts included, not required to be absent) |
| 2 | Throughput and latency reported at every concurrency level (14b) | completeness | INCONCLUSIVE | stage 14b-nf4-prod was not found in the evidence |
| 2 | At least 10,000 requests executed overall | completeness | PASS | 20568 requests executed across all stages (need 10000; the 100000-request stretch target is not met) |
| 2 | Maximum sustainable arrival rate found by driving to saturation (7b) | completeness | PASS | 12 req/s is the highest rate that, with every lower rate, met the objective; a higher rate did not |
| 2 | Maximum sustainable arrival rate found by driving to saturation (14b) | completeness | INCONCLUSIVE | 14b-nf4-prod ladder-poisson was not found in the evidence |
| 2 | Continuous batching is at least 1.5x the serial baseline on comparable workloads | performance | PASS | cmp-c8: 5.98x, cmp-c32: 15.50x (need 1.5x; median of repeats, wall-clock tokens/s) |
| 3 | At sustainable load at least 99.9% of requests succeed | reliability | PASS | 3000 of 3000 succeeded (1.00000, need 0.999); intentionally cancelled requests and documented overload rejections are excluded, timeouts and failures are not |
| 3 | Zero CUDA out-of-memory events | reliability | PASS | 3095 telemetry samples: max cuda_oom_events 0; OOM text in engine logs: none |
| 3 | At overload the queue stays bounded, rejections are controlled and nothing leaks | reliability | FAIL | nothing was rejected: the overload never exceeded the queue, so the bound was not exercised; uncontrolled failures: {'incomplete_stream': 318} |
| 3 | A worker crash recovers within 60 seconds with request accounting intact | reliability | PASS | recovered in 16.3 s (limit 60 s); failed accounting checks: none |
| 4 | Workload-specific SLOs were declared before execution and used unchanged | reproducibility | PASS | 42 runs carry exactly the SLO committed in criteria.json |
| 4 | P95 TTFT <= 2 s and P95 inter-token <= 150 ms at the accepted-load operating point (7b) | performance | PASS | at 12 req/s: P95 TTFT 0.428 s (max 2.0), P95 inter-token 58.7 ms (max 150); P50/P95/P99 e2e 2.304627623001579/8.08743955940008/11.006301456529158 s |
| 4 | P95 TTFT <= 2 s and P95 inter-token <= 150 ms at the accepted-load operating point (14b) | performance | INCONCLUSIVE | 14b-nf4-prod ladder-poisson was not found in the evidence |
| 4 | Timeouts, rejections and failures are counted in every rate and SLO attainment | reproducibility | PASS | 46 runs checked |
| 5 | vLLM and TensorPilot ran on the same physical GPU, model revision and precision | completeness | INCONCLUSIVE | vllm-7b-bf16-prod has no benchmark manifest (it produced no measurements), so nothing was compared |
| 5 | Three measured repetitions per key configuration, on identical workloads | completeness | INCONCLUSIVE | cmp-c8: 3 TensorPilot / 0 vLLM repeats; cmp-c32: 3 TensorPilot / 0 vLLM repeats; cmp-c128: 3 TensorPilot / 0 vLLM repeats |
| 5 | Throughput, goodput, latency, VRAM and error rates are reported for both engines | completeness | PASS | throughput, goodput, latency percentiles, VRAM and error rates are present for both |
| 5 | Performance gaps to vLLM are explained with evidence | completeness | INCONCLUSIVE | no written report was supplied (--report) |
| 6 | Continuous GPU telemetry covers every measured experiment | reproducibility | PASS | 22 experiments have continuous samples |
| 6 | Configurations, model revisions and the image are pinned and reproducible | reproducibility | PASS | 2 stages ran the committed config, a pinned revision and a declared image |
| 6 | Evidence archives are SHA-256 verified and secret-free | reproducibility | INCONCLUSIVE | no .tar.gz archives were supplied (--archives) |
| 7 | Every stage stopped within its hard cap | budget | INCONCLUSIVE | no ledger was supplied (--ledger) |
| 7 | Total spend stayed within the approved budget | budget | INCONCLUSIVE | pass --spent-usd from the Runpod billing page. The ledger shows 0 stage minutes, but boot, downloads, loads and idle time are not in it |

## Provenance

| stage | model | revision | engine | quantization | dtype | GPU | driver | exit | minutes | mode |
|---|---|---|---|---|---|---|---|---|---|---|
| 7b-bf16-prod | Qwen/Qwen2.5-7B-Instruct | a09a35458c | tensorpilot | none | bfloat16 | NVIDIA GeForce RTX 4090 | 595.91.07 | n/a | n/a | batch |
| 7b-bf16-prod-serial | Qwen/Qwen2.5-7B-Instruct | a09a35458c | tensorpilot | none | bfloat16 | NVIDIA GeForce RTX 4090 | 595.91.07 | n/a | n/a | serial |
| faults-7b | n/a | n/a | n/a | n/a | n/a | not recorded | not recorded | n/a | n/a | n/a |
| probe-7b | n/a | n/a | n/a | n/a | n/a | not recorded | not recorded | n/a | n/a | n/a |
| vllm-7b-bf16-prod | n/a | n/a | n/a | n/a | n/a | not recorded | not recorded | n/a | n/a | n/a |

## Experiment status

Every planned experiment is listed. Failed, aborted, skipped and never-run experiments are results too.

### 7b-bf16-prod

| experiment | status | detail |
|---|---|---|
| cold-start | completed | 5.6 s of 300 s cap |
| cmp-c1 | completed | 59.2 s of 300 s cap |
| cmp-c8 | completed | 27.1 s of 300 s cap |
| cmp-c32 | completed | 21.2 s of 300 s cap |
| cmp-c128 | completed | 119.2 s of 360 s cap |
| scale-c1 | completed | 137.4 s of 480 s cap |
| scale-c32 | completed | 164.5 s of 540 s cap |
| scale-c64 | completed | 135.9 s of 420 s cap |
| scale-c128 | completed | 135.8 s of 420 s cap |
| scale-c256 | completed | 177.8 s of 480 s cap |
| scale-c512 | completed | 177.8 s of 480 s cap |
| scale-c1024 | completed | 178.6 s of 480 s cap |
| ladder-poisson | completed | 731.1 s of 1200 s cap |
| sustained-c32 | completed | 318.0 s of 840 s cap |
| cancellation | completed | 33.1 s of 300 s cap |
| timeouts | completed | 23.0 s of 300 s cap |
| overload | completed | 50.0 s of 300 s cap |
| recovery-after-overload | completed | 8.4 s of 300 s cap |

### 7b-bf16-prod-serial

| experiment | status | detail |
|---|---|---|
| cold-start | completed | 5.6 s of 300 s cap |
| cmp-c1 | completed | 58.6 s of 300 s cap |
| cmp-c8 | completed | 150.4 s of 480 s cap |
| cmp-c32 | completed | 288.7 s of 780 s cap |

### vllm-7b-bf16-prod

| experiment | status | detail |
|---|---|---|
| cold-start | NOT RUN | no record in the manifest |
| cmp-c1 | NOT RUN | no record in the manifest |
| cmp-c8 | NOT RUN | no record in the manifest |
| cmp-c32 | NOT RUN | no record in the manifest |
| cmp-c128 | NOT RUN | no record in the manifest |
| scale-c1 | NOT RUN | no record in the manifest |
| scale-c32 | NOT RUN | no record in the manifest |
| scale-c64 | NOT RUN | no record in the manifest |
| scale-c128 | NOT RUN | no record in the manifest |
| scale-c256 | NOT RUN | no record in the manifest |
| scale-c512 | NOT RUN | no record in the manifest |
| scale-c1024 | NOT RUN | no record in the manifest |
| ladder-poisson | NOT RUN | no record in the manifest |
| sustained-c32 | NOT RUN | no record in the manifest |

## Concurrency sweep: 7b-bf16-prod

Closed loop, mixed prompt and output lengths; values are the median over repeats with (min–max).

| clients | repeats | completed | output tok/s (wall) | tok/s (active window) | TTFT p50 s | TTFT p95 s | TTFT p99 s | e2e p50 s | e2e p95 s | e2e p99 s | TPOT p95 s | error rate | rejected | timed out | goodput req/s | SLO attainment | GPU util % (mean) | GPU mem max MiB | power W | mean batch width | decode tok/s (GPU-active) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2 | 80† | 52.1 (52.1–52.1) | 52.1 (52.1–52.1) | 0.023 (0.023–0.023) | 0.088 (0.088–0.089) | 0.109 (0.109–0.109) | 0.96 (0.96–0.96) | 3.88 (3.88–3.88) | 5.34 (5.33–5.34) | 0.0194 (0.0194–0.0194) | 0.000 (0.000–0.000) | 0 | 0 | 0.61 (0.61–0.61) | 1.000 (1.000–1.000) | 95 (95–95) | 19436 (19436–19436) | 292 (291–292) | 1.0 (1.0–1.0) | 53.6 (53.6–53.6) |
| 32 | 1 | 1500 | 822.9 | 823.3 | 0.075 | 0.329 | 0.976 | 2.13 | 8.22 | 11.41 | 0.0421 | 0.000 | 0 | 0 | 9.58 | 1.000 | 91 | 19436 | 327 | 30.4 | 1285.4 |
| 64 | 1 | 1500 | 1006.4 | 1007.2 | 1.430 | 1.914 | 2.277 | 3.90 | 11.23 | 15.19 | 0.0511 | 0.000 | 0 | 0 | 11.38 | 0.971 | 91 | 19452 | 338 | 45.5 | 1763.5 |
| 128 | 1 | 1500 | 1005.5 | 1006.5 | 6.748 | 7.740 | 8.002 | 9.28 | 16.63 | 20.39 | 0.0511 | 0.000 | 0 | 0 | 0.26 | 0.022 | 91 | 19436 | 337 | 45.6 | 1763.7 |
| 256 | 1 | 2000 | 1012.4 | 1013.4 | 17.272 | 19.041 | 19.277 | 19.92 | 27.27 | 31.23 | 0.0512 | 0.000 | 0 | 0 | 0.18 | 0.015 | 91 | 19436 | 337 | 46.0 | 1787.3 |
| 512 | 1 | 2000 | 1013.6 | 1014.9 | 37.800 | 40.296 | 40.853 | 40.36 | 47.90 | 51.59 | 0.0511 | 0.000 | 0 | 0 | 0.18 | 0.015 | 91 | 19436 | 337 | 46.2 | 1788.1 |
| 1024 | 1 | 2000 | 1010.2 | 1013.1 | 78.492 | 82.715 | 83.028 | 80.50 | 89.31 | 93.31 | 0.0513 | 0.000 | 0 | 0 | 0.15 | 0.013 | 90 | 19436 | 336 | 46.0 | 1783.2 |

![Output tokens/s (wall clock), 7b-bf16-prod](charts/throughput-7b-bf16-prod.svg)

![TTFT p95 (s), 7b-bf16-prod](charts/ttft-7b-bf16-prod.svg)

![End-to-end latency p95 (s), 7b-bf16-prod](charts/latency-7b-bf16-prod.svg)

![Goodput under the declared SLO (requests/s), 7b-bf16-prod](charts/goodput-7b-bf16-prod.svg)

## Engine profile: 7b-bf16-prod

From the worker's synchronized step counters and the GPU samples, per experiment. A decode-step time that stays flat as tokens per step grow, with low GPU utilization, means the step is bound by launch and host overhead, not by the GPU; prefill share is the fraction of engine time spent in prompt processing.

| experiment | tokens/step | ms/decode step | ms/prefill chunk | engine busy | prefill share | GPU util % | power W | max queue | max active |
|---|---|---|---|---|---|---|---|---|---|
| cancellation | 22.01 | 21.9 | 27.3 | 0.95 | 0.34 | 90 | 314 | 3 | 32 |
| cmp-c1 | 1.00 | 18.3 | 20.9 | 0.97 | 0.03 | 95 | 291 | 0 | 1 |
| cmp-c128 | 33.04 | 23.4 | 27.5 | 0.93 | 0.35 | 88 | 324 | 80 | 48 |
| cmp-c32 | 22.12 | 20.5 | 23.7 | 0.91 | 0.27 | 91 | 291 | 2 | 32 |
| cmp-c8 | 6.76 | 19.3 | 21.7 | 0.96 | 0.13 | 92 | 274 | 1 | 8 |
| cold-start | 1.00 | 18.2 | 21.1 | 0.97 | 0.03 | 94 | 276 | 0 | 1 |
| ladder-poisson | 7.36 | 20.2 | 26.1 | 0.90 | 0.13 | 86 | 269 | 115 | 48 |
| overload | 40.65 | 21.3 | 23.5 | 0.91 | 0.40 | 86 | 310 | 976 | 48 |
| recovery-after-overload | 7.32 | 19.4 | 21.7 | 0.96 | 0.14 | 76 | 247 | 1 | 8 |
| scale-c1 | 1.00 | 18.6 | 24.5 | 0.98 | 0.02 | 94 | 291 | 0 | 1 |
| scale-c1024 | 40.79 | 24.3 | 27.2 | 0.93 | 0.39 | 90 | 333 | 976 | 48 |
| scale-c128 | 39.14 | 24.0 | 27.3 | 0.94 | 0.38 | 90 | 333 | 80 | 48 |
| scale-c256 | 40.79 | 24.2 | 27.2 | 0.93 | 0.39 | 90 | 334 | 208 | 48 |
| scale-c32 | 27.67 | 22.8 | 26.3 | 0.95 | 0.32 | 92 | 325 | 3 | 32 |
| scale-c512 | 40.79 | 24.2 | 27.1 | 0.93 | 0.39 | 91 | 334 | 464 | 48 |
| scale-c64 | 39.14 | 24.0 | 27.3 | 0.94 | 0.38 | 90 | 334 | 16 | 48 |
| sustained-c32 | 29.56 | 23.0 | 26.2 | 0.95 | 0.33 | 91 | 326 | 3 | 32 |
| timeouts | 15.61 | 21.0 | 27.7 | 0.96 | 0.34 | 90 | 319 | 0 | 32 |

## Open-loop rate ladder: 7b-bf16-prod

Objective: SLO attainment ≥ 95% and error rate ≤ 1%. **Maximum sustainable rate under it: 12 requests/s.**

| offered req/s | completed | output tok/s | TTFT p95 s | e2e p95 s | error rate | SLO attainment | goodput req/s |
|---|---|---|---|---|---|---|---|
| 1 | 240 | 83.7 | 0.181 | 4.88 | 0.000 | 1.000 | 0.95 |
| 2 | 240 | 163.5 | 0.186 | 5.08 | 0.000 | 1.000 | 1.86 |
| 3 | 240 | 239.0 | 0.192 | 5.74 | 0.000 | 1.000 | 2.72 |
| 4 | 240 | 310.6 | 0.197 | 5.93 | 0.000 | 1.000 | 3.54 |
| 6 | 240 | 443.9 | 0.223 | 6.09 | 0.000 | 1.000 | 5.06 |
| 8 | 240 | 562.9 | 0.284 | 6.66 | 0.000 | 1.000 | 6.42 |
| 12 | 240 | 765.9 | 0.428 | 8.09 | 0.000 | 1.000 | 8.73 |
| 16 | 240 | 851.8 | 2.411 | 9.65 | 0.000 | 0.812 | 7.89 |
| 24 | 240 | 857.6 | 7.124 | 13.59 | 0.000 | 0.392 | 3.83 |
| 32 | 240 | 856.3 | 9.601 | 15.44 | 0.000 | 0.312 | 3.05 |

![Goodput against offered rate, 7b-bf16-prod](charts/ladder-7b-bf16-prod.svg)

## Serial against continuous batching: 7b-bf16-prod-serial vs 7b-bf16-prod

The same model on the same card; only the execution mode differs.

| pair | serial tok/s | batch tok/s | ratio | serial e2e p95 s | batch e2e p95 s |
|---|---|---|---|---|---|
| cmp-c1 vs cmp-c1 | 53.9 (53.7–54.3) | 53.4 (53.3–53.5) | 0.99× | 1.15 (1.14–1.15) | 1.17 (1.16–1.17)† |
| cmp-c8 vs cmp-c8 | 53.8 (53.2–54.0) | 321.7 (321.6–321.8) | 5.98× | 7.01 (6.95–7.03) | 1.46 (1.46–1.46)† |
| cmp-c32 vs cmp-c32 | 53.7 (53.7–53.8) | 832.7 (827.4–833.9) | 15.50× | 25.74 (25.70–25.79) | 2.17 (2.17–2.18) |

## TensorPilot against vLLM: 7b-bf16-prod vs vllm-7b-bf16-prod

Identical workloads (asserted by the suite's tests), same card, never run at the same time. These are different engines on different CUDA toolkits (torch 2.9.1 with CUDA 12.8 against torch 2.13.0 with CUDA 13.0), so differences are not attributable to one cause. The ratio is vLLM ÷ TensorPilot.

| clients | TP tok/s | vLLM tok/s | vLLM÷TP | TP TTFT p95 | vLLM TTFT p95 | TP e2e p95 | vLLM e2e p95 | TP goodput | vLLM goodput |
|---|---|---|---|---|---|---|---|---|---|

![Output tokens/s, 7b-bf16-prod vs vLLM](charts/engines-throughput-7b-bf16-prod.svg)

![End-to-end latency p95 (s), 7b-bf16-prod vs vLLM](charts/engines-latency-7b-bf16-prod.svg)

## Cancellation, timeouts and overload

Client outcomes beside the server's own counters, so a request that one side counts and the other does not is visible. Rejections are expected under overload; an OOM event is never expected.

| stage | experiment | requests | client outcomes | error rate | rejected | timed out | cancelled by client | server counter deltas | OOM events |
|---|---|---|---|---|---|---|---|---|---|
| 7b-bf16-prod | cancellation | 320 | {"client_cancelled": 90, "completed": 230} | 0.000 | 0 | 0 | 90 | {"admission_decisions_total{decision=accepted}": 320.0, "jobs_total{outcome=completed}": 230.0, "jobs_total{outcome=cancelled}": 90.0, "output_tokens_total": 21032.0} | 0 |
| 7b-bf16-prod | timeouts | 240 | {"completed": 117, "timed_out": 123} | 0.512 | 0 | 123 | 0 | {"admission_decisions_total{decision=accepted}": 240.0, "jobs_total{outcome=completed}": 117.0, "jobs_total{outcome=timed_out}": 123.0, "output_tokens_total": 10665.0} | 0 |
| 7b-bf16-prod | overload | 1500 | {"completed": 1182, "timed_out": 318} | 0.212 | 0 | 318 | 0 | {"admission_decisions_total{decision=accepted}": 1500.0, "jobs_total{outcome=completed}": 1182.0, "jobs_total{outcome=timed_out}": 318.0, "output_tokens_total": 52588.0} | 0 |
| 7b-bf16-prod | recovery-after-overload | 64 | {"completed": 64} | 0.000 | 0 | 0 | 0 | {"admission_decisions_total{decision=accepted}": 64.0, "jobs_total{outcome=completed}": 64.0, "output_tokens_total": 2647.0} | 0 |

## Worker crash and recovery: faults-7b

SIGKILL under load, supervised restart, then an audit of the request journal. **Overall: PASSED.**

| check | result | detail |
|---|---|---|
| worker_became_ready | pass | the supervised worker reported ready |
| worker_was_killed | pass | SIGKILL sent to pid 14347 |
| supervisor_restarted_the_worker_with_a_new_process | pass | restart took 16.29 s from the kill to /readyz |
| one_record_per_request | pass | 900 records for 900 planned requests |
| every_request_has_exactly_one_terminal_outcome | pass | every record carries one of completed, rejected, timed_out, failed, client_cancelled |
| completed_requests_received_exactly_the_tokens_generated | pass | client and server token counts agree on every completed request |
| no_request_completed_across_the_crash | pass | no completed request was in flight when the worker was killed |
| in_flight_requests_ended_in_a_terminal_failure | pass | 5 request(s) were in flight at the kill; 5 ended in a terminal failure |
| requests_during_the_outage_failed_fast_and_none_completed | pass | 81 request(s) were dispatched during the outage; none completed |
| requests_after_recovery_complete | pass | 691 request(s) were dispatched after recovery; all completed |
| fresh_requests_succeed_after_recovery | pass | 5 of 5 succeeded |
| authentication_survives_the_restart | pass | no_token=401, wrong_token=401, valid_token=200 |
| no_worker_process_left_running | pass | every worker process the supervisor started is gone |

Readiness timeline (seconds on the load run's clock): 0.0 s ready, 25.1 s unreachable, 41.3 s ready

## Batch invariance: 7b-bf16-prod

Does a request's output depend on what else is in the batch? Each request is compared with its solo run. Divergence late in a sequence is a numeric near-tie; divergence at the start is a defect.

| comparison | identical | mean prefix | earliest divergence | median divergence |
|---|---|---|---|---|
| together vs solo | 20/32 | 0.730 | 1 | 9 |
| staggered vs solo | 20/32 | 0.730 | 1 | 9 |

## Limitations recorded by the runs

- The worker batches sequences in one forward pass over a fixed-slot KV pool (not paged); each active sequence reserves a full row, so concurrency is bounded by slots rather than by tokens in use.
- Client timings include loopback HTTP, a fresh connection per request and streaming parse time on the load-generating host.
- One model on one GPU; no multi-GPU or scaling claim.
- CUDA peak memory is process-lifetime from the PyTorch caching allocator, not total device pressure.
- Planning estimates are assumptions and are never reported as measurements.
- GPU clocks, temperature and co-tenancy are recorded by nvidia-smi samples, not controlled.
- The worker executes one sequence at a time; concurrency above one measures queueing, not batched throughput.
- Client timings include loopback HTTP and streaming parse time on the load-generating host.

## What this report does not establish

- Anything not listed under Experiment status as *completed* was not measured.
- SLO thresholds are declared in the suite, not derived from user requirements.
- Hypotheses about causes (for example NF4 dequantization overhead, or the Python event loop at high batch width) are not measurements; the telemetry and counters in this report are what would test them.
