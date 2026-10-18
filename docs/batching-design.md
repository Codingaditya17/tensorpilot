# Opt-in MLX continuous batching

Serial remains the default. Enable with:

```bash
tensorpilot-server --execution batch --batch-size 2 --offline
```

A single owner thread loads the model and owns BatchGenerator. The engine admits
up to the configured number of sequences (1..8), advances bounded prefill/decode
steps, and continuously fills available slots from the bounded waiting scheduler.
Default prefill step is 256 tokens. Per-sequence IDs, cache membership,
detokenizers, output budgets, cancellation and deadlines remain independent.
Empty decode output during prefill is progress, not completion. EOS is sampled
but excluded from output counts. Cleanup precedes terminal success/failure.

Each response buffer holds at most 16 events. A full batch response buffer fails
that sequence with slow_consumer rather than stopping its peers. Backend removal
failure invalidates the shared context; readiness fails and all jobs terminate.
The watchdog detects a stalled shared operation but cannot interrupt a running
Metal kernel. Cancellation/deadlines take effect at operation boundaries.
Batch memory samples describe the shared process, not one request's allocation.
Per-token cache profiling currently requires serial mode.

## Evidence on Apple M5, MLX 0.32.4 / mlx-lm 0.32.2

- Adapter conformance: greedy serial/batch token IDs matched exactly for two
  prompts at budgets 8/12. Removing another sequence left its peer completing.
  Saved in results/batch_adapter_conformance_v01.json.
- HTTP validation: six normal completions, one cancelled client with a successful
  peer, 10 completions/30 rejections during overload, four recovery completions.
  Health ready, no active sequences afterward; metrics show 21 completions and
  one cancellation. Saved in results/batch_http_validation.json.
- CPU suite: 95 tests passed; Ruff clean. Includes shared failure, removal
  failure, watchdog stall, late admission, shutdown, slow consumer and accounting.

## Matched execution study

Reproduce with no server already using the selected port:

```bash
tensorpilot-experiment --comparison execution --requests 30 --concurrency 6 \
  --repeat 3 --batch-size 2 --offline --output-dir results/my_batch_study
```

The recorded results/batch_execution_seed42 study alternated mode order and used
FIFO for both, seed 42, identical model artifact/source and 60/20/20 short/medium/
long output budgets (16/64/128), eight waiting slots and 6 GiB allocator limit.
Each process warmed the request classes before measurement. Batch warmup used
concurrency two; serial used one. Client load is closed-loop, so actual arrival
instants differ by completion timing. Journals, source archive, warmups, versions,
logs and manifests accompany reports. There were 90 successful requests per mode,
zero failures/rejections. Comparing medians of per-run measures:

| Measure | Serial | Batch width 2 | Change |
| --- | ---: | ---: | ---: |
| Output tokens/s | 288.46 | 450.58 | +56.2% |
| P95 client latency | 1.4198 s | 0.8852 s | -37.7% |
| Short-class P95 | | | -45.8% |
| Long-class P95 | | | -37.1% |

Maximum process allocator peak observed across measured responses was 341,534,144
bytes serial and 348,010,500 bytes batch. These are cumulative MLX allocator
peaks, not per-request memory or physical system/GPU resident usage.

This small single-model study does not establish statistical significance,
production readiness, model-quality equivalence across arbitrary prompts, or
behavior at larger capacity/context. Wider batch limits remain experimental.
Earlier v0.3 artifacts stay unchanged. The experiment archives preserve the
pre-v0.4 version-label source used for measurement.

## Bounded soak

A 30-second run at concurrency six completed 288 requests with no failures or
rejections, 13,872 output tokens at 452.99 tokens/s, including drain time
(30.623 seconds). All 20 readiness snapshots were healthy. Reports, raw journal
and monitor are results/batch_soak*. This is a short stability check, not evidence
of long-term leak freedom. Regression gates also fail increased rejection counts.
