# v1.0 audit, implementation plan and risk register

## Audit of the starting point (v0.9.1-4)

What existed and worked: an MLX runtime with continuous batching behind an adapter interface, a bounded-admission engine, a control plane and data plane with
separate credentials, a serial CUDA backend, GPU snapshots, a benchmark runner, a supervisor, and the first successful RTX 4090 run (Qwen2.5-0.5B, serial, 57-59 tok/s,
20 of 20 requests, preserved in `benchmarks/baselines/v0.9.1-4-rtx4090`).

Architectural bottlenecks and gaps found on reading the code:

1. **The CUDA worker was serial-only.** `BatchEngine` was already adapter-driven, so the gap was a CUDA adapter, not a new engine.
2. **Batch width was capped at 8 in four places** (CLI, engine, adaptive controller, MLX adapter), and the CLI refused `--execution batch` for CUDA.
3. **Admission used a flat 16 KiB per token against a 512 MiB budget.** The real figure is 12 KiB (0.5B), 56 KiB (7B) and 192 KiB (14B); the budget had no relation to VRAM.
4. **Telemetry was single snapshots.** There was no time series, no prefill/decode split on CUDA and no batch-width or KV-pool state.
5. **No explicit outcome per request.** Cancellations by design were indistinguishable from errors, and there was no goodput.
6. **The load generator had no cancellation, no arrival processes, one fixed-spacing rate and a 3-key workload schema.** The benchmark runner capped concurrency at 64 and requests at 2,000.
7. **Tests exercised the data plane only against an unauthenticated fake worker**, which is how a missing worker credential reached a real GPU Pod.
8. **The CUDA path had never executed**, so errors only CUDA could raise (a bare-device `ValueError`) were invisible to every test.
9. **No explicit model support check.** Anything Transformers could load was assumed servable.
10. **No quantized path**, so a 14B model could only be excluded.

## What was built, against the brief's phases

| Phase | Delivered | Not delivered / why |
|---|---|---|
| 1. Audit, baseline, plan | this document; baseline archive with an enforced checksum | – |
| 2. Models and VRAM | planner from real configs and file sizes; exact revisions and licenses; NF4 for 14B; estimator checked against a real load | measured VRAM waits for a GPU |
| 3. Continuous batching | model-level batching over a fixed slot pool, chunked prefill, GQA without K/V copies; serial worker kept for A/B | paged KV, prefix caching and CUDA graphs not implemented (cannot be verified without a GPU; not claimed) |
| 4. Workloads | seeded specs, Poisson and burst arrivals, cancellation, timeouts, SLO goodput, cold and warm starts, 1–128 clients | control/data-plane disruption left to tests |
| 5. Observability | continuous GPU and server telemetry, window statistics, active decode rate, overhead A/B | per-request KV-cache utilization beyond the pool snapshot |
| 6. vLLM | OpenAI-compatible client with pinned greedy sampling; isolated hash-locked environment; stage runner with guards; identical workloads by construction | not executed; needs a GPU |
| 7. Security and faults | crash audit; invariance check; tests for forwarding, redirects, allow-list, authentication and redaction | worker drain and removal exercised by existing control-plane tests, not a GPU stage |
| 8. Deliverables | code, tests, suite, runner, runbook, report generator, matrix | image **not published**; no GPU results; no report with real numbers |

## Risk register

| # | Risk | Likelihood | Impact | Mitigation | Status |
|---|---|---|---|---|---|
| R1 | A CUDA-only error in the batched path (as the bare-device `ValueError` was) | high: it has never run on CUDA | a wasted first session | cheapest stage first (`0.5b-serial`, `0.5b-batch`); preflight and smoke gates; evidence always written | open: cannot be retired without a GPU |
| R2 | bf16 batching changes numerics so outputs depend on co-batched requests | medium | correctness claim weakened | `batch_invariance` measures it; a deliberately broken worker is caught in tests | measured on the first GPU run |
| R3 | The 7B pool is tight (about 0.15 GiB spare at 64 slots) | medium | fewer slots than planned | the planner grants fewer slots, never silently more; a gate requires at least 32 | gated |
| R4 | bitsandbytes' CUDA kernels fail or are slow with torch 2.9.1 | medium | 14B stage unusable | the wheel ships a CUDA 12.8 binary; stack check and preflight fail loudly; session B starts after session A | open |
| R5 | NF4 is slower than bf16 at batch widths above one (per-step dequantization) | medium | a negative result | stated as a hypothesis; measured, not assumed | by design |
| R6 | The Python event loop or per-token JSON streaming bottlenecks before the GPU does | medium at 64–128 clients | throughput capped below the GPU's | server telemetry and the GPU-active decode rate separate the two | measured |
| R7 | vLLM's torch 2.13 / CUDA 13.0 needs driver 580+ | low on a recent host | vLLM stage unrunnable | the stage and installer check the driver before any download | gated |
| R8 | vLLM silently applies the model's sampling defaults | high if unaddressed | invalid comparison | explicit request fields, `--generation-config vllm`, and a stage guard | mitigated |
| R9 | Cost overrun | medium | money | per-experiment and per-stage hard caps, a budget ledger that bounds stage time, termination checklist | bounded for stage time only |
| R10 | Forgetting to terminate the Pod | the largest cost risk | money | the report of billing is manual; the runbook and the stage runner print the reminder; the lifetime backstop does not terminate | **cannot be automated away** |
| R11 | Two Pods are different physical cards | certain | uncontrolled variable | each vLLM stage runs in the same Pod as its TensorPilot stage | designed out within a session |
| R12 | Small samples make tails look precise | certain at c=1 | overclaiming | smallest-repeat flag, hollow markers, a caveat on every report | mitigated |
| R13 | A stage hangs past its cap | low | cost | `timeout` with a kill-after, a soft stop a minute earlier, a ledger entry regardless | tested with a runaway stub |
| R14 | An API key leaks into evidence | low | credential exposure | keys travel by environment only; archives are refused if they contain a key; redaction in traces | tested, including a refusal test |
| R15 | The published tags and images are mutable | low | irreproducible evidence | immutable tags and digests; a new fix gets a new tag; the baseline checksum is a test | enforced |

## Decisions worth being able to defend

- **Own forward pass over a fixed slot pool, not paged attention.** Chosen because it is verifiable on CPU (batched equals serial down to logits) and fails at startup. It trades memory efficiency
  for simplicity; the vLLM comparison is expected to show what that costs.
- **Architecture allowlist.** Qwen3 is refused because its QK-norm is not implemented; refusing is cheaper than a silent wrong answer.
- **Fail the whole worker on a batch step error.** A half-failed shared batch cannot be trusted; recovery is the supervisor's job, and the crash experiment audits it.
- **The handler states why it cancels.** Inferring timeout versus cancellation from a clock comparison was wrong at timer granularity.
- **vLLM in an isolated environment with hashes.** It cannot share TensorPilot's torch, and an unpinned install on a meter is not reproducible.
- **No version bump to 1.0.** The package says 0.9.0 until a GPU has verified this work; "1.0" in the milestone name is a plan, not a release.
- **Three layers of evidence, never mixed:** CPU and container tests prove logic; a GPU run proves performance and numerics; the report refuses to fill a gap between them.
