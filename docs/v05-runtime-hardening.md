# TensorPilot v0.5 runtime hardening

This milestone adds conservative resource admission, guarded soak tooling,
deterministic correctness gates, process restart supervision and a local browser
dashboard. Serial/FIFO remains the default. Previous evidence is preserved.

## Resource admission

Each job reserves estimated KV-cache capacity before entering the bounded
scheduler. The default estimate is `2 MiB + 16 KiB × (UTF-8 prompt bytes + output
budget)`. UTF-8 bytes provide a conservative prompt-token bound for ordinary
text. The coefficient and budget are configurable.

This is an overload-control estimate. It is not tokenizer output, cache-array
`nbytes`, MLX active allocation, process RSS, physical GPU memory, or unified
system pressure. Health and Prometheus use `estimated_*` names for reservations
and `last_measured_memory`/`mlx_*` for allocator samples. Reservations release
once on completion, rejection rollback, cancellation, deadline, failure and
shutdown paths.

In a deliberately constrained real MLX smoke at a 10 MiB estimate budget, six
simultaneous 256-token requests produced one completion and five
`memory_pressure` rejections. The completed request emitted 256 tokens; the
service stayed ready and reservations returned to zero. This validates policy
and accounting, not estimator accuracy. Evidence:
`results/v05_memory_admission_10mib_smoke.json`.

## Correctness gate

`tensorpilot-correctness REFERENCE --record` records token IDs twice under the
pinned artifact, revision, greedy sampling and prompt format. Verification
refuses metadata drift before comparing IDs. Three real cases (short, serving,
Unicode) recorded in serial mode matched the real batch path exactly at budgets
12, 20 and 16. Evidence: `results/qwen25_05b_4bit_reference_v01.json` and
`results/qwen25_05b_4bit_batch_verify_v01.json`.

## Worker recovery

`tensorpilot-supervisor` owns one server child, waits for `/readyz`, logs lifecycle
events, and restarts unexpected exits with bounded retries and backoff. In the
real test ready worker PID 98639 was terminated with SIGKILL. The supervisor
recorded exit `-9`, waited one second, started PID 98764 and confirmed readiness.
Five post-restart requests completed with no failures or rejections. Interrupted
in-flight requests cannot be replayed and must fail at the client. Evidence:
`results/v05_worker_lifecycle.jsonl` and `results/v05_post_restart_smoke.json`.

## Guarded stress tests

`tensorpilot-soak` provides 15/30/60-minute presets, deterministic context/output
shape, concurrency/rate controls, request caps, raw journals and a guard record.
It polls readiness and actual sampled MLX active bytes. Loss of readiness, probe
failure or crossing the threshold cancels dispatch and preserves recoverable
results. A sample can age during a long GPU operation, so this is a coarse process
guard rather than an operating-system memory-pressure controller.

## Validation boundary

The suite contains 104 tests and lint is clean. The final wheel also passed all
104 tests from an isolated CPU-only installation without MLX. The dashboard,
inference, memory-pressure policy, exact token gate and process restart were
exercised on the Apple M5.

The guarded 15-minute serial/FIFO run completed 3,650 requests, zero failed or
rejected, and 233,600 output tokens in 900.402 seconds (259.44 output tokens/s).
P50/P95/P99 client latency was 0.489/0.521/0.548 seconds. All 180 five-second
guard samples were ready; sampled MLX active allocation stayed about 268 MiB and
never approached the 4 GiB threshold. This establishes bounded stability for
this window/workload, not long-term leak freedom. Evidence:
`results/v05_soak_15m_run2*`.

The in-flight crash check received one token, SIGKILLed only the verified worker,
observed an explicit client `transport_error`, waited for a different ready
instance, then completed an eight-token request. Evidence:
`results/v05_inflight_crash_recovery.json`.

No Kubernetes, multi-GPU, AMD/ROCm or cluster behavior is represented by this
milestone.
