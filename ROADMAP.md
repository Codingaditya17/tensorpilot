# Engineering roadmap

## Implemented

- [x] Real local MLX transformer inference; original sequential baseline retained.
- [x] Streaming API, bounded admission/output buffers and concurrent load generation.
- [x] Exact streamed token accounting, raw timings and wall-clock throughput.
- [x] Request IDs, safe errors, shared bearer authentication and operational probes.
- [x] Prometheus counters, gauges and queue/TTFT histograms.
- [x] FIFO and shortest-output-budget scheduling with an aging override.
- [x] Waiting cancellation/expiry, generator cleanup checks and worker supervision.
- [x] Seeded mixed workloads, matched repeated experiments and regression gating.
- [x] Model/content/source provenance, package installation and CPU CI configuration.

## Current validation work

- [x] Characterize aging thresholds with mixed real MLX workloads.
- [x] Publish the measured latency/fairness tradeoff with per-class results.
- [x] Verify package installation in an isolated CPU-only environment.

## Next runtime depth

- [x] Recoverable journals and bounded task/record soak testing.
- [x] Real cache-array profiling and allocator sample-age metrics.
- [x] Worker progress watchdog and explicit degraded-state health semantics.
- [ ] Context/memory admission based on measured backend resource information.
- [x] Shared SDK/load stream protocol conformance.
- [ ] Broader backend conformance and pinned quality evaluation set.
- [x] Opt-in real MLX continuous batching with independent sequence state,
      cancellation, deadlines, failure isolation and matched throughput evidence.
- [ ] Broaden batching capacity/context/model coverage and sustained memory testing.

## Separate future modules

- [ ] Node acceptance/health rules and an auditable recovery state machine.
- [ ] Clearly marked cluster/GPU fault simulations; real kind control-plane tests.
- [ ] Adapter boundaries for real AMD GPU/ROCm metrics when hardware is available.
- [ ] Model artifact registry, quality/performance validation and canary rollback.

No cluster, ROCm, RDMA or failover behavior is represented as real hardware
measurement on this Mac. A feature moves to "implemented" only with code and
appropriate validation evidence recorded in `WORK_LOG.md`.
