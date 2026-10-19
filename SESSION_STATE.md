# Current engineering state

TensorPilot v0.6.0 GPU runtime intelligence is implemented on branch
`codex/runtime-platform`. The source adds synchronized phase telemetry,
deterministic adaptive batching, workload-matrix preflight, throughput/latency
gates, 512-token decode support and bounded Apple Metal capture.

Validation currently passes 123 tests with Ruff clean. Real MLX evidence includes
three matched repetitions of serial versus fixed width four and fixed width four
versus adaptive width four, a changing-traffic width 1/8 run, a nine-case
120/504/2040-token by 128/256/512-output matrix and a real `.gputrace`.

Measured fixed width four versus serial: +137.3% median throughput, -39.4% P95
latency and +375.1% P95 inter-token latency. Adaptive maximum width four was at
parity with fixed width four under sustained demand; no speedup claim is made.
See `docs/v06-gpu-runtime.md` and `WORK_LOG.md`.

The v0.5 service was stopped before v0.6 experiments. Final packaging, isolated
wheel validation, local commit and restart of the persistent v0.6 supervisor are
the remaining checkpoint steps. Nothing is pushed or published.
