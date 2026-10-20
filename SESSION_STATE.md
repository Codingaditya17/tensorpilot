# Current engineering state

TensorPilot v0.8.3 is implemented on `codex/v08-distributed-scale`. The immutable
v0.7 baseline is commit `675d1fb`; the first thousand-worker routing checkpoint is
`40d031a`, and operational scalability checkpoint v0.8.2 is `0f99e10`.

The control plane now has indexed power-of-two placement, constant-size health,
revision-safe worker pagination, bounded heartbeat concurrency/timeouts and
fleet-safe metric cardinality. A GPU-free harness exercises real loopback HTTP with
independently scheduled control plane, client and worker endpoints, including
latency/jitter, heartbeat hangs, offline workers, new instances and safe
register/drain/remove churn.

Final validation passes 152 tests with Ruff clean. Three repeated 32-worker fault
runs found median useful throughput of 1,084/1,019/982 requests/s at concurrency
128/512/1,000. Median routing P95 rose from 137 ms to 2.30 s and 3.75 s. More
concurrency did not increase useful throughput; the single streaming proxy event
loop is the next measured bottleneck. These are one-Mac synthetic network results,
not independent-machine or GPU scaling evidence.

Nothing is pushed or published. Raw evidence is retained under
`results/v082_operational_scale/` and `results/v083_synthetic/`.
