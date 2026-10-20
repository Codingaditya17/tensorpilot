# Current engineering state

TensorPilot v0.9.0 is implemented locally on `codex/v08-distributed-scale`, on top of
the v0.8.4 checkpoint `7af0317`. The v0.9 changes are not pushed or published.

The CUDA worker is implemented and tested on CPU: the same streaming contract as MLX,
greedy decoding with an explicit KV cache, bounded failures, sampled `nvidia-smi`
telemetry, and a pinned Linux lock. It has not run on an NVIDIA GPU. The next
milestone is the Runpod session in `docs/v09-runpod-runbook.md`. Its outputs decide
whether any CUDA performance claim is made.

Validation at the release commit: 202 tests passed, ruff clean, a real pinned
checkpoint matched transformers greedy output on CPU, the sdist tests pass against
the installed wheel. The Dockerfile has not been built. See
`docs/v09-validation-report.md`.

## v0.8.4 state (preserved)

TensorPilot v0.8.4 is implemented on `codex/v08-distributed-scale`. The immutable
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

v0.8.4 separates streaming into a data-plane service using expiring route leases.
In three matched trials per architecture/concurrency, separation reduced saturated
routing tails and failure recovery but lowered useful throughput by 44–45%. At
concurrency 1,000, median routing P95 changed from 3.916 s to 1.921 s and recovery
from 2.958 s to 0.332 s; separated throughput was 577 req/s versus 1,048 req/s.

Nothing is pushed or published. Raw evidence is retained under
`results/v082_operational_scale/`, `results/v083_synthetic/`, and
`results/v084_separation/`. See `docs/v084-control-data-separation.md`.
