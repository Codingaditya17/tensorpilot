# v0.8.4 control/data-plane separation

## Problem and design

The v0.8.3 control plane selected workers quickly, then kept every streaming
worker connection on the same event loop that handled heartbeats, membership and
placement. At high concurrency, stream copying delayed both client responses and
cluster management.

v0.8.4 introduces a separate `tensorpilot-data-plane` service. A request now
follows four bounded steps:

1. The gateway asks `POST /v1/routes` for a compatible worker.
2. The control plane reserves one worker slot and returns a short-lived lease.
3. The gateway connects directly to that worker and owns stream backpressure and
   client cancellation. It never retries a stream after any output is visible.
4. The gateway completes the lease in `finally`. If it disappears, the control
   plane expires the lease and releases capacity.

The integrated `/v1/generate` endpoint remains available as the preserved
comparison path. Shared request validation prevents the two paths from accepting
different contracts. Control credentials are kept in a dedicated HTTP session
and are never forwarded to workers. Worker drain, compatibility, capacity and
failure decisions remain control-plane responsibilities.

## Matched experiment

The benchmark used 32 independently scheduled loopback worker endpoints, one
control event loop, a separate data event loop in separated mode, latency and
jitter, four hanging/offline workers, restart and register/drain/remove churn.
Each cell has three trials. Concurrency 128 used 5,000 requests; 512 and 1,000
used 10,000. Values below are medians.

| concurrency | architecture | useful req/s | error rate | routing P95 | routing P99 | detection | recovery |
|---:|---|---:|---:|---:|---:|---:|---:|
| 128 | integrated | 1,155.8 | 2.24% | 0.125 s | 0.156 s | 0.725 s | 0.293 s |
| 128 | separated | 647.6 | 1.20% | 0.222 s | 0.249 s | 0.615 s | 0.186 s |
| 512 | integrated | 1,110.1 | 1.11% | 1.977 s | 3.371 s | 0.751 s | 0.816 s |
| 512 | separated | 608.9 | 0.04% | 0.946 s | 1.005 s | 0.519 s | 0.591 s |
| 1,000 | integrated | 1,048.5 | 1.60% | 3.916 s | 5.147 s | 0.713 s | 2.958 s |
| 1,000 | separated | 577.3 | 0.00% | 1.921 s | 2.169 s | 0.065 s | 0.332 s |

Separation improved isolation, tail latency under saturation, and failure
recovery. At concurrency 1,000 it reduced routing P95 by 50.9%, routing P99 by
57.9%, and median recovery time by 88.8%; all three separated trials completed
10,000/10,000 requests.

It did **not** improve throughput. Useful throughput fell 44–45% at every tested
concurrency because each request adds a control-plane HTTP exchange and a second
Python HTTP proxy. At concurrency 128, where the integrated loop was not yet
saturated, separated routing latency was worse. Peak traced Python memory also
rose from 56.3 MB to 107.2 MB at concurrency 1,000. This is a reliability
architecture checkpoint, not a performance win.

The next measured bottleneck is the single gateway process plus per-request
lease RPC overhead. A credible next experiment would use multiple gateway
processes, a persistent multiplexed control channel or batched lease acquisition,
and independent hosts. Per the v0.8 plan, synthetic optimization should pause
here while real independent-machine and GPU validation is pursued.

## Evidence and limits

Raw request records, metrics excerpts and configuration live in
`results/v084_separation/`; `summary.json` contains machine-readable medians.
Ruff and 154 CPU/integration tests pass.

Everything ran in one Python process on one Apple M5 Mac using real loopback TCP.
Threads had independent asyncio loops but still shared the GIL, CPU and memory.
No GPU was allocated. These results do not establish multi-host, multi-GPU, AMD,
or production scalability.
