# v0.8 distributed scalability and fault-tolerance plan

v0.7 commit `675d1fba65092183467b86dbe5b525ccdb2a871a` is the immutable
comparison baseline. v0.8 work lives on `codex/v08-distributed-scale`. Every
optimization must use a recorded workload and report availability, throughput,
P95/P99 latency and resource cost against the baseline.

## Scaling audit

The v0.7 control plane is correct for a small local worker set, but these design
choices prevent credible large-cluster operation:

1. Placement scans every worker and runs under one global asyncio lock. Selection
   is O(workers), and heartbeat updates and stream releases wait for the same lock.
2. Pull health starts a task for every worker every interval. The aiohttp connector
   defaults to a small global connection pool, so probe cycles can overlap once
   worker count or network latency grows.
3. `/health` materializes every worker plus two audit histories. Response size and
   serialization time grow linearly and can block operational polling.
4. Registry and drain state are process-local. A control-plane restart loses
   membership, reservations and audit state; multiple control-plane replicas would
   make conflicting placement decisions.
5. The proxy parses and copies every streamed token. One event loop owns admission,
   scheduling, health traffic and every client/worker stream.
6. There is no worker epoch/fencing protocol, durable request identity or
   deduplication. Automatic replay after ambiguous failure would be unsafe.
7. The load client retains request records and token timings in memory. One client
   process can become the benchmark bottleneck before the control plane does.
8. The local launcher is capped at eight loopback workers and has no remote node
   bootstrap, service discovery, TLS, topology awareness or independent GPU model.

## Measurement sequence

### M1: establish limits without GPU load

- Benchmark registry registration, full heartbeat updates, placement P50/P95/P99,
  placement throughput, snapshot cost, memory and distribution fairness at 100,
  500 and 1,000 simulated workers.
- Add an end-to-end synthetic worker fleet and distributed load drivers for 1,000,
  5,000 and 10,000 concurrent streams. Stream payload and delay distributions are
  fixed and seeded. Report client CPU and event-loop lag so the driver cannot hide
  the control-plane limit.
- Inject delayed probes, refused connections, truncated streams and worker churn.
  Measure detection time, eviction time, false eviction rate, recovery time and
  request loss by failure phase.

Exit criterion: reproduce the saturation point three times and identify the first
resource that reaches its bound.

Initial 20,000-placement run on the Apple M5, with 1,000 simultaneous synthetic
reservations and no network or GPU work:

| Workers | v0.7 placements/s | v0.7 P50 | power-of-two placements/s | candidate P50 |
|---:|---:|---:|---:|---:|
| 100 | 9,258 | 102.8 µs | 42,248 | 15.3 µs |
| 500 | 2,164 | 452.8 µs | 56,905 | 15.2 µs |
| 1,000 | 1,060 | 901.3 µs | 56,358 | 15.3 µs |

The v0.7 scan is the first demonstrated limit. The indexed power-of-two candidate
kept exact distribution fairness in this uniform workload and improved 1,000-worker
placement throughput by 53.2×. This microbenchmark does not include lock contention,
HTTP or failures. Snapshot generation still scales linearly: at 1,000 workers it
produced about 745 KB and took 36–40 ms, making API pagination/summary separation
the next measured control-plane target.

### M2: remove measured control-plane bottlenecks

- Separate membership mutation from request placement. Maintain indexed eligible
  sets by model/artifact and a capacity heap or power-of-two candidate sampler.
- Bound heartbeat concurrency, add jitter and measure complete sweep age. Publish
  stale-worker and probe-duration histograms.
- Paginate worker APIs and make `/health` a constant-size summary. Move audit reads
  to bounded cursor endpoints.
- Tune separate HTTP pools for probes and inference. Measure stream proxy event-loop
  delay before considering multiple routing processes.

Each change is accepted only when the same M1 workload improves the limiting
metric without violating fairness, compatibility, reservation accounting or
failure semantics.

#### v0.8.2 measured checkpoint

The health response now contains only aggregate counts, membership revision,
uniform model/artifact identity and bounded audit sizes. Worker details moved to a
maximum-500-item paginated endpoint. Cursors contain a membership revision and
return an explicit conflict after registration or removal, so clients restart the
scan instead of silently skipping or duplicating workers.

Heartbeat probes now pass through a configurable fixed-size queue with separate
connect and total timeouts. Metrics record queue depth, duration, sweep duration,
outcomes, ready-state transitions and health-handler latency. Probe traffic uses a
separate HTTP session from inference proxy traffic.

| Synthetic workers | Legacy snapshot | Summary payload | Summary P50 | Full paginated scan |
|---:|---:|---:|---:|---:|
| 100 | 123 KB | 387 B | 0.052 ms | 3.4 ms / 1 page |
| 1,000 | 745 KB | 391 B | 0.490 ms | 33.7 ms / 2 pages |
| 5,000 | 3.51 MB | 391 B | 2.69 ms | 171.1 ms / 10 pages |

The payload is constant-size and about 1,900× smaller than the old 1,000-worker response.
The aggregation pass is still O(N), reaching 2.69 ms at 5,000 workers. Maintaining
incremental state counters is a future optimization if HTTP measurements show this
CPU cost becoming limiting. A complete worker inventory remains deliberately O(N)
and explicit through pagination.

### M3: distributed correctness

- Define worker leases with monotonically increasing epochs. Reject heartbeats and
  completions from replaced instances.
- Persist desired membership, deployments and rollouts in a transactional store.
  Start with SQLite for zero-cost local fault tests; keep a storage interface for a
  later consensus-backed implementation.
- Add idempotency keys and worker-side terminal-result records. Keep replay disabled
  until tests prove which failure phases are safe.
- Build a deterministic network-fault harness for latency, loss, disconnects and
  partitions. Validate drain and recovery state machines under process restarts.

#### v0.8.3 measured checkpoint

The synthetic harness uses real loopback TCP. The load driver, control plane and
each worker own independent event loops, preventing one simulated worker's delay
from blocking another worker. Workers expose the real health and NDJSON interfaces.
The scenario applies seeded latency/jitter, hangs four health endpoints, marks the
same workers offline, restores them with new instance IDs, registers a live churn
worker, drains it, waits for zero routes and removes it.

Three final repetitions used 32 workers, 10 ms execution, 3 ms jitter, a 500 ms
heartbeat timeout and a 400 ms injected outage:

| Concurrency | Requests/run | Successful req/s median | Error median | Routing P95 median | Routing P99 median | Recovery median |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 5,000 | 1,084 | 2.02% | 137 ms | 160 ms | 282 ms |
| 512 | 10,000 | 1,019 | 1.50% | 2.30 s | 4.23 s | 8.75 s |
| 1,000 | 10,000 | 982 | 3.57% | 3.75 s | 5.23 s | 6.04 s |

Successful throughput peaked at the lowest tested concurrency. Increasing client
concurrency reduced useful throughput and inflated tail routing latency before
memory became a concern. Traced Python peak memory medians were about 13.8, 34.9
and 55.6 MB. The first harness version also exposed two correctness issues that
were fixed and retained in regression tests: worker occupancy and local routes
were double-counted, and a churn worker could not be removed safely before drain.

Heartbeat tuning showed a real tradeoff. At 512 concurrency, a 100 ms timeout
caused 18.9% rejection and 8.25 s recovery; 500 ms reduced rejection to 1.88% in
the matched tuning run and recovery to 0.83 s, while detection slowed from 0.30 s
to 0.93 s. Final repeated runs still showed wide recovery variance under stream
load. The current bottleneck is the single control-plane event loop proxying every
stream: placement microbenchmarks remain fast, but HTTP routing queues dominate
P95/P99 at 512+ concurrency.

### M4: MLOps deployment control

- Store immutable model manifests with artifact digest, tokenizer/config digest,
  runtime constraints and benchmark gates.
- Implement staged rollout states: validate, canary, promote, halt and rollback.
- Route a fixed traffic fraction to a canary and compare error rate, P95/P99,
  tokens/s and memory against the active version. Rollback decisions and thresholds
  are recorded and reproducible.

### M5: independent GPU validation

- Package workers for Linux NVIDIA and AMD ROCm without claiming backend parity.
- Use free university/HPC access or explicitly approved resources only. Record GPU,
  driver, runtime, topology, network and model artifact provenance.
- Run single-GPU baselines first, then two independent machines. Measure aggregate
  tokens/s, per-request P95/P99, placement efficiency, memory high-water marks,
  control-plane overhead and recovery time. Compare against v0.7 methodology while
  clearly separating different hardware.

## Mac safety and budget

Synthetic M1–M3 tests allocate no MLX model and default to bounded worker/request
counts. Every stress command has hard caps, writes recoverable evidence and runs a
preflight estimate. GPU experiments remain sequential unless a recorded memory
budget permits otherwise. No paid service or hardware is used without explicit
agreement.
