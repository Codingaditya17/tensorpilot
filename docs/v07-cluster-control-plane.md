# v0.7 local cluster control plane

TensorPilot v0.7 adds a real control-plane process in front of independently
supervised worker processes. It is designed to make placement and failure behavior
observable before introducing remote hosts or orchestration infrastructure. The
measured setup uses two processes on one Apple M5 GPU and is not evidence of
multi-GPU scaling.

## Run it

```bash
tensorpilot-cluster \
  --workers 2 \
  --offline \
  --execution adaptive \
  --batch-size 2 \
  --output-dir results/v07-cluster
```

The launcher starts worker supervisors on `127.0.0.1:8101` and `:8102`, waits for
both workers, then starts the control plane on `127.0.0.1:8090`. Its output folder
must not already exist so evidence from an earlier run cannot be overwritten.

```bash
curl http://127.0.0.1:8090/health
curl http://127.0.0.1:8090/v1/workers
tensorpilot-load --url http://127.0.0.1:8090 \
  --requests 20 --concurrency 4 --max-tokens 64 \
  --output results/cluster-load.json
```

The control plane supports `/livez`, `/readyz`, `/health`, `/metrics`,
`GET/POST /v1/workers`, worker heartbeat, drain and delete operations, and
`POST /v1/generate`. Generation preserves the worker NDJSON stream. Response
headers identify the worker and placement reason; terminal events include the
same placement record.

## Placement and admission

Registration accepts plain HTTP URLs with explicit ports and a configured host
allowlist. Credentials, paths, queries and fragments are rejected. Loopback is the
default boundary. A non-loopback control-plane bind requires
`TENSORPILOT_CONTROL_API_KEY`; worker calls can use
`TENSORPILOT_WORKER_API_KEY`.

Before dispatch, routing excludes workers that are unhealthy, stale, draining,
full, incompatible with an optional model/artifact constraint, or unable to reserve
the estimated KV bytes. Capacity-aware routing orders the remainder by current
load ratio, estimated reservation pressure, assignment count and stable worker ID.
The assignment count avoids repeatedly selecting the first worker when all live
signals tie. Round-robin is also available as an explicit policy.

The registry keeps the most recent 100 placement and lifecycle records in memory.
It stores request IDs, selection constraints, reasons and worker state; prompts are
never stored. Prometheus counters cover placement, rejection and proxy outcomes,
with gauges for registered, ready and in-flight workers.

## Failure semantics

Workers are polled through `/health`. Stale or failed workers leave the eligible
set. Supervisors restart crashed processes, and a changed worker `instance_id` is
recorded as lifecycle history. Draining immediately blocks new placement while
allowing existing streams to finish; removal is rejected while routes are active.

There is deliberately no automatic retry after dispatch. A worker can generate a
token and fail before the control plane observes the complete stream, so replay
could duplicate work or side effects. If transport fails before headers, the API
returns HTTP 503. If it fails after streaming begins, it appends a terminal
`worker_transport_error` event with the original worker request ID and placement.

## Real experiment

The experiment used the same pinned Qwen model artifact on both workers:
`909346d914889eadd92f3073fd789c19bf963886a07f88946551e975a79153a5`.

- Initial routing completed 20/20 requests, generated 1,280 tokens and placed 12
  requests on worker 2 and eight on worker 1.
- Draining worker 1 sent all six subsequent requests to worker 2; all succeeded.
- Worker 2 was killed after the first streamed token. The stream ended with
  `worker_transport_error`, no retry occurred, and a recovery request completed on
  worker 1. Restarting worker 2 produced a new observed instance ID.
- Three direct single-worker and three two-worker control-plane repetitions
  completed 180/180 requests. The two-process median throughput was 3.0% higher,
  while median P95 latency was 34.0% worse, P95 TTFT 23.3% worse and P95
  inter-token latency 76.7% worse.

These processes contend for one physical GPU and unified-memory subsystem.
Alternating runs reduce simple ordering bias but do not eliminate thermal or
background-load effects. The result shows that a control plane works and exposes
the cost of local process contention; it does not establish horizontal scaling.

## Remaining production work

The registry is in memory and static registration is the normal launch path.
There is no durable queue, consensus, service discovery, TLS termination,
tenant-level policy, worker-side request deduplication, Kubernetes controller or
independent-GPU measurement. Those boundaries must be addressed before treating
this as a remote production cluster.
