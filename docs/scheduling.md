# Scheduling and admission

The service has one physical execution worker. Policies choose among waiting
requests when that worker becomes free. They do not implement batching or
preemption; a long active prefill/decode operation still delays every waiting job.

## Policies

- `fifo` (default): dispatch in admission order. This is the baseline.
- `shortest-budget`: prefer the smallest declared `max_tokens`, preserving FIFO
  among equal budgets. If the oldest waiting job has waited at least the aging
  threshold, dispatch that job instead. This protects older large requests from
  being repeatedly bypassed by a continuous stream of small requests.

```bash
python -m tensorpilot.server --scheduler shortest-budget --max-queue-wait 1.0
```

The flag is an **aging threshold**, not a queue-latency guarantee. Dispatch still
depends on active request duration and on older aged requests ahead of a job.
The chosen output budget is an upper bound, not actual output length. This cost
hint ignores prompt length and can be misleading or manipulated by callers.
Only experiments can establish whether the policy benefits a specific workload.

`done` events record `scheduler` and `schedule_reason` (`fifo`, `shortest-budget`,
or `aging`). `/health` reports current policy/threshold. Prometheus counts dispatch
decisions using the bounded `reason` label.

## Admission and lifecycle

Capacity limits **waiting** jobs; one additional job may be active. A full queue
returns 503 immediately. Each admitted job has a deadline timer while queued, so
an expired request releases its slot even when a GPU operation is still running.
Disconnect cancellation removes a waiting job immediately. Active cancellation
waits for a generation boundary. Repeated cancellation is safe and counted once.

Transitions are `queued -> running -> completed/failed/cancelled/timed_out`.
Waiting jobs can also become `cancelled` or `timed_out` without execution. Queue
join accounting includes both active and waiting jobs and remains balanced when
requests are discarded. Each job records exactly one terminal outcome.

Before publishing `done`, the worker closes its generator. Cleanup failure marks
the backend unhealthy, fails waiting jobs, rejects new work, and makes readiness
fail. A worker task that stops unexpectedly also fails pending requests instead
of leaving them stranded. Model/runtime recovery currently requires a restart.

## Validation boundaries

Deterministic scheduler tests inject a clock and cover equal-budget ordering,
FIFO, shortest-budget, aging under continuing short arrivals, bounded capacity,
and wakeup/join behavior. CPU integration tests block a fake model step to verify
independent queued expiry/cancellation and test backend cleanup failure.

These verify policy mechanics. They provide no evidence of GPU speedup or a
hard bound on real request latency. Mixed-length MLX comparisons are recorded
separately in the experiment reports.
