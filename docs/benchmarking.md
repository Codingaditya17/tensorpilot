# Workload generation, soak testing and recovery

Start the normal local server first. Every CLI workload writes an atomic final
JSON report plus a raw per-request JSONL journal beside it. Default filenames are
unique. Existing report/journal files require `--overwrite`; keep separate files
for repeated experiments instead.

## Fixed-count and paced workloads

```bash
tensorpilot-load --requests 30 --concurrency 6 --workload mixed --seed 42 \
  --output results/mixed.json
tensorpilot-load --requests 30 --concurrency 6 --workload mixed --seed 42 \
  --rate 4 --output results/paced.json
```

Mixed plans contain a fixed shuffled 60/20/20 composition of short, medium and
long output budgets (16/64/128). A seed controls the shuffle. For request counts
not divisible by five, composition truncates the repeating class pattern before
shuffling. The three prompt templates are a controlled workload, not a general
sample of production traffic. Save a validated custom JSON plan with exact
`prompt`, `max_tokens`, and `class` fields and pass `--workload-file` to use it.

Rate mode schedules target arrivals but caps concurrency. Saturation delays
dispatch; raw records and P95 dispatch lag reveal this. It is not an uncapped
open-loop generator. HTTP 503 alone is not necessarily an overload rejection:
reports count the explicit `overloaded` error code separately from other failures.

## Time-based soak

```bash
tensorpilot-load --duration 30 --requests 30 --concurrency 2 \
  --workload mixed --compact --max-requests 500 \
  --output results/soak_30s.json
```

In duration mode, `--requests` is the **plan cycle length**. The runner repeats
that plan until the dispatch duration or the request safety cap is reached, then
drains outstanding requests. Wall-clock throughput includes the drain. A default
cap of 10,000 total requests bounds retained record growth; explicit caps can go
up to 100,000. Concurrency is limited to 256 and duration to one hour per run.

`--compact` stores full token timing/cache profile arrays in the journal while
retaining smaller request records for final summaries. Task count is fixed;
there are no thousands of waiting HTTP tasks. This bounds task and record growth,
but is not a guarantee about total macOS memory pressure.

## Journals and interrupted runs

Each journal starts with run/configuration/provenance metadata, then records each
completed request (success or failure), then a status/window footer. It contains
prompt text. Records are flushed individually; normal interruption recovery is
supported. `--journal-fsync` additionally requests filesystem durability after
each record and changes measured client overhead. The footer is always fsynced.

On `Ctrl-C`, the runner cancels HTTP tasks, closes the journal with interrupted
status, and exits 130. A hard kill may leave no footer or a truncated final line.

```bash
tensorpilot-recover results/soak_30s.requests.jsonl \
  --output results/soak_recovered.json
```

Recovery preserves valid completed records, tolerates only an incomplete final
JSON/UTF-8 line, rejects corruption in the middle and duplicate request indices,
and verifies footer counts. Without a footer, the observation window ends at
the last recorded completion. In-flight requests are unknown and never counted
as successful. Such a partial-window throughput is not a complete experiment.
The scheduler comparator refuses recovered or duration-based runs.

## Measurement contract

- Successful output counts must agree between received token events and the
  server's terminal count; mismatches fail the request.
- Full workload wall time, not summed request durations, is the throughput
  denominator. Report error/rejection rates alongside successful-request tails.
- Per-class throughput uses the same whole-run denominator.
- Raw server token timestamps include queue wait. Client timestamps additionally
  include transport and buffering. Neither is a GPU kernel timeline.
- Match workload, model artifact, sampling, memory/context limits, runtime,
  source/configuration, and profiling mode before comparing policies.

The [measured scheduling study](experiments/scheduler_2026-10-17.md) shows why
median latency alone is insufficient for a policy decision.
