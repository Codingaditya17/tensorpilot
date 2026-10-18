# TensorPilot engineering work log

This is the durable record of implemented work and measured evidence. Dates use
America/New_York. Real MLX experiments and CPU-only tests are identified explicitly.
No scheduler optimization or production availability claim is assumed in advance.

## Starting state — 2026-10-17

- v0.1 sequential benchmark preserved, with its approximate token-count caveat.
- v0.2 bounded streaming MLX service and concurrent workload generator implemented.
- Added versioned generation API, request IDs, shared bearer token authentication,
  probes, Prometheus metrics, Python integration client and CPU CI configuration.
- Local validation: **13 CPU tests passed**, no warnings.
- Real MLX service smoke: 10 successes / 0 failures; overload: 9 successes / 31
  HTTP 503 rejections; recovery: 5 successes / 0 failures.
- Prometheus validation matched 25 completed jobs, 31 rejected jobs and 800 output
  tokens (includes one 32-token Python-client smoke request).
- Evidence: `results/service_validation.json` and referenced raw workload reports.
- CI configuration exists; GitHub-hosted CI has not been executed.
- Project currently lives inside a parent Downloads Git checkout. Changes are
  restricted to this project's files; no parent Git state is staged or committed.
- A pre-upgrade source backup is retained in the Codex working directory.

## Current plan

1. Bounded scheduler abstraction: FIFO baseline plus shortest declared output
   budget with an oldest-job aging override. Deterministic fairness tests.
2. Immediate cancellation and expiry of waiting requests, correct accounting,
   generation cleanup reliability and worker readiness supervision.
3. Mixed-length, seeded workloads and repeated policy comparisons, preserving
   raw records, environment details and honest statistical limits.
4. Packaging and operational contract improvements justified by findings.

Each completed milestone below must include changes, validation and limitations.

## Milestone 1 — fair scheduling and queue lifecycle — 2026-10-17

Implemented:

- Added an explicit bounded scheduler with FIFO and shortest-output-budget policies.
- Added oldest-job aging override, stable ties, per-request decision metadata and
  Prometheus dispatch counters.
- Queued deadlines now expire independently of active model execution. Waiting
  cancellation immediately frees admission capacity and has exactly-once outcomes.
- Model generator cleanup now precedes successful completion. Cleanup failure
  fails readiness/admission and drains waiting work. Dead worker supervision also
  fails waiting clients. Shutdown remains idempotent.
- Added scheduling architecture/limitations in `docs/scheduling.md`.

Validation: **30 CPU tests passed**, no warnings, in the existing Python 3.14
environment. Includes an injected-clock continuing-arrival fairness test, a
blocked-model queue-expiry test, cancellation, cleanup failure, dead-worker
supervision, real engine dispatch order and all previous HTTP/client tests.

Limits: no GPU scheduler-performance claim yet. Output budget is a cost hint;
aging is not a hard latency guarantee. Active GPU operations are not preemptible.
The already-running service still uses the preceding source until its next restart.

## Milestone 2 — reproducible experiments and packaging — 2026-10-17

Implemented:

- Added seeded uniform/mixed workload plans and content fingerprints, per-class
  reports, raw request IDs/budgets, and client/server token-count consistency checks.
- Added immutable model revision resolution plus streamed file-content hashing,
  actual MLX device details (Apple M5 / 16 GiB), runtime dependency versions and
  source fingerprints. Offline cached loading is supported.
- Added an experiment orchestrator that refuses an occupied port, owns one server
  process at a time, counterbalances policy order and warms each workload class.
- Added atomic JSON reports, incremental experiment manifests, source archives,
  installed-version snapshots and a comparison tool that rejects mismatched runs.
- Added an explicit P95 regression gate with failure exit status.
- Added installable package metadata, editable installation, CLI entry points,
  static checks and CI configuration. Version is 0.3.0.
- Reorganized README, architecture, roadmap and historical benchmark notes.

Validation:

- Editable package installed successfully. **46 CPU tests passed**; Ruff passed.
- Executed **18 real MLX runs / 540 measured requests**, zero failures/rejections,
  plus separate warmup requests. Three matched repetitions per policy at each
  aging threshold (0.25 s, 1 s, 2 s).
- At 1 s: median latency -28.1%, overall P95 +12.2%, output throughput -0.2%.
- At 2 s: short P95 -28.1%, long P95 +106.0%; aggregate throughput unchanged.
- The 1 s candidate failed the explicit 5% P95 gate (expected exit status 1).
- Retained FIFO as the default. Full study: `docs/experiments/scheduler_2026-10-17.md`.
- Raw evidence: `results/scheduler_mixed_seed42*` directories. Source ZIPs include
  the README required by package metadata; archive hashes are saved in manifests.

Limits: three prompt templates, small per-class samples and closed-loop arrivals.
These experiments establish an observed tradeoff, not statistical significance
or universal policy superiority. No GitHub-hosted CI run has been performed.

## Milestone 3 — worker progress watchdog — 2026-10-17

- Added configurable model-call stall detection (30 seconds by default).
- A stall fails readiness, active/waiting requests and new admissions while leaving
  the asyncio HTTP loop responsive. Public health includes the degraded reason.
- Added backend-call-age and watchdog-trip metrics. Output-buffer backpressure
  is deliberately excluded from model-call stall detection.
- Validation: **48 CPU tests passed**, no warnings; Ruff passed. Injected blocked
  backend and slow-reader tests verify degradation and avoid double-counted outcomes.
- Limit: the watchdog cannot interrupt a GPU operation or restart the process.
  A process supervisor is still needed for a permanently stuck runtime.


## Milestone 4 — recoverable workloads and soak — 2026-10-18

- Added append-only raw journals, unique default filenames, explicit overwrite,
  flushed records, optional per-record fsync, and atomic final reports.
- Added fixed-worker duration runs, repeated seeded plans, bounded request counts,
  compact reports and complete/interrupted journal recovery. Corruption in the
  middle, duplicates and footer-count mismatches fail; only a truncated final
  JSON/UTF-8 record is tolerated. Partial/soak windows cannot enter regression gates.
- Real MLX soak: **182 successes, zero failures/rejections**, 30.581 s including
  drain, 8,896 tokens, 290.9 output tokens/s. Thirty monitor samples were ready;
  observed queue depth <=1, zero watchdog trips. Maximum cached active MLX sample
  was 281,161,480 bytes (not process RSS or system pressure).
- Complete journal recovery reproduced every summary value exactly.
- Deliberate real-load interruption: exit 130; four completed records recovered;
  unknown in-flight outcomes excluded. Service returned idle/ready and generated
  a subsequent response successfully.
- Evidence: results/soak30_v03*, results/interruption_v03*, verified source ZIP.

## Milestone 5 — real cache analysis and metrics — 2026-10-18

- Added MLX prompt-token counts and finish reasons, opt-in per-token cache-owned
  array nbytes and active allocation samples, with unknown values for unsupported
  cache types. Fresh caches are owned independently by each request.
- Added allocator active/peak/sample-age metrics, collected on the model thread.
- Real profiling: **15 measured requests plus five excluded warmups**, all passed.
  Prompt counts 19/75/267/523/1,035 gave median cache-array peaks 3/3/6/9/15 MiB.
  Includes spare cache capacity; not total physical GPU allocation. No savings or
  throughput improvement claimed. Profiling timings include collection overhead.
- Added profiling CLI, methodology, curated study and verified source snapshot.
- Evidence: results/cache_profile_v03*; docs/experiments/cache_growth_2026-10-18.md.

## Milestone 6 — protocol and release validation — 2026-10-18

- Shared SDK/load protocol checks reject request-ID mismatch, invalid payloads,
  token-budget/count violations, nonfinite timings and incomplete streams.
- Added distinct client-source provenance and client context/deadline checks.
- Final source: **74 tests passed**, no warnings; Ruff passed.
- Built the final wheel, installed it outside the source checkout in an isolated
  CPU environment, and passed **all 74 copied tests with no MLX/mlx-lm installed**.
  This is local macOS/Python 3.14 evidence; remote Linux CI was not run.
- Final GPU harness smoke: both five-request policy runs passed, including raw
  journals, archives and comparison publication. Too small for performance claims.
- Final GPU service checks: normal 10/10; overload 9 completed /31 rejected;
  recovery 5/5; hardened SDK generation passed. Prometheus reconciled exactly:
  25 completed jobs, 560 output tokens, 31 rejected, zero watchdog trips.
- Evidence: results/final_validation_v03.json and results/final_harness_smoke_v03.
- Editable CLI installation refreshed. A no-build-isolation attempt failed because
  setuptools is absent from the runtime environment; the normal isolated build
  succeeded without changing runtime dependencies.

## Final checkpoint

Service runs locally at 127.0.0.1:8080, FIFO, eight waiting slots, 6 GiB MLX limit,
profiling disabled. Runtime log: results/runtime_v03_final.log. Raw experiments
stay under ignored results/. Source/evidence ZIPs, a wheel and progress notes are
exported to the Codex chat outputs directory; no weights, venv or credentials.
A dedicated local Git repository/checkpoint keeps project history separate from
unrelated Downloads files. Nothing is published or pushed remotely.

Remaining scope: real batching, durable serving queues, per-tenant identity,
automatic process restart, broad model-quality evaluation, distributed/AMD GPU
execution and cluster fault control. These are roadmap items, not completed claims.
