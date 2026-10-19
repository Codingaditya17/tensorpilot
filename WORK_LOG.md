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


## Continuation — experimental batching adapter — 2026-10-18

Inspected installed continuous-batching API and tokenizer ownership. Added an
experimental model-thread adapter with independent sequence IDs, detokenizers,
budgets and cancellation. It is not connected to HTTP serving. Validation:
**78 CPU tests passed**, Ruff clean. Real batch execution and engine integration
are pending; design and boundaries are in docs/batching-design.md. Existing v0.3
release artifacts remain the earlier 74-test checkpoint, unchanged.

## Continuation — v0.4 continuous batching — 2026-10-18

Implemented opt-in BatchEngine and MLXBatchAdapter with independent UID state,
continuous admission, bounded delivery, per-sequence removal and shared-context
failure handling. Serial/FIFO remains default. Extended matched experiment and
comparison tools with execution-mode axis and configuration mismatch rejection.

Validation: 95 tests pass, Ruff clean. Real adapter greedy token equality passed.
Real HTTP lifecycle: six completions, independent disconnect cancellation,
overload 10 complete/30 rejected, recovery four complete; 21 completed jobs and
one cancellation reconcile with metrics. Three 30-request repetitions per mode:
180/180 successful, batch median 450.58 vs serial 288.46 tokens/s (+56.2%); P95
0.8852 vs 1.4198 seconds (-37.7%). Shared allocator maximum 348,010,500 vs
341,534,144 bytes. Scope/limitations and reproduction: docs/batching-design.md.
Evidence: results/batch_execution_seed42, results/batch_http_validation.json,
results/batch_adapter_conformance_v01.json. Experiment source ZIP retains exact
measurement source before version-label/doc updates. Earlier v0.3 exports remain
unchanged. No remote push/publication.

Follow-up validation: 30-second batch soak completed 288 requests, zero failures/
rejections, 13,872 output tokens, 452.99 tokens/s over 30.623 seconds. All 20
readiness snapshots healthy. Raw journal/report/monitor in results/batch_soak*.
Hardened regression gate: reject candidates with increased rejections, even when
latency improves. Added dedicated test. Final wheel CPU verification recorded
below after rebuild. This short soak does not establish long-term memory bounds.

Final v0.4 wheel rebuilt and installed outside the checkout: all 95 tests passed
in the isolated CPU environment with no MLX dependency. Source lint clean.
Restored local serial/FIFO server at 127.0.0.1:8080; log results/runtime_v04_serial.log.

## v0.5 runtime hardening — 2026-10-18

Added a local browser dashboard after diagnosing that `/` returned 404 and the
service process was down. The page shows live health/model/queue/device state and
a streaming inference console. Added deterministic resource admission, guarded
soak presets, exact token-reference gates and restart supervision. KV reservation
estimates remain explicitly separate from real MLX allocator samples.

Validation before the long soak: 104 tests passed and Ruff clean. Real serial
reference and batch verification matched exact token IDs for three pinned cases.
A deliberate 10 MiB reservation budget admitted one of six simultaneous long
requests, rejected five for memory pressure, completed 256 output tokens, stayed
ready and released all reservations. A controlled SIGKILL of ready worker PID
98639 produced logged exit -9; the supervisor restarted PID 98764, gated on
readiness, and a five-request recovery smoke passed. Nothing pushed or published.

Final guarded soak: 3,650/3,650 successful, zero failed/rejected, 233,600 tokens
over 900.402 seconds (259.44 tokens/s). P50/P95/P99 was 0.489/0.521/0.548 seconds;
180/180 guard samples were ready and active MLX allocation stayed about 268 MiB.
The installed v0.5 wheel passed all 104 tests without MLX. Browser UI was visually
verified and streamed 64 tokens. Final in-flight crash: one token arrived, the
verified worker was SIGKILLed, the client received `transport_error`, a new ready
instance appeared, and an eight-token recovery request completed. Evidence is in
results/v05_soak_15m_run2*, results/v05_inflight_crash_recovery.json and
docs/v05-runtime-hardening.md.

## v0.6 GPU runtime intelligence — 2026-10-18

Added synchronized model-thread phase timing, batch-wide counters, per-request
TTFT/inter-token profiles, startup phase timing, estimated KV provenance and
Prometheus occupancy/step metrics. Added a deterministic adaptive controller with
live decision explanations and fixed/FIFO baselines preserved. Added a bounded
128/512/2048 by 128/256/512 matrix generator, 512-token service support,
throughput regression gates and a real Apple Metal capture command.

Correctness validation: 123 tests pass and Ruff is clean. New coverage includes
traffic-dependent width decisions, memory-pressure caps, reservation cleanup,
prefill cancellation isolation, live health state, matrix safety and runtime
counter deltas. Existing v0.5 compatibility, crash supervision and stream tests
remain in the suite.

Real MLX comparisons used three counterbalanced repetitions per pair. Serial vs
fixed width four completed 180/180 requests; fixed batching delivered +137.3%
median throughput and -39.4% median P95 latency, with +375.1% P95 inter-token
latency. Fixed width four vs adaptive max four completed 180/180; throughput was
+0.13% and P95 latency -0.02%, effectively parity because sustained demand kept
both at width four. No adaptive speedup claim is made.

Changing traffic selected width one then width eight with zero failures. The
nine-case context/output run recorded actual inputs 120/504/2040 and output caps
128/256/512; all completed, with 2,688 output tokens. Synchronized batch-wide
work was 0.065 s prefill and 6.954 s decode; mean occupancy was 1.73. Maximum MLX
active allocation was 339,040,188 bytes and process-lifetime peak 614,778,088.
A real 616 MiB `.gputrace` was captured for one bounded 16-token request; its
27.2 s captured timing is diagnostic overhead and excluded from performance claims.

Evidence: `results/v06_serial_vs_fixed4/`, `results/v06_fixed4_vs_adaptive4/`,
`results/v06_context_output_matrix_run.json`, `results/v06_adaptive_*_traffic*`,
`results/v06_experiment_matrix.json`, and `results/v06_kernel_profile.json`.
