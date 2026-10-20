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

## v0.8.4 — control/data-plane separation — 2026-10-20

- Added expiring route leases and moved stream proxying/backpressure/cancellation
  into a separately runnable data plane. Preserved the integrated endpoint.
- Added control-loop lag, active stream, proxy duration/chunk, lease completion and
  expiry metrics. Control credentials are isolated from worker connections.
- Added matched integrated/separated synthetic experiments: three trials each at
  concurrency 128, 512 and 1,000, retaining raw request evidence.
- Separation reduced saturated P95/P99 routing latency and recovery time, and all
  three 1,000-concurrency trials completed without request failure. It also cut
  throughput 44–45%, exposing the gateway/RPC path as the next bottleneck.
- Validation: Ruff clean and 154 tests passed. This is one-Mac, GPU-free evidence;
  no multi-host, GPU performance or production-readiness claim is made.


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

## v0.7 local cluster control plane — 2026-10-19

Implemented an aiohttp control plane and worker registry with capacity-aware and
round-robin placement. Eligibility accounts for readiness, stale heartbeats,
draining, execution and queue slots, estimated KV reservations, model identity and
artifact SHA-256. Added bounded placement/lifecycle audit history, Prometheus
cluster metrics, authenticated non-loopback operation and strict worker URL
validation to keep registration from becoming an unrestricted proxy.

Added worker drain/removal/heartbeat APIs, compatibility generation proxying and
placement response metadata. Requests are never retried after worker dispatch. A
mid-stream connection loss emits a terminal `worker_transport_error` with the
original worker request ID and selected placement. Added `tensorpilot-cluster` to
own multiple existing restart supervisors and gate the control plane on readiness,
plus `tensorpilot-cluster-compare` to reject unmatched workloads, repetitions and
model artifacts.

Validation passes 141 source tests and Ruff. Tests cover fair ties, reservations,
stale/drain/model/artifact/memory exclusion, registration and authentication
boundaries, route accounting, lifecycle history, mid-stream failure and the
no-replay invariant. Exact installed-wheel validation is part of the release
checkpoint.

Real MLX validation used two adaptive workers with the same artifact SHA-256.
Initial load completed 20/20 requests and 1,280 tokens, split 12/8 across workers.
After draining worker 1, 6/6 requests selected worker 2. Killing worker 2 after its
first token yielded the expected terminal transport error without replay; a new
request completed on worker 1, and the restarted worker 2 appeared with a new
instance ID.

Three single-process and three two-process repetitions completed 180/180 requests.
Two local worker processes measured +2.97% median throughput, -20.95% P50 latency,
+33.97% P95 latency, +23.31% P95 TTFT and +76.68% P95 inter-token latency relative
to the direct single process. Positive latency percentages are regressions. Both
workers share one Apple M5 GPU, so this does not demonstrate cluster scaling.

Evidence: `results/v07_two_worker_load.json`, `results/v07_drain_load.json`,
`results/v07_inflight_worker_crash.json`, `results/v07_cluster_health_recovered.json`,
`results/v07_single_worker_run*.json`, `results/v07_two_worker_run*.json`,
`results/v07_cluster_comparison.json` and `results/v07_final_cluster_*`.

## v0.8 scalability baseline and first routing candidate — 2026-10-20

Preserved v0.7 commit `675d1fba65092183467b86dbe5b525ccdb2a871a` and
created branch `codex/v08-distributed-scale`. Added a bounded, GPU-free registry
saturation benchmark that records placement latency/throughput, heartbeat-update
cost, snapshot cost, traced Python memory and assignment fairness. It supports up
to 10,000 synthetic workers and one million placements with explicit hard limits.

The 20,000-placement baseline measured 9,258, 2,164 and 1,060 placements/s at
100, 500 and 1,000 workers. Median placement latency increased from 102.8 to
452.8 to 901.3 microseconds, confirming the v0.7 full-registry scan as an O(N)
bottleneck.

Added an opt-in indexed power-of-two policy without changing v0.7's default
capacity-aware semantics. On the same workloads it measured 42,248, 56,905 and
56,358 placements/s, with 15.2–15.3 microsecond median latency. At 1,000 workers
this is 53.2× baseline throughput and 98.3% lower median decision latency. Every
worker received exactly 20 assignments in the uniform 1,000-worker workload.

This result isolates synchronous registry cost. It does not include HTTP, global
lock contention, network faults or GPU inference. The 1,000-worker snapshot was
about 745 KB and required 36–40 ms, identifying full health serialization as the
next scaling target. Evidence is under `results/v08_scale_baseline/`.

## v0.8.2 operational scalability — 2026-10-20

Replaced full-registry control-plane health responses with bounded aggregate state.
The summary retains readiness, availability, drain/stale counts, in-flight routes,
membership revision, policy and uniform artifact/model identity without embedding
workers or audit records. Added deterministic worker pagination with limits of
1..500. Opaque cursors carry the membership revision; registration or removal
causes HTTP 409 so a client cannot continue an inconsistent scan.

Replaced unbounded heartbeat task fan-out with a configurable fixed worker pool,
queue and separate connect/total timeouts. Probe and inference traffic now use
separate HTTP sessions. Added metrics for heartbeat queue depth, request duration,
sweep duration, success/failure, ready-state transitions and health-handler latency.
A slow-worker integration test confirmed eight endpoints never exceeded the
configured concurrency of two.

At 100/1,000/5,000 synthetic workers the final summary payload stayed 387/391/391
bytes, versus legacy snapshots of 123 KB/745 KB/3.51 MB. Median summary generation
took 0.052/0.490/2.69 ms. Complete paginated scans required 1/2/10 pages and
3.4/33.7/171.1 ms. These are in-process CPU measurements with no network or GPU.
Evidence is under `results/v082_operational_scale/`.

## v0.8.3 real-network synthetic fault cluster — 2026-10-20

Added a bounded GPU-free harness with real loopback HTTP interfaces. The client,
control plane and every synthetic worker run on independently scheduled event
loops. Workers provide health and streaming generation endpoints with seeded
latency/jitter. The scenario hangs and offlines workers, observes failure, restores
new instance IDs, and registers/drains/removes a live churn worker during sustained
traffic. It retains raw request records, Prometheus excerpts, assignment counts,
event-loop lag, traced memory and transition timing.

Early high-concurrency testing found that worker active/queue observations were
being added to control-plane in-flight routes even though they represented the
same work. Changed capacity accounting to the maximum of observed occupancy and
local routes, preserving delayed-heartbeat conservatism without double charging.
Testing also confirmed removal correctly rejects a worker with active routes; the
harness now drains and waits for zero before deletion. Per-worker labels were
removed from placement metrics to prevent fleet-sized Prometheus cardinality.

Three repetitions at 32 workers measured median successful throughput of
1,084/1,019/982 requests/s at concurrency 128/512/1,000. Median error rates were
2.02%/1.50%/3.57%. Median routing P95 was 0.137/2.296/3.753 seconds and P99 was
0.160/4.228/5.233 seconds. Median recovery was 0.282/8.751/6.036 seconds, with
substantial run-to-run variation at high load. More concurrency reduced useful
throughput and sharply worsened tails; the next bottleneck is the single streaming
proxy event loop, not registry selection or memory. Evidence is under
`results/v083_synthetic/`, including `summary.json` and all raw request reports.

## v0.9 — local CUDA worker preparation — 2026-10-20

Implemented the Linux/NVIDIA worker locally, without renting or starting paid
infrastructure. No GPU has executed this code. Performance is not claimed.

- Added `InferenceBackend` as the documented serving contract. `MLXBackend` and the
  new `CUDABackend` both satisfy it. The engine, admission, cancellation, deadlines
  and HTTP streaming are unchanged. Batch execution stays MLX-only.
- `CUDABackend`: Transformers and PyTorch CUDA, greedy decoding with an explicit KV
  cache so each token passes the engine's cancellation checkpoints. EOS is never
  counted as output. Incremental detokenization holds back incomplete UTF-8. A
  per-process allocator cap and bounded OOM handling keep failures per request.
- Telemetry: background `nvidia-smi` sampler with `measured`, `unavailable` and
  `error` states. Unsupported fields are null, never zero.
- Found and fixed a data-plane defect: a client disconnect mid-stream was recorded as
  a worker transport failure, leaving a healthy worker unplaceable until the next
  heartbeat. The regression fails on the v0.8.4 code and passes on v0.9. The
  integrated proxy received the same classification hardening; no test reproduced it.
- Found and fixed a packaging gap: the sdist omitted the shared test fixture.
- Deployment: env-configurable server, locked CUDA requirements (`uv`, torch 2.9.1 with
  CUDA 12.8 wheels), non-root Dockerfile with readiness healthcheck, compose file,
  Runpod session script with `--plan` mode, diagnostics script and an env template.
- Benchmarks: validated config, planning with declared assumptions, runner that
  records server provenance per report, and planned-only experiments that refuse to
  run. The single-GPU suite plans 114 minutes against a 300-minute session.
- Validation: 202 tests passed (154 existing, 48 new). Ruff clean. A real pinned
  checkpoint (`Qwen2.5-0.5B-Instruct@7ae55760`) matched transformers greedy output
  token for token on CPU. SIGTERM during decode exits cleanly. The wheel and sdist
  build, and the sdist tests pass against the installed wheel.
- Not verified: the Dockerfile build (the Docker daemon started, but registry access
  hung), the CI CPU job on GitHub, and every CUDA-dependent behavior. See
  `docs/v09-validation-report.md`.
- The MLX Apple path was not re-run on a model. The change to it is the shared proxy
  fix, a `cache_dir` argument, and environment defaults. The existing MLX-independent
  tests pass unchanged.
