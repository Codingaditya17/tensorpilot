# Current engineering state

TensorPilot v0.5.0 runtime hardening is active on branch
`codex/runtime-platform`. A supervised serial/FIFO service exposes the dashboard
at `http://127.0.0.1:8080/`; batching remains opt-in.

Implemented: browser dashboard, conservative KV reservation admission, separate
estimated/actual memory telemetry, guarded soak presets, configurable stress
workloads, pinned token-ID correctness, and readiness-gated bounded worker
restart. The last complete suite passed 104 tests with Ruff clean.

Real evidence: serial/batch IDs matched for three pinned cases; constrained
admission accepted one and rejected five without leaks; a SIGKILLed worker was
restarted and served recovery traffic. The final in-flight crash produced an
explicit transport error, a new ready instance and a successful next request.
The 15-minute soak completed 3,650/3,650 requests with all 180 samples ready.
See docs/v05-runtime-hardening.md and WORK_LOG.md.

Nothing is pushed or published. Prior release/evidence archives remain intact.
The v0.5 wheel passed 104 tests outside the checkout without MLX. Remaining:
record final checksums, package evidence and create a local Git checkpoint.
