# Current engineering state

TensorPilot v0.7.0 implements a local multi-worker GPU inference control plane on
branch `codex/runtime-platform`. It adds health-aware registration, capacity-aware
routing, drain and stale-worker exclusion, model/artifact placement constraints,
bounded audit history, cluster metrics and a launcher that owns one restart
supervisor per worker.

Source validation passes 141 tests with Ruff clean before final package validation.
Real MLX evidence includes balanced routing across two workers, complete drain to
one worker, a worker replacement with a new instance ID and an in-flight SIGKILL.
The interrupted stream received `worker_transport_error`, was not replayed and a
new request succeeded through the remaining worker.

Three matched repetitions completed 180/180 requests. Compared with direct
single-process service, two worker processes sharing one Apple GPU measured +3.0%
median throughput, while P95 latency worsened 34.0% and P95 inter-token latency
worsened 76.7%. This is not evidence of independent-node scaling. See
`docs/v07-cluster-control-plane.md` and `WORK_LOG.md`.

Nothing is pushed or published. Release artifacts are copied outside the repository
only after exact-wheel validation.
