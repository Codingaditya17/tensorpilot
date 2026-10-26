# TensorPilot

**GPU inference serving with bounded admission, observable scheduling, and
reproducible workload experiments.**

TensorPilot runs a local transformer through MLX on Apple Silicon. It implements
its own queue/scheduling lifecycle and measures the effects of concurrency,
overload, deadlines and workload composition. The current runtime has one model
owner thread. Serial execution is the default; opt-in continuous batching advances
multiple independent sequences through that thread.

## What works

- Versioned streaming HTTP API and async Python integration client.
- Bounded request admission and bounded response buffers, with explicit overload.
- FIFO baseline and shortest-output-budget scheduling with an aging override.
- Immediate cancellation/expiry of waiting work, active cancellation at generation
  boundaries, backend cleanup checks and worker readiness supervision.
- Request IDs, safe public errors, bearer authentication, probes and Prometheus.
- Seeded mixed-length workloads, exact streamed token counting, queue/TTFT/token
  timings, per-class reports and repeated policy comparisons.
- Immutable model snapshot/content fingerprints, runtime/source metadata, source
  archives and experiment manifests.
- Installable Python package, command-line tools, CPU test suite and CI config.
- Browser dashboard at `/` with live readiness, queue, device and streaming inference.
- Conservative KV-cache reservation admission, separate from measured MLX memory.
- Guarded 15/30/60-minute soaks, exact token gates and worker restart supervision.
- Synchronized prefill/decode telemetry, adaptive batch width, workload-matrix
  preflight, throughput/latency gates and bounded Apple Metal kernel capture.
- A local multi-worker control plane with health-aware placement, draining,
  artifact constraints, lifecycle audit history and supervised worker recovery.
- Constant-size cluster health, revision-safe worker pagination, bounded heartbeat
  concurrency and GPU-free real-network fault/saturation experiments.

## v1.0 GPU scale evaluation (implemented, **not yet run on a GPU**)

- **CUDA continuous batching:** a batched forward pass over a fixed KV slot pool for Qwen2 and Llama models, behind the existing batch engine
  (`--backend cuda --execution batch`). Not paged, no prefix cache, no CUDA graphs. The serial worker remains for A/B tests.
- **Explicit NF4 quantization** (`--quantization nf4`, bitsandbytes) so Qwen2.5-14B can be planned for a 24 GB card. It never falls back to another
  precision. Lower memory does not imply higher speed, and 14B still cannot hold 128 concurrent sequences at any tested context length.
- **Evaluation harness:** seeded workloads, Poisson and bursty arrivals, cancellation, timeouts, SLO goodput, continuous GPU and server telemetry,
  a batch-invariance check, a worker-crash audit, a vLLM comparison client and a hash-locked vLLM environment, a budget-ledgered stage runner, and a
  report generator that invents nothing.
- Verified on CPU and inside the Linux image; **nothing has run on a GPU**, so there are no v1.0 performance numbers. The only GPU evidence is the
  immutable v0.9.1-4 baseline (`benchmarks/baselines/`). Start with [`docs/v10-evaluation-plan.md`](docs/v10-evaluation-plan.md); the design is in
  [`docs/v10-architecture.md`](docs/v10-architecture.md) and the Pod procedure in [`docs/v10-runbook.md`](docs/v10-runbook.md). The production-scale plan, its automated acceptance checks and the analysis of the
  quick-session evidence are in [`docs/v10-production-evaluation.md`](docs/v10-production-evaluation.md).

Real MLX inference and CPU behavior tests are separate evidence. A scheduling
policy is experimental until measured against a baseline; improvements to one
latency percentile can worsen another. FIFO remains the default.

## Install

Use native Apple Silicon Python 3.11 or newer for real inference. The existing
environment has been validated on Python 3.14.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ruff check tensorpilot tests
python -m pytest -q
```

For CPU-only development/CI, install `requirements-service.txt` instead. It does
not install MLX or download a model. Inference requires macOS/Metal; a Linux
container is not a substitute for the local GPU backend.

## Linux/NVIDIA CUDA worker (v0.9, unmeasured on GPU)

v0.9 adds an optional CUDA worker: PyTorch and Hugging Face Transformers, greedy
decoding with an explicit KV cache, the same streaming API, and the same control and
data planes. It is prepared and tested locally on CPU, and it has not run on an NVIDIA
GPU yet. No CUDA performance is claimed.

```bash
uv pip install -r requirements-cuda.txt            # Linux x86_64 lock (Python 3.11)
TENSORPILOT_API_KEY=... python -m tensorpilot.server --backend cuda \
  --model Qwen/Qwen2.5-0.5B-Instruct --revision 7ae557604adf67be50417f59c2c2f167def9a775
python -m tensorpilot.gpu_preflight --model Qwen/Qwen2.5-0.5B-Instruct \
  --revision 7ae557604adf67be50417f59c2c2f167def9a775 --output results/v09_gpu/preflight.json
```

The [architecture and decisions](docs/v09-cuda-runtime.md), the
[Runpod session runbook](docs/v09-runpod-runbook.md) and the
[local validation report](docs/v09-validation-report.md) describe what is verified
and what requires a GPU.

## Serve and call

```bash
tensorpilot-server --queue-size 8 --memory-gib 6
```

Enable measured, experimental continuous batching with
`tensorpilot-server --execution batch --batch-size 2 --offline`.
Each sequence has independent output budgets, detokenization, cancellation and
deadlines. A full response buffer sheds that sequence so other clients can
continue. See [batching design and results](docs/batching-design.md).

Enable the v0.6 deterministic adaptive controller with
`tensorpilot-server --execution adaptive --batch-size 4 --offline`. It changes
the target width from queue demand, oldest wait and estimated reservation pressure.
Inspect its live explanation in `/health` and see
[GPU runtime intelligence](docs/v06-gpu-runtime.md) for measurements and limits.

Open `http://127.0.0.1:8080/` for the local dashboard. For automatic recovery,
run `tensorpilot-supervisor --offline`; it owns the worker process, requires
readiness after every start, applies a bounded restart/backoff policy and records
lifecycle JSONL under `results/`.

Run a reproducible two-worker local cluster with:

```bash
tensorpilot-cluster --workers 2 --offline --output-dir results/local-cluster
```

The cluster control API binds to `http://127.0.0.1:8090`; worker dashboards bind
to ports 8101 and 8102. `POST /v1/generate` uses the existing NDJSON protocol.
`GET /health` exposes placement and lifecycle history, and `GET /metrics` exposes
cluster metrics. Workers can be drained through
`POST /v1/workers/{worker_id}/drain`. See the
[v0.7 control-plane design](docs/v07-cluster-control-plane.md) for the API,
failure semantics, authentication boundaries and measured results.

For bounded distributed-systems testing without loading MLX, run
`tensorpilot-scale-bench` for 100–10,000 registry workers and
`tensorpilot-synthetic-cluster` for independently scheduled loopback workers,
network delay, heartbeat timeouts, worker replacement and membership churn. The
[v0.8 plan and results](docs/v08-distributed-scale-plan.md) distinguish registry
microbenchmarks from real HTTP tests and from unproven physical-node scaling.

Resource admission defaults to a 512 MiB conservative KV estimate budget. Tune
it with `--admission-memory-mib` and `--admission-bytes-per-token`. These are
overload-control reservations, not MLX allocator measurements.

Run a guarded soak with `tensorpilot-soak --minutes 15 --concurrency 2
--prompt-repeats 32 --output-tokens 64 --max-mlx-active-mib 4096`. Presets also
support 30 and 60 minutes, cap total requests, poll readiness and measured MLX
active bytes, and preserve a recoverable journal on safety abort.

On this Mac's Apple M5, three matched 30-request runs per mode measured 56.2%
higher median throughput and 37.7% lower median P95 latency with batching at width
two. All 180 measured requests succeeded. These results apply to the recorded
model/workload; serial remains the default.

The default model is `mlx-community/Qwen2.5-0.5B-Instruct-4bit`. First use downloads
the model; subsequent cached runs can use `--offline`. Pin a specific snapshot
with `--revision COMMIT_SHA`. Initialization and warmup finish before binding the
HTTP listener. Defaults are local-only at `http://127.0.0.1:8080`.

In another terminal in the same environment:

```bash
tensorpilot-load --requests 10 --concurrency 2 --max-tokens 32 --output results/smoke.json
```

`POST /v1/generate` streams newline-delimited JSON. `/v1/model` describes the
model and limits; `/livez`, `/readyz`, and `/metrics` support operations. Configure
`TENSORPILOT_API_KEY` to require a shared bearer token for non-probe endpoints.
The load runner reads it from the environment. The Python client accepts it as
`Client(api_key=...)`. See [service integration](docs/service.md) for examples,
error semantics and authentication boundaries.

## Compare scheduler policies

Stop the serving process first. This command refuses an occupied port and owns
one model-server child at a time:

```bash
tensorpilot-experiment --requests 30 --concurrency 6 --repeat 3 --offline
```

It counterbalances policy order, warms each request class, pins the resolved
model snapshot, and saves raw request reports, source/environment snapshots,
server logs, a manifest and per-class comparison. Every policy sees the same
seeded request plan. A concurrency-capped closed-loop workload does not guarantee
identical wall-clock arrival times across policies.

To explore a different fairness threshold, use `--max-queue-wait 0.25`. This is
an aging threshold at dispatch boundaries, **not a latency guarantee**. See
[scheduling design](docs/scheduling.md).

An explicit regression gate can reject a slower candidate:

```bash
tensorpilot-compare results/EXPERIMENT/repeat*_fifo.json \
  results/EXPERIMENT/repeat*_shortest-budget.json \
  --max-p95-regression 0.05 --output results/gate.json
```

Select only measured reports, excluding `*.warmup.json`. The comparator refuses
mismatched workloads, model artifacts, environment/configuration, or unbalanced
repetitions. A failed gate exits with status 1. Small sample sizes still limit
conclusions; a passing gate is not proof of general performance superiority.
Use `--max-throughput-regression 0.05` to gate throughput as well as latency and
availability. `tensorpilot-matrix` emits preflighted v0.6 configurations, while
`tensorpilot-kernel-capture` records a bounded Apple `.gputrace` for Instruments.

## Architecture and engineering record

See [architecture](ARCHITECTURE.md), [work log](WORK_LOG.md),
[current session state](SESSION_STATE.md), and [roadmap](ROADMAP.md).
The [v0.8.4 control/data-plane study](docs/v084-control-data-separation.md)
records the measured reliability gains and throughput regression from separation.
See [soak testing and recovery](docs/benchmarking.md) and
[cache profiling](docs/memory-profiling.md) for deeper experiments.
The [first scheduling study](docs/experiments/scheduler_2026-10-17.md) records
540 real requests and the measured median/tail-latency tradeoff.
The historical sequential benchmark is preserved as `tensorpilot.bench`;
[v0.2 notes](docs/baseline-v02.md) explain its measurement limitations.

## Current boundaries

This release includes a multi-process control plane. All measured MLX workers run
on one Mac and share one Apple GPU. The CUDA worker is unmeasured on GPU hardware. It does not implement distributed tensor or
pipeline execution, per-tenant isolation, a durable request queue or transparent
in-flight replay. Supervisors restart failed worker processes after downtime.
Allocator limits and context caps do not measure all macOS memory pressure, and a
stuck GPU operation can delay cancellation and shutdown. Remote deployment needs
TLS, durable state, service discovery and stronger identity controls.
