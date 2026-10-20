# v0.9 CUDA worker: architecture and decisions

This document explains what v0.9 adds, why each decision was made, and what is
verified on which hardware. It describes an implementation prepared on an Apple
M5 MacBook Pro. **No NVIDIA GPU has run this code yet.** Performance is not claimed
anywhere in this release. The measured CUDA results come from the GPU session in
[`v09-runpod-runbook.md`](v09-runpod-runbook.md).

## Goal and non-goals

Goal: a TensorPilot worker that loads a real open-weight Transformers checkpoint on
a Linux NVIDIA host, serves the same streaming API as the MLX worker, and plugs into
the existing control plane and data plane without changing their contracts.

Non-goals for v0.9: continuous batching on CUDA, multi-GPU scheduling, quantized
CUDA weights, chat templates, and any performance claim made before real hardware
runs.

## Architecture

```
client ──► data plane :8091 ──► control plane :8090 (placement + route lease)
                │                         │
                └──── stream ────────────►│ worker registry (health, drain, compatibility)
                                          ▼
                        CUDA worker :8080  (tensorpilot.server --backend cuda)
                        ├─ HTTP handlers, auth, request IDs, metrics
                        ├─ Engine: bounded queue, admission, deadlines, cancellation
                        ├─ model thread (single): CUDABackend
                        │     PyTorch CUDA + Transformers, greedy, explicit KV cache
                        └─ GPUTelemetry: background nvidia-smi sampler (never per request)
```

The worker is the only process that touches the GPU. The control plane and data
plane are CPU-only and unchanged in their contracts.

## Decision 1: one backend contract, two runtimes

`tensorpilot/backend.py` now documents `InferenceBackend`, the members the serial
engine reads: `backend_name`, `provenance`, `max_prompt_tokens`, `stream()` and
`memory()`. `MLXBackend` and `CUDABackend` both satisfy it. The engine,
admission, cancellation, deadline and HTTP streaming code are unchanged.

The batch engine stays MLX-only. It drives MLX's `BatchGenerator` and reads
`backend.model` and `backend.tokenizer` directly. Forcing CUDA through it would
mean reimplementing the batch scheduler, so the server rejects
`--backend cuda --execution batch` with an explicit error.

## Decision 2: a manual greedy loop instead of `generate()`

`CUDABackend.stream()` runs the decode loop itself: one forward pass per token,
with `past_key_values` carried between steps. Each token is yielded to the engine,
which checks cancellation and deadlines between tokens.

A library `generate()` call would run the whole sequence inside one call. A client
that disconnects would then keep the GPU busy until the sequence finished. The
explicit loop lets the engine stop between any two tokens, and closing the
generator releases the cache immediately.

Correctness is checked, not assumed. `tests/test_cuda_backend.py` compares the
streamed token IDs against `model.generate(do_sample=False)`, and the same check
passed on the real `Qwen2.5-0.5B-Instruct` checkpoint at its pinned revision
(12 of 12 tokens, on CPU).

Decoding is greedy, matching the MLX backend's temperature-zero sampler, so the two
runtimes are comparable on the same prompts.

## Decision 3: accounting that matches the protocol

- EOS is yielded with `output=False` and is never counted as generated output.
  `generated_tokens` in the terminal event therefore equals the number of output
  events the client receives, as the protocol validator requires.
- `finish_reason` is `stop` for EOS and `length` when the budget is reached.
- Text is decoded incrementally. An incomplete UTF-8 sequence (`�`) is held
  back until its remaining bytes arrive, so no replacement characters are streamed.
- Prompts longer than `--max-prompt-tokens` are rejected before inference as
  `invalid_prompt`.

## Decision 4: memory safeguards that fail per request

- `torch.cuda.set_per_process_memory_fraction(--cuda-memory-fraction)` caps the
  caching allocator at 90% of the device by default. A runaway allocation then
  raises an out-of-memory error inside this process rather than starving other
  processes on the GPU.
- An out-of-memory error fails only the request that caused it. The backend calls
  `torch.cuda.empty_cache()`, increments an OOM counter exposed as
  `tensorpilot_cuda_oom_events`, and keeps serving.
- Admission remains a deterministic KV-cache *estimate*, labeled as such. It is not
  allocator telemetry, and the documentation never presents it as measured memory.
- Serial execution means one sequence occupies the device at a time, so memory
  pressure from concurrent sequences cannot compound.

## Decision 5: telemetry is sampled, labeled, and never invented

`tensorpilot/gpu_telemetry.py` reads `nvidia-smi --query-gpu` in a background task.
The health endpoint returns the latest sample and never runs the binary per request.
Each sample carries a status:

| status | meaning |
|---|---|
| `measured` | `nvidia-smi` returned devices; values are the driver's reading |
| `unavailable` | the binary is not on this host (for example, a Mac) |
| `error` | the binary failed, timed out or returned an unexpected shape |

Fields the driver reports as `[N/A]` are null, never zero. The sampler is
`nvidia-smi` rather than a Python binding so it adds no dependency and works under
the NVIDIA container runtime, where the host injects the binary.

Device identity and the CUDA stack come from PyTorch and are recorded in the
provenance block: device name, compute capability, total memory, CUDA runtime,
cuDNN, PyTorch and Transformers versions, the resolved model revision and the
artifact SHA-256. Memory comes from the backend sample: allocated, peak allocated,
reserved, and peak reserved bytes. On CPU these fields are null.

Latency, throughput, TTFT, inter-token latency and P50/P95/P99 are client-side
measurements from `tensorpilot.load`. Each report includes the server health
snapshot taken before and after the run, so the GPU identity, driver and telemetry
are recorded alongside the numbers.

## Decision 6: identity, drain and restart stay with the control plane

A worker advertises its model, artifact hash, instance ID and capacity through
`/health`. The control plane already enforces model and artifact compatibility,
drain, removal and instance-change tracking. The CUDA worker needed no new routing
logic. The integration tests exercise those guarantees over real loopback HTTP
with the CUDA backend class (on CPU).

Two proxy defects surfaced during this work and are fixed:

- **Data plane (confirmed).** A client that disconnected mid-stream was recorded as a
  worker transport failure, because the route's default outcome was
  `worker_transport_error`. The control plane then left a healthy worker unplaceable
  until its next heartbeat. The outcome now defaults to `client_disconnected`, and
  only real upstream failures are classified as worker failures. A regression test
  fails on the v0.8.4 code and passes on v0.9.
- **Integrated proxy (latent).** A `ConnectionError` while writing to a departed
  client was classified with worker failures. It now releases the route without
  penalizing the worker. This path was not reproduced by a test, so it is a hardening
  change rather than a confirmed bug.

## Decision 7: deployment without source edits

Every operational setting has an environment default, so a deployment sets
variables rather than editing code:

| variable | flag | default |
|---|---|---|
| `TENSORPILOT_BACKEND` | `--backend` | `mlx` |
| `TENSORPILOT_MODEL` | `--model` | the MLX default (CUDA requires an explicit model) |
| `TENSORPILOT_REVISION` | `--revision` | unset; a 40-character commit SHA is strongly recommended |
| `TENSORPILOT_CACHE_DIR` | `--cache-dir` | the Hugging Face default cache |
| `TENSORPILOT_HOST` / `TENSORPILOT_PORT` | `--host` / `--port` | `127.0.0.1` / `8080` |
| `TENSORPILOT_DEVICE` | `--device` | `cuda` |
| `TENSORPILOT_API_KEY` | (environment only) | unset; required for any non-loopback bind |
| `HF_TOKEN` | read by `huggingface_hub` | unset; for gated models only, never committed |

CUDA-specific flags: `--dtype {float16,bfloat16,float32}` (default `float16`),
`--cuda-memory-fraction` (default `0.9`), `--max-prompt-tokens` (default `2048`),
`--gpu-telemetry-interval` (default `2` seconds; `0` disables).

The server refuses unsafe combinations at startup: CUDA with the MLX default model,
CUDA with batch execution, CUDA with MLX-only memory profiling, and any
non-loopback bind without an API key.

## Decision 8: pinned, reproducible dependencies

`requirements-cuda.in` lists the direct dependencies. `requirements-cuda.txt` is the
full lock for Linux x86_64 and Python 3.11, produced with `uv`:

```
uv pip compile requirements-cuda.in --python-platform x86_64-unknown-linux-gnu --python-version 3.11 -o requirements-cuda.txt
```

The lock pins `torch==2.9.1`, which resolves to CUDA 12.8 runtime wheels from PyPI
(`nvidia-cuda-runtime-cu12==12.8.90`, `nvidia-cudnn-cu12==9.10.2.21`), and
`transformers==5.20.1`. Local CPU tests use the same Transformers and PyTorch
versions, so the code that is tested is the code that is deployed.

## Decision 9: the image is non-root, probed, and stops cleanly

`deploy/docker/Dockerfile.cuda` runs as UID 10001, keeps the Hugging Face cache on a
volume at `/data/huggingface`, and exposes `/readyz` as its healthcheck. Readiness
means the model is loaded and warmed, not merely that the process is alive. The
server handles SIGTERM by cancelling the active decode, failing queued requests
with `shutting_down` and releasing the model. A test sends SIGTERM to a live
worker mid-decode and asserts a clean exit and an explicit terminal event
(`tests/test_graceful_shutdown.py`).

The image's base is `python:3.11-slim-bookworm`. The CUDA libraries come from the
pinned PyTorch wheels, so no CUDA toolkit is needed. The host must provide an
NVIDIA driver through the NVIDIA Container Toolkit. See the driver assumptions
below.

## Driver and runtime assumptions

These are assumptions to confirm on the Pod, not verified facts:

- The host NVIDIA driver must support CUDA 12.8 (the R570 driver branch, 570.26 or
  newer, per NVIDIA's CUDA 12.8 release notes). `nvidia-smi` on the host reports the
  driver version, and `tensorpilot.gpu_preflight` confirms that PyTorch can
  initialize CUDA.
- The container needs `NVIDIA_VISIBLE_DEVICES=all` and
  `NVIDIA_DRIVER_CAPABILITIES=compute,utility`, both set in the image.
- The device needs compute capability 7.0 or newer for float16 and 8.0 or newer for
  bfloat16. The preflight records the capability, and the operator chooses the dtype.
- A 16 GiB device is the preflight default minimum. The 0.5B reference model needs far
  less. The threshold is conservative and deliberately adjustable.

## What is verified where

| claim | evidence | hardware |
|---|---|---|
| Streamed tokens equal transformers greedy decoding | `test_cuda_backend.py` (tiny model) and the real checkpoint on CPU | CPU |
| Real pinned checkpoint loads and matches the reference | local run on `Qwen2.5-0.5B-Instruct@7ae55760` | CPU |
| Cancellation releases the reservation and the worker recovers | `test_cuda_backend.py`, engine on the real decode loop | CPU |
| OOM fails one request, is counted and recovers | `test_cuda_backend.py` with an injected fault | CPU |
| Placement, compatibility, streaming, cancellation, drain, removal, restart identity, mid-stream fault with no replay, connection failure | `test_cuda_cluster.py` over loopback HTTP | CPU backend class |
| SIGTERM drains a live decode and exits 0 | `test_graceful_shutdown.py` | CPU |
| nvidia-smi parsing, unavailable and error states | `test_gpu_telemetry.py` | none |
| Preflight decisions and the Mac's honest not-ready result | `test_gpu_preflight.py`; `gpu_preflight` run on the Mac | none |
| Benchmark config validation, planning, evidence layout | `test_gpu_bench.py` | CPU worker |
| Compose file is valid | `docker compose config` | none |
| **CUDA kernels, VRAM behavior, throughput, latency, multi-request GPU behavior** | **not yet measured** | **NVIDIA GPU required** |

## Known limitations

- Serial execution only. Concurrency above one queues requests rather than batching
  them, so concurrent-client results measure queueing, not throughput gains.
- Greedy decoding only, with no sampling parameters and no chat template. This matches
  the MLX baseline and keeps outputs comparable.
- Memory and utilization values are process-lifetime allocator counters and
  driver samples. They are not an audit of total device pressure.
- Two-GPU and batch-width experiments are listed as planned and refuse to run. They
  are not claimed.
- The Dockerfile has not been built on this machine. The Docker daemon started, but
  registry access hung, so the image is validated only by compose-config and review.
  Build it on a host with registry access before the Pod run.
