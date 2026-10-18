# TensorPilot — inference service

The service now includes a versioned streaming API, request tracing, bearer-token authentication, readiness/liveness probes, Prometheus metrics, a Python client, and CPU CI configuration. See [service integration and operations](docs/service.md) for usage and limits.

These are service capabilities, not a claim of production readiness. The v0.2 benchmark workflow below remains supported.

# TensorPilot v0.2 — bounded streaming inference

Concurrent HTTP clients feed a bounded FIFO queue. One dedicated thread loads,
warms, tokenizes, and runs the MLX model. GPU execution is serial; this milestone
does **not** implement batching or parallel model execution. The existing v0.1
benchmark and tests are preserved below.

## Run the first controlled experiment

Terminal 1, from the existing project directory:

```bash
cd /Users/adityapatel/Downloads/tensorpilot-v0.1
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python -m tensorpilot.server --queue-size 8 --memory-gib 6
```

Wait for the server startup message after model loading and warmup. Terminal 2:

```bash
cd /Users/adityapatel/Downloads/tensorpilot-v0.1
source .venv/bin/activate
python -m tensorpilot.load --requests 5 --concurrency 1 --max-tokens 32 --output results/v02_c1.json
python -m tensorpilot.load --requests 10 --concurrency 2 --max-tokens 32 --output results/v02_c2.json
```

Run these sequentially, against the same idle server. Repeat each configuration
three times with distinct output filenames before drawing conclusions. `Ctrl-C`
stops the server. No performance claim is supported until these commands run.
Tests use an explicitly synthetic CPU backend; they never load MLX or download a
model. They cover FIFO admission, saturation, streaming, deadlines, cancellation,
slow consumers, failure recovery, shutdown, and an end-to-end workload report.

For a small paced experiment:

```bash
python -m tensorpilot.load --requests 10 --concurrency 2 --rate 2 --max-tokens 32 --output results/v02_rate2.json
```

`--rate` schedules evenly spaced target arrivals. Concurrency caps outstanding
requests; when saturated, arrivals slip rather than spawning unbounded tasks.
Inspect each request's `dispatch_lag_s`. This is not an uncapped open-loop load
generator. Without `--rate`, each worker sends its next request after completion.

## API and architecture

`GET /health` returns readiness, current queue depth, execution activity and model
configuration. `POST /generate` accepts `prompt`, `max_tokens` (default 64), and
`timeout_s` (default 60, maximum 120). The response is newline-delimited JSON with
`token` events followed by one `done` or `error` event. Once HTTP streaming starts,
generation failures appear in the stream; HTTP 200 alone does not mean success.
Admission rejects a full queue with HTTP 503 and Retry-After. Invalid input returns
400; excessive request-body size returns 413.

Each job has a 16-event output buffer. A slow reader eventually blocks its worker,
with a deadline preventing indefinite stalls. Request cancellation is checked
between MLX generation steps and while waiting for buffer space. A currently
executing GPU operation cannot be forcibly interrupted; long prefill can delay
cancellation and shutdown. One process owns one model; do not launch extra workers.

Limits default to eight waiting jobs plus one active job, 32 KiB HTTP bodies,
16 KiB prompt text, 2,048 prompt tokens, and 256 output tokens. Prompt-token
validation happens on the model thread after admission and can return a stream
error. The MLX allocator limit defaults to 6 GiB with at most 512 MiB allocator
cache. These are safeguards, not a bound on total process RSS or macOS unified
memory pressure. Keep initial runs small and watch Activity Monitor for swap.

## Measurement definitions

- Output tokens are counted from MLX token IDs, excluding sampled EOS. EOS is
  counted separately in `sampled_tokens_including_eos`. Empty decoded text can
  still represent an output token. No output re-tokenization is used.
- `queue_wait_s` is admission to worker start. `ttft_s` and `token_times_s` use
  admission as zero and include queue wait, tokenization and prefill. The final
  token's streaming event can include detokenizer finalization overhead.
- Client TTFT measures dispatch to first received non-EOS token event. Client
  inter-token times include HTTP buffering and are not GPU kernel timings.
- Client elapsed time ends at the terminal event or failure. Server elapsed time
  excludes final HTTP delivery. Each clock is monotonic; clocks are never
  subtracted across machines.
- Throughput uses completed output tokens divided by the full workload wall
  time, including pacing and failed requests. Failed partial output is recorded
  separately. Latency percentiles describe successful requests; error and rejection
  rates must be read alongside them. Percentiles use linear interpolation.
- MLX active memory is sampled after generation; peak memory is the process
  lifetime MLX allocator peak, including model loading and warmup. Neither is
  per-request GPU memory or total system memory pressure.

Reports include all request records, prompts, request configuration, timing arrays,
server model/configuration and runtime versions, and client environment. Prompts
are raw text with tokenizer-default special tokens, using greedy sampling. The
model repository revision is not pinned yet, and the dependency ranges are not a
lockfile; archive the exact environment/model snapshot for publication. This is
an initial experiment format, not a claim of fully reproducible research results.

## Files

- `backend.py`: MLX streaming/token adapter and allocator limits.
- `engine.py`: FIFO admission, dedicated execution thread and bounded event buffers.
- `server.py`: local HTTP service, validation and disconnect handling.
- `load.py`: bounded concurrent/paced workload runner and raw JSON reports.
- `tests/test_serving.py`: CPU-only functional and HTTP integration tests.

The original sequential baseline follows. Its approximate token throughput is
not directly comparable to the v0.2 exact output-token measurement.

---

# TensorPilot v0.1 — Real local ML inference baseline

**Goal:** Execute an actual 4-bit transformer on the Apple GPU via MLX, benchmark sequential requests, and save reproducible results. This is a first measurement milestone, not a GPU cluster or inference scheduler yet.

## Requirements
- Apple Silicon Mac, 16 GB unified RAM recommended
- Python 3.11 or 3.12, arm64-native interpreter
- 3+ GB free disk for Python environment, model cache, and outputs (actual use varies)
- Internet for the initial model download; inference runs locally afterward

## Quick start
```bash
cd tensorpilot-v0.1
python3 --version
python3 -c 'import platform; print(platform.machine())'  # should be arm64
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
pytest -q
python -m tensorpilot.bench --requests 5 --max-tokens 64 --output results/baseline.json
```

The default open model is `mlx-community/Qwen2.5-0.5B-Instruct-4bit`, downloaded from Hugging Face on first run. If model download/access fails, see the model's Hugging Face page and choose an accessible MLX-compatible model using `--model`.

## What measurements mean
- `elapsed_s`: elapsed wall-clock time of the *entire* `generate()` call (not TTFT).
- `generated_tokens`: estimated by re-tokenizing returned text, not the generator's exact token count.
- `estimated_tokens_per_s`: estimated output tokens divided by end-to-end generation time, **including prompt processing**.
- `p95_request_latency_s`: 95th percentile of successful request latencies; small samples are not statistically reliable.
- `aggregate_output_tokens_per_s`: total approximate output tokens divided by sum of elapsed times; requests are **sequential**, not concurrent.

### Benchmark integrity
Do not publish these numbers as cross-hardware GPU benchmarks, model-only decode throughput, precise TTFT, or representative concurrency results. Track hardware, macOS version, model name and revision, token lengths, background load, and warmup. Next milestone: accurate token counting and first-token timing using supported streaming generation APIs.

## Next planned features
1. Streaming metrics: TTFT, exact generated token counts, prefill and decode separation.
2. Scheduler experiments with an explicit FIFO baseline and fixed test workloads.
3. GPU memory profiling, inference cache investigations.
4. Cluster health and Kubernetes simulations, with real vs simulated metrics clearly separated.
5. MLOps regression gates, model versioning, and reproducible reports.

## Troubleshooting
- `platform.machine()` must print `arm64`. If you see `x86_64`, switch to native Apple Silicon Python.
- With 16 GB RAM, close unnecessary Chrome tabs and apps before benchmarking; swapping can distort results.
- Run from the project folder so `python -m tensorpilot.bench` finds the package.
- A one-time model download requires Internet and free storage.
