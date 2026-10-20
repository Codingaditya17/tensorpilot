# v0.9 Runpod GPU session runbook

Use this runbook for one bounded, five-hour GPU session. Nothing here has been run
on a GPU yet. Each step has a check, so a failure stops the session early instead
of burning rental time. Do not rent or start a Pod until the pre-rental checklist
passes locally.

## 0. Pre-rental checklist (local, no GPU)

- [ ] `git status` is clean on the release commit, and the wheel and sdist are built:
      `uv build --out-dir dist`.
- [ ] Tests pass: `python -m pytest -q` and `python -m ruff check tensorpilot tests`.
- [ ] The model revision is pinned in `benchmarks/gpu/v09-single-gpu.json` and in
      `TENSORPILOT_REVISION`. It must be a 40-character commit SHA. The current pin is
      `7ae557604adf67be50417f59c2c2f167def9a775` for `Qwen/Qwen2.5-0.5B-Instruct`.
- [ ] `TENSORPILOT_API_KEY` is generated and stored in the Runpod secret store, not in
      any file. Rotate it after the session.
- [ ] Budget reviewed: `python -m tensorpilot.gpu_bench plan benchmarks/gpu/v09-single-gpu.json`
      must report `within_budget=true`.
- [ ] The Docker image, if used, has been built and pushed from a host with registry
      access. The local Docker daemon could not pull images during this preparation.

## 1. Pod selection (cost decision is yours)

- One NVIDIA GPU with at least 16 GiB of memory. The reference model needs much less;
  the preflight minimum is 16 GiB.
- Compute capability 8.0 or newer is recommended, so bfloat16 is available as well as
  float16.
- Host driver that supports CUDA 12.8 (R570 branch, 570.26 or newer). Confirm with
  `nvidia-smi` on the Pod before anything else.
- Expose the worker port only if you need it. Prefer tunnelling or running the
  benchmark on the Pod itself, because the benchmark drives `127.0.0.1`.
- Storage: 20 GiB of container disk is ample for the reference model and the cache.
- Pricing, GPU availability and billing rules change. Check the Runpod console at the
  time you rent. Nothing in this repository records a price.

## 2. Pod start and environment

On the Pod, from the repository root:

```bash
export TENSORPILOT_MODEL=Qwen/Qwen2.5-0.5B-Instruct
export TENSORPILOT_REVISION=7ae557604adf67be50417f59c2c2f167def9a775
export TENSORPILOT_CACHE_DIR=/workspace/huggingface
export TENSORPILOT_RESULTS=/workspace/results/v09-$(date -u +%Y%m%dT%H%M%SZ)
# TENSORPILOT_API_KEY comes from the Runpod secret store.

nvidia-smi
python3 -m pip install -r requirements-cuda.txt
python3 -m pip install --no-deps -e .
deploy/runpod/session.sh --plan
```

`session.sh --plan` validates the environment and prints the steps without starting
anything. A missing variable or a non-40-character revision stops it here.

Alternative: run the built image with `docker run --init --gpus all -e TENSORPILOT_API_KEY ...`
(see `deploy/docker/compose.cuda.yml`). The bare-metal path above is the one the
session script drives end to end.

## 3. Run the session

```bash
deploy/runpod/session.sh
```

The script runs seven steps and stops at the first failure:

| step | what it proves | artifact |
|---|---|---|
| 1 diagnostics | host, driver, CUDA runtime, package versions | `diagnostics/` |
| 2 preflight | nvidia-smi sees the device, PyTorch initializes CUDA, VRAM meets the minimum, the revision is pinned, and a smoke inference produces tokens | `preflight.json` (exit 1 unless ready) |
| 3 worker | the model loads, warms up and reports ready on `/readyz` | `worker.log` |
| 4 gateway | control plane registers the worker; data plane streams one request end to end | `control.log`, `data.log`, `smoke.ndjson` |
| 5 benchmark | the bounded experiment suite runs against the worker | `plan.txt`, `bench/` |
| 6 evidence | the results directory is archived with a checksum | `<results>.tar.gz`, `.sha256` |
| 7 teardown | every started process is stopped with SIGTERM | exit status and a reminder |

A smoke failure, a preflight failure or a benchmark timeout all end the run. The
`timeout` cap on step 5 is `TENSORPILOT_BENCH_TIMEOUT` (default `3h`).

## 4. Estimated GPU time per experiment

These estimates come from `python -m tensorpilot.gpu_bench plan`, using the
placeholder decode speed (60 tokens/s) and prefill speed (2000 tokens/s) in the config.
They are **not measurements**. Replace the assumptions with the preflight numbers
before relying on the budget.

| experiment | kind | estimated GPU minutes |
|---|---|---:|
| single-sequential-baseline | closed loop, concurrency 1 | 5.5 |
| concurrency-queueing-c4 | closed loop, concurrency 4 | 5.5 |
| concurrency-queueing-c8 | closed loop, concurrency 8 | 5.5 |
| length-short-prompt-short-output | length matrix | 1.8 |
| length-medium-prompt-medium-output | length matrix | 3.3 |
| length-long-prompt-long-output | length matrix | 4.2 |
| arrival-rate-ladder | open loop, 0.5 to 4 req/s | 18.5 |
| saturation-backpressure | open loop, 8 and 16 req/s | 9.7 |
| recovery-supervised-kill | manual, 15 minutes | 15.0 (manual) |
| batch-width-planned | planned, not runnable | 0 |
| **experiments total** | | **69.0** |
| session overhead (preflight, load, warmup, evidence, teardown) | | **45** |
| **session estimate** | | **114** |

Five hours is 300 minutes, so the planned session uses 114 minutes and leaves 186
minutes for debugging and retries. If the measured decode speed is well below 60
tokens/s, the arrival-rate ladder and saturation runs grow in proportion. Regenerate
the plan with the measured numbers before starting step 5.

## 5. Manual recovery experiment

This experiment is listed as `recovery-supervised-kill` because it needs a person to
kill a process. Run it after step 5 finishes. It uses its own supervised worker on
port 8180, so it never collides with the session worker on 8080.

```bash
export TENSORPILOT_BACKEND=cuda TENSORPILOT_MODEL="$TENSORPILOT_MODEL" TENSORPILOT_REVISION="$TENSORPILOT_REVISION"
python3 -m tensorpilot.supervisor --url http://127.0.0.1:8180 --port 8180 --execution serial \
  --max-restarts 3 --lifecycle-log "$TENSORPILOT_RESULTS/recovery.lifecycle.jsonl" \
  --worker-log "$TENSORPILOT_RESULTS/recovery.worker.log" &
```

The supervisor starts `tensorpilot.server` as a child and inherits the `TENSORPILOT_*`
environment, so the backend and model are set by the variables above. Then:

1. Read the worker PID from the lifecycle log, not from `pgrep`, so no other process
   can match:
   `grep worker_ready "$TENSORPILOT_RESULTS/recovery.lifecycle.jsonl" | tail -1`.
2. Start a closed-loop load against the supervised worker:
   `python3 -m tensorpilot.load --url http://127.0.0.1:8180 --requests 40 --concurrency 2 --max-tokens 128 --output "$TENSORPILOT_RESULTS/recovery-load.json"`.
3. While a stream is active, send SIGKILL to that PID with `kill -9 <pid>`. Record
   the terminal event that the affected client receives. It must be an error with a
   terminal code, and no tokens may be replayed.
4. Wait for the lifecycle log to show `worker_restart_scheduled`, then `worker_ready`
   with a new PID.
5. Run a short closed-loop load and confirm successful completion.
6. Stop the supervisor with Ctrl-C or `kill -INT <supervisor pid>`.

The pass condition is: exactly one terminal event per request, no duplicated tokens,
and recovery to readiness within the supervisor's ready timeout. Keep the lifecycle
log and the load report together in the evidence directory.

## 6. Export the evidence

The session archives `$TENSORPILOT_RESULTS` automatically. To download it:

```bash
ls -lh "$TENSORPILOT_RESULTS.tar.gz" "$TENSORPILOT_RESULTS.tar.gz.sha256"
sha256sum -c "$TENSORPILOT_RESULTS.tar.gz.sha256"
```

Download the archive and its checksum file with your usual transfer tool, and verify
the checksum locally before you rely on it. Each benchmark report carries the model
revision, the artifact hash, the GPU identity and driver from the server snapshot, the
software versions, the workload hash, the raw records, and the limitations list.

## 7. Teardown and billable-resource checklist

`session.sh` stops the processes it started. Stopping processes does not stop
billing. Work through this list before you walk away:

- [ ] Confirm `session.sh` printed its teardown message and no `tensorpilot` process
      is still running: `pgrep -fa tensorpilot` returns nothing.
- [ ] Confirm the evidence archive and its `.sha256` file are downloaded and the
      checksum verifies on your machine.
- [ ] Rotate `TENSORPILOT_API_KEY` in the secret store.
- [ ] Terminate the Pod in the Runpod console. Do not just stop it: check whether
      stopped Pods or their volumes still incur storage charges.
- [ ] Delete any network volume you created for this session, unless you intend to
      keep it, and note that you kept it.
- [ ] Delete any serverless endpoint or template you created for this session.
- [ ] Delete any container image pushed for this session if storage charges apply.
- [ ] Check the Runpod billing page for running or billed resources from this session.
- [ ] Record the session's start and end times, the Pod type, and the total billed
      time in the work log.

## 8. Troubleshooting quick reference

| symptom | likely cause | action |
|---|---|---|
| `nvidia-smi: command not found` | no GPU or no container runtime | confirm the Pod has a GPU; do not continue |
| preflight `torch_cuda` fails | driver too old for the CUDA 12.8 wheels | confirm the driver version; choose a Pod with a newer driver |
| preflight `vram` fails | GPU below the 16 GiB minimum | choose a larger GPU, or lower `--min-vram-gib` with a note in the record |
| `model_smoke` fails with `OutOfMemoryError` | the device is shared or too small | check `nvidia-smi` for other processes |
| worker never becomes ready | model download failed or the revision is wrong | read `worker.log`; check `TENSORPILOT_CACHE_DIR` and network access |
| `session.sh` says the revision must be 40 characters | the revision is a branch or tag | use the commit SHA |
