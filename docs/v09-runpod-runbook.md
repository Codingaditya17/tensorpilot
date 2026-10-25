# v0.9 Runpod GPU session runbook

Use this runbook for the first bounded GPU session, which is about thirty minutes of
measured work inside a Pod that you have budgeted for longer. Nothing here has run on
a GPU yet. Every step has a check, so a failure stops the session early rather than
spending rental time on troubleshooting.

Do not rent or start a Pod until the GO/NO-GO checklist in
[`v09-readiness-report.md`](v09-readiness-report.md) reads GO.

## 0. Pre-rental checklist (local, no GPU)

- [ ] The release commit is clean: `git status` shows no changes.
- [ ] Tests pass: `python -m pytest -q` and `python -m ruff check tensorpilot tests`.
- [ ] The container image was built by the manual CI workflow (see section 1 of the
      readiness report) or locally on a machine with registry access. The build must
      have completed; a partial build is not evidence.
- [ ] The model revision is pinned: `TENSORPILOT_REVISION` is a 40-character commit SHA.
      Current pin: `7ae557604adf67be50417f59c2c2f167def9a775` for
      `Qwen/Qwen2.5-0.5B-Instruct`.
- [ ] Three distinct secrets are generated and stored in the Runpod secret store:
      `TENSORPILOT_API_KEY` (worker), `TENSORPILOT_CONTROL_API_KEY` and
      `TENSORPILOT_DATA_API_KEY`. The session script refuses to run if any two match.
- [ ] The Pod template exposes **no public HTTP ports** (see section 2).

## 1. Pod selection

- One NVIDIA GPU with 24 GB of memory, as planned. The reference model's weights are
  about 1 GB in bfloat16; the preflight checks the real size from the Hub before any
  download.
- Host driver: the preflight applies NVIDIA's documented policy. A driver below 525 is
  a hard failure. A driver from 525 to 570.25 is in the CUDA 12.x minor-version
  compatibility range; the gate warns and requires the smoke test to pass. 570.26 or
  newer meets the CUDA 12.8 toolkit requirement. Check `nvidia-smi` on the Pod first.
- Container disk: at least 40 GB. This covers the image or Python environment (the CUDA
  wheels are several gigabytes), the Hugging Face cache, and the results directory.
- Pricing, availability and billing rules change. Check the Runpod console at the time
  you rent. Nothing in this repository records a price.

## 2. Network exposure and credentials

- Every TensorPilot process binds `127.0.0.1`. Do not map any of these ports
  (8080 worker, 8090 control plane, 8091 data plane) as public HTTP ports in the Pod
  template. Map only SSH, if you use it.
- To view a service from your laptop, tunnel it over SSH:
  `ssh -L 8080:127.0.0.1:8080 <pod-ssh-target>`.
- The control plane and the data plane are internal services. Their credentials are
  separate from the worker's, so a leaked worker key does not grant placement access.
- Keys live in the Runpod secret store and are injected as environment variables. They
  must never be written to a file in the repository or into the evidence directory. The
  session script scans the evidence for every key value and refuses to archive if one
  appears.
- Rotate all three keys after the session.

## 3. Pod start and environment

**Rented Pod: use the experiment image and follow [`v09-runpod-experiment-pod.md`](v09-runpod-experiment-pod.md).**
It ships the scripts and configs, SSH, and a hard-capped `first-run.sh`. The steps below are for a bare
environment you set up yourself.

On the Pod, from the repository root (the session script changes to the repository
root itself):

```bash
export TENSORPILOT_MODEL=Qwen/Qwen2.5-0.5B-Instruct
export TENSORPILOT_REVISION=7ae557604adf67be50417f59c2c2f167def9a775
export TENSORPILOT_CACHE_DIR=/workspace/huggingface
export TENSORPILOT_RESULTS=/workspace/results/v09-$(date -u +%Y%m%dT%H%M%SZ)
# The three keys come from the Runpod secret store.

nvidia-smi
python3 -m pip install --require-hashes -r requirements-cuda.txt
python3 -m pip install --no-deps -e .
deploy/runpod/session.sh --plan
```

`--plan` validates the environment and prints the steps without starting anything.

If you run the container image instead of the bare environment, use
`docker run --init --gpus all -p 127.0.0.1:8080:8080 ...` with the same variables. Do not
publish the port beyond loopback.

## 4. Run the session

```bash
deploy/runpod/session.sh
```

By default the session runs `benchmarks/gpu/v09-first-30min.json`, which is three short
experiments with a 30-minute session budget. The full single-GPU suite is opt-in:
`TENSORPILOT_BENCH_CONFIG=benchmarks/gpu/v09-single-gpu.json deploy/runpod/session.sh`.

| step | what it proves | evidence |
|---|---|---|
| 1 diagnostics | host, driver, CUDA runtime, installed versions | `diagnostics/` |
| 2 preflight | driver policy, pinned and reachable revision, storage, VRAM fit against real weight size, dependency pins, smoke inference. Any failed required check stops the session | `preflight.json` |
| 3 worker | the model loads, warms up and reports ready on `/readyz` | `worker.log` |
| 4 gateway | control plane registers the worker; data plane streams 5 requests, each validated by the protocol checker | `gateway-smoke.json` |
| 5 benchmark | the bounded experiments run against the worker, with time caps and abort rules | `plan.txt`, `bench/manifest.json` |
| 6 evidence | secret scan, archive, SHA-256 checksum | `<results>.tar.gz`, `.sha256` |
| 7 teardown | every started process is stopped with SIGTERM | exit status |

A preflight failure, a gateway failure, or a benchmark timeout ends the session. Fix the
cause and rerun from step 2 in a new results directory.

## 5. Measured parameters and placeholders

- Measured fields come only from the runs: latency, TTFT, inter-token latency,
  throughput, errors, VRAM and telemetry. None of them is copied from a placeholder.
- `benchmarks/gpu/*.json` declare `assumptions.status: "placeholder_unmeasured"`. The
  plan command labels its time figures as **ESTIMATES**, and the runner refuses any
  config that declares the assumptions as measured.
- After the first 30-minute run, read the measured decode speed from
  `first-closed-c1`, then regenerate the plan before budgeting the full suite. The
  assumptions in the config are not updated automatically.

## 6. Time limits and abort conditions

Every runnable experiment has these limits in its config:

| field | meaning |
|---|---|
| `max_minutes` | hard cap on the experiment's wall time, enforced with a timeout; required |
| `abort_error_rate` | an experiment stops when a report's error rate exceeds this fraction (0 means any error stops it) |
| `max_session_minutes` (top level) | the session budget; experiments that start after it is exhausted are skipped |

An experiment also aborts when, after any report:

- the worker is not ready,
- the CUDA out-of-memory counter increases, or
- the time cap is reached.

The manifest records each experiment's status, abort reason, elapsed and capped seconds.
`session.sh` also wraps the benchmark in an outer `timeout` (default 35 minutes), so a
hung process cannot hold the Pod.

## 7. Manual recovery experiment

This experiment is listed as `recovery-supervised-kill` in the full suite. It needs a
person to kill a process, so run it by hand after the scripted steps. It uses its own
supervised worker on port 8180, so it never collides with the session worker.

```bash
export TENSORPILOT_BACKEND=cuda TENSORPILOT_MODEL="$TENSORPILOT_MODEL" TENSORPILOT_REVISION="$TENSORPILOT_REVISION"
python3 -m tensorpilot.supervisor --url http://127.0.0.1:8180 --port 8180 --execution serial \
  --max-restarts 3 --lifecycle-log "$TENSORPILOT_RESULTS/recovery.lifecycle.jsonl" \
  --worker-log "$TENSORPILOT_RESULTS/recovery.worker.log" &
```

The supervisor inherits the `TENSORPILOT_*` environment, so the backend and model come
from the variables above. Then:

1. Read the worker PID from the lifecycle log, not from `pgrep`:
   `grep worker_ready "$TENSORPILOT_RESULTS/recovery.lifecycle.jsonl" | tail -1`.
2. Start a closed-loop load against the supervised worker:
   `TENSORPILOT_API_KEY="$TENSORPILOT_API_KEY" python3 -m tensorpilot.load --url http://127.0.0.1:8180 --requests 40 --concurrency 2 --max-tokens 128 --output "$TENSORPILOT_RESULTS/recovery-load.json"`.
3. While a stream is active, `kill -9` that PID. Record the terminal event the affected
   client receives: it must be an error with a terminal code, and no tokens may be
   replayed.
4. Wait for `worker_restart_scheduled`, then `worker_ready` with a new PID.
5. Run a short closed-loop load and confirm success.
6. Stop the supervisor with `kill -INT <supervisor pid>`.

Pass condition: exactly one terminal event per request, no duplicated tokens, and
readiness restored within the supervisor's ready timeout.

## 8. Export the evidence before terminating anything

The session archives `$TENSORPILOT_RESULTS` and writes its checksum. Before you
terminate the Pod:

```bash
ls -lh "$TENSORPILOT_RESULTS.tar.gz" "$TENSORPILOT_RESULTS.tar.gz.sha256"
sha256sum -c "$TENSORPILOT_RESULTS.tar.gz.sha256"
tar -tzf "$TENSORPILOT_RESULTS.tar.gz" | head
```

Download the archive with `scp` (exact command in `v09-runpod-experiment-pod.md`) and compare
`shasum -a 256` on your machine with the hash the session printed. Only after that check passes should you treat the
evidence as saved.

## 9. Teardown and billable-resource checklist

Stopping processes does not stop billing. Work through this list:

- [ ] `session.sh` printed its teardown line, and `pgrep -fa tensorpilot` returns nothing.
- [ ] The evidence archive is downloaded, and its checksum verifies on your machine.
- [ ] Rotate `TENSORPILOT_API_KEY`, `TENSORPILOT_CONTROL_API_KEY` and
      `TENSORPILOT_DATA_API_KEY` in the secret store.
- [ ] Remove the Hugging Face cache from the Pod if you will not reuse it:
      `rm -rf "$TENSORPILOT_CACHE_DIR"`.
- [ ] Terminate the Pod. Do not just stop it: a stopped Pod can keep its volumes billed.
- [ ] Delete any network volume you created for this session, or record that you kept it.
- [ ] Delete any serverless endpoint or template created for this session.
- [ ] Delete any container image pushed for this session, if storage charges apply.
- [ ] Check the Runpod billing page for resources still running or billed.
- [ ] Record start and end times, the Pod type, and total billed time in the work log.

## 10. Troubleshooting quick reference

| symptom | likely cause | action |
|---|---|---|
| `nvidia-smi: command not found` | no GPU or no container runtime | confirm the Pod has a GPU; do not continue |
| preflight `driver` fails (below 525) | driver too old for CUDA 12.x | choose another Pod |
| preflight `driver` warns | driver is 525 to 570.25 | expected; the smoke test must pass |
| preflight `torch_cuda` fails | PyTorch cannot initialize CUDA | check the wheels match the lock and the driver |
| preflight `dependency_pins` fails | environment drifted from the lock | reinstall with `--require-hashes` |
| preflight `remote_revision` fails | revision unreachable, wrong commit, or network | confirm the SHA on the Hub; check egress |
| preflight `storage` fails | container disk too small for the cache | use a larger container disk |
| preflight `vram` fails | weights plus headroom exceed free memory | use a larger GPU |
| `model_smoke` fails | out-of-memory or a backend error | read the detail; check `nvidia-smi` for other processes |
| worker never becomes ready | model download or load failed | read `worker.log` |
| `session.sh` refuses the keys | two of the three keys are equal | generate distinct keys |
| session refuses to archive | a credential value is in the evidence | find and remove it, then rerun the archive step |
