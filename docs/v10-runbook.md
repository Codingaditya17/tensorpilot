# v1.0 GPU evaluation: Pod runbook

Nothing here has been run. Do not rent a Pod or publish the image until the checklist in section 0 is yours to tick. The plan, costs and caveats
are in [`v10-evaluation-plan.md`](v10-evaluation-plan.md).

## 0. Before renting (free)

- [ ] You have read the cost table and chosen a `SESSION_BUDGET_MINUTES` per session. A stage is refused if used time plus its hard cap would exceed it.
- [ ] You know the price your Pod will cost per hour (set `PRICE_PER_HOUR_USD` to see cost in the ledger; the repository records no price).
- [ ] The v1.0 experiment image exists. **It does not yet.** Publishing it is a separate approval: push a tag such as `exp-v1.0.0-rc1-1` (the workflow
      builds, runs the test suites against the image's own package, and publishes `…/cuda-experiment:v1.0.0-rc1-1`). Use its digest below.
- [ ] You have one SSH public key on your Runpod account.

## 1. Pod settings

| Setting | Value |
|---|---|
| Pod type | 1 × RTX 4090 (24 GB), a host that offers a public IP and a driver of 580 or newer (check `nvidia-smi` first; vLLM's CUDA 13.0 runtime needs it) |
| Image | `ghcr.io/codingaditya17/tensorpilot/cuda-experiment@sha256:<digest, published only after your approval>` |
| Container disk | 50 GB |
| Volume disk | 120 GB mounted at `/workspace` (model cache about 43 GiB, vLLM environment about 15 GiB, results) |
| RAM | 32 GB or more (the 14B shards are quantized as they load) |
| Expose HTTP ports | none |
| Expose TCP ports | `22` only |
| Start command / entrypoint | empty |
| Environment variables | none; the API key is generated inside your SSH session |

The entrypoint chooses a verified-writable directory for the model cache and evidence; the login banner prints it (`/workspace/...` normally, container disk with a
warning if the volume refuses). The experiment image's hard 120-minute lifetime backstop stops GPU billing for a forgotten Pod **only if the container exits**; it does not
terminate or delete the Pod. Raise it for these sessions: set `TENSORPILOT_POD_MAX_MINUTES` to your session budget plus an hour in the template (a public variable, not a secret). The stage runner **refuses to start** if the backstop is shorter than the budget plus 60 minutes, so a forgotten setting cannot end a session mid-stage.

## 2. Connect and verify (about 3 minutes)

```bash
ssh -i ~/.ssh/<key> -p <public-port> tensorpilot@<public-ip>      # the user is tensorpilot, not root
tmux new -s run                                                    # survives a dropped connection
nvidia-smi                                                         # one RTX 4090, driver >= 580 for the vLLM stage
export SESSION_BUDGET_MINUTES=360 PRICE_PER_HOUR_USD=<your price>
export TENSORPILOT_API_KEY="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
scale_session.sh --list
scale_session.sh --plan 0.5b-serial
```

## 3. Session A (expected about 3.7 hours of Pod time)

Run one stage at a time and read its output before the next. **Each gate below is a stop condition.**

```bash
scale_session.sh 0.5b-serial          # regression against the immutable baseline
scale_session.sh 0.5b-batch           # the first real test of CUDA batching, including batch invariance
scale_session.sh 7b-bf16-serial
scale_session.sh 7b-bf16-batch        # the main experiment
scale_session.sh quality-7b           # bf16 reference, then NF4, then the comparison
scale_session.sh faults-0.5b
scale_session.sh faults-7b
deploy/runpod/install_vllm.sh --plan && deploy/runpod/install_vllm.sh
scale_session.sh vllm-7b-bf16         # same Pod, after the TensorPilot stages
```

Gates:

1. **After `0.5b-serial`:** throughput of `regression-c1` should be near the baseline's 57.5 tok/s. If it is far lower, stop: something regressed or the Pod is unlike the baseline's.
2. **After `0.5b-batch`:** the smoke passed; `invariance.json` shows requests matching their solo runs and no divergence near the start of a sequence; no OOM event; `sweep-c8` beat `serial-c8`. If batching is
   incorrect or no faster, **stop and bring the evidence back**; do not continue to 7B.
3. **Before `7b-bf16-batch`:** `/health` `kv_pool.capacity_rows` should be near the 64 requested (at least 32). If the pool is much smaller, the stage's high-concurrency points are meaningless: stop.
4. **Before vLLM:** the TensorPilot 7B stages are archived. vLLM needs `--generation-config`; the stage refuses to run without it.

## 4. Session B on a fresh Pod (expected about 4.3 hours)

```bash
scale_session.sh 7b-bf16-long
scale_session.sh 14b-nf4-batch        # first NF4 load on CUDA: watch the engine log and the first smoke
scale_session.sh 14b-nf4-long
scale_session.sh quality-14b          # absolute metrics only: no bf16 reference fits
deploy/runpod/install_vllm.sh && scale_session.sh vllm-14b-nf4
```

Gate: do not start session B unless session A's `quality-7b` comparison is back and reads sensibly. NF4 on CUDA is untested; if `14b-nf4-batch`'s preflight or smoke fails, terminate rather than debug on a meter.

## 5. Export evidence, then terminate (before anything else is left running)

```bash
ls -lh "$TENSORPILOT_V10_SESSION"/*.tar.gz                        # one archive and checksum per stage, plus ledger.jsonl
```

From your Mac (the ledger and stage directories are the report's input; the archives are the checksummed originals):

```bash
scp -r -i ~/.ssh/<key> -P <port> tensorpilot@<ip>:<session-dir> ./v10-evidence
shasum -a 256 v10-evidence/*.tar.gz        # compare with each stage's printed hash
python -m tensorpilot.report_v10 v10-evidence --output report --baseline benchmarks/baselines/v0.9.1-4-rtx4090
```

**Termination checklist:**

- [ ] every stage archive downloaded, its SHA-256 matches, and `tar -tzf` lists its contents
- [ ] `ledger.jsonl` downloaded and every stage appears with an exit status
- [ ] the Pod is **terminated** in the Runpod console, not stopped (a stopped Pod can keep its volume billed)
- [ ] the console's billing page shows nothing still running
- [ ] a network volume, if you made one, is deleted or recorded as kept

## 6. What a failed stage looks like

A stage that fails or hits its cap still writes its evidence and a ledger entry with the exit status (`124` is the hard cap, `143` a termination signal). The archive refuses to include a
file containing the API key. A stage whose preflight fails stops before the engine starts. The report lists every planned experiment as completed, aborted, skipped or NOT RUN.

## 7. Troubleshooting

| Symptom | Likely cause | Action |
|---|---|---|
| `refused: … exceeds the … minute session budget` | the ledger guard | raise `SESSION_BUDGET_MINUTES` only if you accept the cost |
| `the stage plans assume a 24 GB card` | a smaller GPU | do not run; the slot counts assume 24 GB |
| `vLLM's CUDA 13.0 runtime needs driver 580 or newer` | an older host | run only the TensorPilot stages on this Pod |
| preflight `quantization_stack` fails | bitsandbytes cannot run | stop; the worker will not fall back to bf16 |
| `kv_pool.capacity_rows` far below the request | less free VRAM than planned | read `/health` `kv_pool` and `last_measured_memory`; stop before the sweep |
| vLLM flag guard fails | a vLLM build without `--generation-config` or bitsandbytes | the stage refuses by design; do not edit it to continue |
