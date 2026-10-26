# v1.0 GPU evaluation: Pod runbook

> The production-scale plan (scale sweep to 1,024 clients, vLLM comparison, acceptance verdict, $9 budget) has its own sessions, gates and commands in
> [`v10-production-evaluation.md`](v10-production-evaluation.md). This runbook still describes the Pod mechanics both plans share.

Nothing here has been run. Do not rent a Pod until you have decided to, and do not run anything beyond the first short session until its gate has passed. The plan, costs and
caveats are in [`v10-evaluation-plan.md`](v10-evaluation-plan.md). **Terminating the Pod, not exiting the container, is what stops billing.**

## 0. Before renting (free)

- [ ] You have read the cost table and chosen the session you want. The image digest is the one in the release notes of the experiment image you approved (the image cannot contain its own digest).
- [ ] You know your Pod's hourly price (set `PRICE_PER_HOUR_USD` so the ledger shows a cost estimate; the repository records no price).
- [ ] Your Runpod account has one SSH public key.

## 1. Pod settings

| Setting | First short session | Session A | Session B |
|---|---|---|---|
| Pod type | 1 × RTX 4090 (24 GB), public IP, driver shown by `nvidia-smi` | same, **driver 580 or newer** (vLLM) | same, driver 580 or newer |
| Image | `ghcr.io/codingaditya17/tensorpilot/cuda-experiment@sha256:<digest>` | same | same |
| Container disk | 20 GB | 50 GB | 50 GB |
| Volume disk at `/workspace` | 20 GB | 120 GB | 120 GB |
| Expose HTTP ports | none | none | none |
| Expose TCP ports | `22` only | `22` only | `22` only |
| Start command | empty | empty | empty |
| **`TENSORPILOT_POD_MAX_MINUTES`** | **150** | **480** | **540** |
| RAM | 16 GB or more | 32 GB or more | 32 GB or more |

`TENSORPILOT_POD_MAX_MINUTES` is a public setting, not a secret; the image's default of 120 is too short for sessions A and B. The container exits when it elapses, and the stage runner
guarantees a stage's evidence is archived before then (section 6), but a container that has exited is not assumed to have stopped billing: you still terminate the Pod.

## 2. Connect and verify (about 3 minutes)

```bash
ssh -i ~/.ssh/<key> -p <public-port> tensorpilot@<public-ip>      # the user is tensorpilot, not root
tmux new -s run                                                    # survives a dropped connection (and an interrupt is archived anyway)
nvidia-smi                                                         # one RTX 4090
export SESSION_BUDGET_MINUTES=60 PRICE_PER_HOUR_USD=<your price>   # 60 for the first short session
export TENSORPILOT_API_KEY="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
scale_session.sh --list
scale_session.sh --plan 0.5b-quick-serial                          # prints the cap, the budget verdict and the Pod lifetime verdict
```

## 3. The first short session (about 25 minutes, hard-capped at 37 minutes of stage time)

```bash
scale_session.sh 0.5b-quick-serial     # cold start, 1 client, 8 clients on the serial worker; each experiment capped at 3 minutes
scale_session.sh 0.5b-quick-batch      # batch-invariance check, then 1, 8, 32 and 128 clients, then cancellation, on the batching worker
SERIAL="$(ls -d "$TENSORPILOT_V10_SESSION"/0.5b-quick-serial-*/ | tail -1)"    # the stage directories (the trailing / skips the .tar.gz files)
BATCH="$(ls -d "$TENSORPILOT_V10_SESSION"/0.5b-quick-batch-*/ | tail -1)"
python -m tensorpilot.quick_compare "$SERIAL" "$BATCH" --output "$TENSORPILOT_V10_SESSION/quick-verdict.json"
```

What the commands check, in order:

1. **Preflight** (GPU, driver, disk, model pinned by revision) and a **smoke test** (5 requests must succeed) stop the stage before any load if something is wrong.
2. **Memory gate** (batching stage): the real free memory inside the allocator cap must be at least 1.5 GiB and every requested slot granted, or the stage stops without benchmarking.
3. **Batch-invariance:** each request run alone, with 32 together, and staggered; outputs are compared token by token.
4. **`quick_compare`** prints PASS or FAIL for: batched c=8 throughput at least 1.5× serial c=8; batched c=1 at least 0.7× serial c=1; at least 90% of requests identical to their solo output;
   no failed request and every experiment completed; memory gate passed; neither stage interrupted.

**If `QUICK GATE: FAIL`, download the evidence and stop; do not run anything larger.** If it passes, that means the 7B session is worth its cost, not that batching is fast.
Then export (section 5) and terminate, or continue to session A on the same Pod only if its lifetime and budget allow (the runner will refuse a stage that does not fit).

## 4. Session A and session B

Run one stage at a time and read each stage's output before the next. Set `SESSION_BUDGET_MINUTES=360` for each session.

```bash
# Session A (Pod lifetime 480)
scale_session.sh 0.5b-serial          # regression against the immutable baseline (skip if the quick session passed)
scale_session.sh 0.5b-batch           # skip likewise
scale_session.sh 7b-bf16-serial
scale_session.sh 7b-bf16-batch        # the main experiment; check kv_pool.capacity_rows in the memory gate output first
scale_session.sh quality-7b           # bf16 reference, then NF4, then the comparison
scale_session.sh faults-0.5b
scale_session.sh faults-7b
deploy/runpod/install_vllm.sh --plan && deploy/runpod/install_vllm.sh
scale_session.sh vllm-7b-bf16         # same Pod, after the TensorPilot stages

# Session B on a fresh Pod (Pod lifetime 540)
scale_session.sh 7b-bf16-long
scale_session.sh 14b-nf4-batch        # first NF4 load on CUDA: watch the engine log and the first smoke
scale_session.sh 14b-nf4-long
scale_session.sh quality-14b          # absolute metrics only: no bf16 reference fits
deploy/runpod/install_vllm.sh && scale_session.sh vllm-14b-nf4
```

Gates: after `7b-bf16-batch`'s memory gate, `kv_pool.capacity_rows` must equal the requested 48; do not start session B unless `quality-7b` has been read and looks sensible; if
`14b-nf4-batch`'s preflight or smoke fails, terminate rather than debug on a meter. vLLM needs `--generation-config vllm`; the stage refuses to run without it.

## 5. Export evidence, then terminate

```bash
ls -lh "$TENSORPILOT_V10_SESSION"/*.tar.gz                        # one archive and checksum per stage, plus ledger.jsonl
```

From your Mac:

```bash
scp -r -i ~/.ssh/<key> -P <port> tensorpilot@<ip>:<session-dir> ./v10-evidence
shasum -a 256 v10-evidence/*.tar.gz        # compare with each stage's printed hash
python -m tensorpilot.report_v10 v10-evidence --output report --baseline benchmarks/baselines/v0.9.1-4-rtx4090   # full stages
```

**Termination checklist:**

- [ ] every stage archive downloaded, its SHA-256 matches, and `tar -tzf` lists its contents
- [ ] `ledger.jsonl` downloaded and every stage appears with an exit status
- [ ] the Pod is **terminated** in the Runpod console, not stopped (a stopped Pod can keep its volume billed)
- [ ] the console's billing page shows nothing still running
- [ ] a network volume, if you made one, is deleted or recorded as kept

## 6. Interrupted, capped and late stages

Every way a stage can end writes its evidence and a ledger entry with an exit status:

| Exit | Meaning |
|---|---|
| 0 | the stage completed |
| 1 | a check or the benchmark failed (preflight, readiness, smoke, memory gate, the benchmark itself) |
| 124 / 137 | the hard cap or its kill fired |
| 125 | the **lifetime watchdog** stopped the stage 12 minutes before the Pod's deadline |
| 129 / 130 / 143 | SIGHUP (SSH dropped), SIGINT (Ctrl-C) or SIGTERM: the running step was stopped |

For 125, 129, 130 and 143 the unfinished experiment is recorded as aborted with reason `interrupted`, the rest as skipped, and the unfinished request journals are kept; the engine is stopped and
the archive is written and checksummed. Press Ctrl-C once and wait: a second signal during the archive is ignored on purpose. A stage is refused outright (exit 3) if the Pod's remaining lifetime
cannot cover its cap, one model-load limit and the archive margin. A stage refused for lifetime means: terminate this Pod and start one with a larger `TENSORPILOT_POD_MAX_MINUTES`.

## 7. Troubleshooting

| Symptom | Likely cause | Action |
|---|---|---|
| `refused: … exceeds the … minute session budget` | the ledger guard | raise `SESSION_BUDGET_MINUTES` only if you accept the cost |
| `refused: this Pod's lifetime backstop leaves N minutes` | the Pod will exit before the stage can finish | terminate; restart with a larger `TENSORPILOT_POD_MAX_MINUTES` or pick a shorter stage |
| `the stage plans assume a 24 GB card` | a smaller GPU | do not run; the slot counts assume 24 GB |
| `vLLM's CUDA 13.0 runtime needs driver 580 or newer` | an older host | run only the TensorPilot stages on this Pod |
| preflight `quantization_stack` fails | bitsandbytes cannot run | stop; the worker will not fall back to bf16 |
| `the memory gate failed` | less free VRAM than planned, fewer slots granted, an OOM, or no memory data | read `memory-gate.json` and `/health`; do not benchmark |
| vLLM flag guard fails | a vLLM build without `--generation-config` or bitsandbytes | the stage refuses by design; do not edit it to continue |
