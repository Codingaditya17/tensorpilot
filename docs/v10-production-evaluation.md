# v1.0 production-scale evaluation: analysis, changes, acceptance rules and GPU plan

**Status:** prepared and tested locally. **Nothing in this document was run at production scale.** No Pod has been rented, no image published for this
plan, nothing about Runpod modified. Every number below comes either from the saved quick-session evidence (cited by file) or from
`python -m tensorpilot.suite --budget` (labeled as an estimate resting on a price assumption).

## 1. What the saved quick-session evidence shows

Source: `~/Downloads/0.5b-quick-serial-20261026T190153Z.tar.gz` (sha256 `2e357d4e…e28d`) and `0.5b-quick-batch-20261026T190423Z.tar.gz`
(sha256 `35d00357…d3ff`); RTX 4090, Qwen2.5-0.5B-Instruct fp16, image `v1.0.0-rc1-1`, torch 2.9.1+cu128. One repeat per point, 24 to 256 requests.

| Point | Serial tok/s | Batched tok/s | Batched P95 TTFT | Batched P50 / P95 inter-token | Batched mean GPU util |
|---|---|---|---|---|---|
| c=1 | 52.3 | 48.8 (0.93×) | 0.027 s | 19.9 / 22.9 ms | 18% |
| c=8 | 52.5 (P95 TTFT 6.5 s: queueing) | 284.6 (**5.42×**) | 0.221 s | 21.6 / 45.1 ms | 19% |
| c=32 | n/a | 644.2 | 1.166 s | 44.6 / 51.4 ms | 19% |
| c=128 | n/a | 759.2 | 5.420 s | 46.9 / 52.5 ms | 19% |

**The 5.42× is real for this workload, and it says little about TensorPilot at scale.** Findings, each computed from the worker's own
counters (`bench/<exp>/telemetry/server.jsonl` and `gpu.jsonl`):

1. **The decode step costs about 20 ms whether it carries 1 sequence or 128.** Milliseconds per decode step: 19.9 (c=1, 1.0 tokens/step), 20.6 (c=8, 5.8),
   20.6 (c=32, 16.3), 20.6 (c=128, 24.7). The speedup is the batch width at constant step time, not efficiency. A 0.5B model should take far less than
   20 ms of GPU time per step; this one takes 20 ms because the step is dominated by launching roughly 24 layers of small kernels from Python.
2. **The GPU was mostly idle while the engine was busy.** Mean GPU utilization 18–22% and power about 65–69 W (a 4090 under heavy load typically draws several
   hundred watts) while the engine spent 91–97% of wall time inside synchronized steps. For this model the step is launch- and host-bound, not GPU-bound.
   That is a hypothesis about the cause, supported by those two facts. A profiler trace would prove it and the plan does not include one; instead, the report's
   *Engine profile* table shows whether the same signature (flat step time, low utilization) appears at 7B, where weight reads should dominate the step.
3. **Prefill is serialized and interleaved one-to-one with decode.** A prefill step costs about 22 ms regardless of how many prompts are waiting, and handles one prompt.
   At c=128 prefill took 5.9 s of 15.8 s of wall time (37%); the median request spent 2.63 s in prefill steps and 2.48 s in decode steps. Because prefill and decode alternate,
   P50 inter-token latency rises from 20 ms to 45 ms as soon as prompts keep arriving (c=32 and c=128), and P95 TTFT at c=128 is 5.4 s. Throughput flattens
   (285 → 644 → 759 tok/s from c=8 to c=32 to c=128) because the pool saturates on prefill, not on decode.
4. **Joining a decode batch copies the whole prompt's KV, layer by layer.** Each admitted sequence costs 2 × 24 small device copies out of a staging row.

These are the targets of the optimization in §3. None of these numbers says anything about a 7B model, where weight reads, not launches, should dominate
the step.

Limits of this evidence: one repeat per point (percentiles in the small points are effectively maxima), a 0.5B model, fp16, a different Pod from any future run, no
overload, no crash, no sustained load, GPU telemetry once per second.

## 2. The 31/32 exact-match discrepancy

From `batch/…/invariance.json`: 32 prompts (23 short, 9 medium), three ways (solo, all together, staggered). Together vs solo: 31 identical, 1 diverged. Staggered vs
solo: the same single request. The request is plan index 21: a **medium** prompt (324 tokens, so two prefill chunks), 56 output tokens, first divergence at **output token 7**
in both runs.

What the evidence supports:

- Prefill always runs one prompt at a time through the same staging row, so a request's prefill is identical solo and batched. Only **decode shapes** differ: the batch height and the
  attention width (`max(length) + 1` across rows).
- Seven identical tokens means the prompt's KV and positions are right; a slot or position defect shows up at the first token or two.
- The same request diverged at the same token in two different schedules, so it is deterministic in the shapes, not a race.
- fp16 reductions change with shape, and random-word prompts have many near-ties. That fits.

What it cannot show: how large the logit gap was at the flip. The saved evidence has no logits and the old tool did not keep the diverging token ids. **So the
discrepancy is plausibly a numeric near-tie, and not proven to be one.**

What was built so the next run proves it (`tensorpilot/divergence_probe.py`, tests in `tests/test_divergence_probe.py`):

- the batch adapter has an off-by-default `margin_probe` that records the two best tokens and their logits at every emitted token;
- the probe loads the model once, runs every prompt alone and then together through the same adapter the worker uses, and for each diverged request compares the two runs at the first
  differing token. A divergence is a **near-tie** only if both runs agree on the two candidates (each run's runner-up is the other run's choice) **and** each run's logit gap is at most
  4 units in the last place of the logit dtype at that magnitude (fp16: 2⁻¹⁰, bf16: 2⁻⁷ of the leading power of two). Anything else is `unexplained` and the probe exits non-zero;
- `batch_invariance` now keeps the token ids and class of diverged requests;
- stages `probe-0.5b` (the quick-session case, cheap) and `probe-7b` (bf16) run it, and the acceptance checks `1.quick-session-divergence-explained` and
  `1.exact-match-7b-bf16` read it: **PASS** if every request is exact or every divergence is a near-tie, **FAIL** if any is unexplained, **INCONCLUSIVE** if there is divergence and no probe.

100% exact BF16 matches is not promised and not claimed possible: batching legitimately changes reduction order. The criterion accepts documented, justified divergence, and the probe is the justification.

## 3. What changed in the code, and what did not

| Change | Why | Verified how | Not verified |
|---|---|---|---|
| **Packed prefill** (`--prefill-pack-tokens N`, default 0 = old path). Fresh single-chunk prompts at the head of the queue run in **one right-padded forward** (≤ N padded tokens) and write K/V **directly into the decode slots they will occupy**. First-token finishers are compacted out. | Findings 3 and 4: one prompt per forward and a 48-copy join per sequence. | CPU, tiny Llama and Qwen2 models, fp32 and NF4: output identical to the serial backend for bursts, first-token finishers, long prompts mid-queue (FIFO order kept), slot reuse; budget and free-row limits; **mutation-tested** (removing the gap-closing copy and corrupting the last-token gather both fail the tests). | **Any GPU effect.** Whether it helps, and by how much, is exactly what the A/B stage measures. It may not help at 7B. |
| Packing is decided by an **A/B gate** on the 0.5B model (stages `0.5b-quick-batch` and `0.5b-quick-batch-packed`, same Pod) before it is used for 7B. | A change that cannot be measured locally must not enter a $4 session untested. | Gate rule in §6. | — |
| `--backlog` (listen backlog, default 128 unchanged; stages use 2048). | `force_close` makes every request a new connection; 1,024 clients arriving at once would overflow a 128 backlog and add SYN-retry delays that look like server latency. | Argument bounds tested. | The kernel caps it at `net.core.somaxconn`; the stage records, not changes, that. |
| Load generator and benchmark accept **1,024** clients (was 256); the client raises its open-file limit or refuses with a clear message; the stage runner runs `ulimit -n 65535`. | The 512 and 1,024-client points were impossible. | Limit and file-descriptor logic tested. | The Pod's hard limit. |
| The client and server share the Pod's CPUs. The acceptance check marks a level **INCONCLUSIVE** if the load generator fell behind its own schedule (P95 dispatch lag > 1 s). | A starved client measures itself, not the server. | Tested. | — |
| `fault_run` stores `kill.recovery_s`. | The 60-second recovery criterion needs a number, not a sentence. | Existing fault tests. | — |
| `stage_info.py` writes GPU UUID, Pod id, build info and the declared image reference (allow-listed environment names only) into every stage. | "Same physical GPU" and "pinned image" must be provable from evidence. | Unit-tested through the acceptance checks. | — |

**Deliberately not changed: CUDA graphs and kernel fusion for the decode step.** They are the likely cure for finding 1. They cannot be built correctly without
a GPU in the loop; shipping untested graph-capture code into a paid session is the wrong risk. The decision is data-driven: the report's *Engine profile* table (ms per decode
step, tokens per step, GPU utilization, prefill share, from the 7B run) shows whether the 7B step is overhead-bound as the 0.5B step is. If it is, CUDA graphs become the next
milestone with this evidence behind them. If the 7B step is weight-bandwidth-bound, they would not have helped.

## 4. The acceptance checks

`python -m tensorpilot.acceptance SESSION_DIR …` reads evidence only and writes `verdict.json` and `verdict.md`. Thresholds are in
`benchmarks/acceptance/criteria.json`, committed **before** any run; changing one afterwards is a visible commit, never an edit to a verdict. Every item is PASS, FAIL or
INCONCLUSIVE, with its evidence. Missing evidence is INCONCLUSIVE, never a pass (an empty session yields INCONCLUSIVE, tested).

**Evaluation verdict** = gate items only: correctness, safety, accounting, sustained-load reliability, reproducibility, the promised measurements being present, and budget. Any gate FAIL →
FAIL; otherwise any gate INCONCLUSIVE → INCONCLUSIVE; otherwise PASS. **Performance targets** (≥1.5× over serial at c=8 and c=32; P95 TTFT ≤ 2 s and P95 inter-token ≤ 150 ms at the
accepted-load operating point) are reported separately: an unmet target is a *performance FAIL* and does not change the evaluation verdict (tested), and nothing may be altered to hide it.

| # | Items | Exit rule |
|---|---|---|
| 1 | 7B BF16 and 14B NF4 load and generate; ≥99.9% protocol-valid responses; zero silent truncation, duplicate completions or unaccounted requests (journal vs summary vs outcomes); BF16 exact-or-explained; the quick-session divergence explained; NF4 7B vs bf16 (top-1 ≥ 0.90, top-5 overlap ≥ 0.80, perplexity ratio ≤ 1.10); NF4 14B perplexity ≤ 1.05× the 7B bf16 reference on the same text (a sanity bound: no bf16 14B fits) | gate |
| 2 | All seven concurrency levels (1, 32, 64, 128, 256, 512, 1,024) measured with throughput and P50/P95/P99 latency and TTFT, rejections allowed; ≥10,000 requests overall; a rate ladder that actually reached saturation; **speedup ≥ 1.5× (performance)** | gate / performance |
| 3 | ≥99.9% success at sustained load (cancels and documented overload rejections excluded; **timeouts and failures are not**; ≥1,000 requests or INCONCLUSIVE); zero CUDA OOM (counter and log text); overload: queue depth ≤ capacity, rejections > 0, no uncontrolled failures, queue and reservations drained within 10 s, CUDA allocated growth ≤ 64 MiB, ≥99.9% success right after; crash recovery ≤ 60 s with the request-accounting audit passing | gate |
| 4 | SLOs in the evidence equal the SLOs in the criteria file; attainment counts every non-cancelled request (so timeouts cannot flatter latency); **P95 TTFT ≤ 2 s and P95 inter-token ≤ 150 ms at the maximum sustainable rate (performance)** | gate / performance |
| 5 | vLLM: same GPU UUID, same Pod, same revision and precision; ≥3 repeats of `cmp-c8`, `cmp-c32`, `cmp-c128` per engine with an identical workload hash; throughput, goodput, latency, VRAM and error rates for both; a written gap analysis (`## Why vLLM differs`) citing ≥3 measurements | gate (14B vLLM is optional) |
| 6 | Continuous GPU samples (≥80% of seconds, no gap over 5 s); config equals the committed config; 40-hex model revision; image digest declared; archives re-hashed and scanned for credentials | gate |
| 7 | Every stage inside its cap + 1 min; total spend ≤ $9 (needs `--spent-usd` from the Runpod billing page: the ledger cannot see boot and idle time) | gate |

The checker itself is tested with a synthetic session built through the real `load.summarize` (about 70 tests) and was mutation-checked: weakening the speedup test, the INCONCLUSIVE rule,
the queue bound or the recovery limit each fails the suite. One surviving mutation (a crash report with no recorded recovery time) exposed a missing test, which was then added.

## 5. Experiment design

Stages (`python -m tensorpilot.suite --plan`; configs are generated, and tests assert the committed files equal the generator):

| Stage | Purpose | Clients / requests |
|---|---|---|
| `7b-bf16-prod-serial` | the serial baseline | `cmp-c1/8/32`, SHORT, 3 repeats |
| `7b-bf16-prod` | TensorPilot batching, 48 slots, queue 1,100, packing on | `cmp-c1/8/32/128` (3 repeats each); `scale-c{1,32,64,128,256,512,1024}` with 40/1,500/1,500/1,500/2,000/2,000/2,000 requests (**10,280**); `ladder-poisson` 1…32 req/s, 240 each; `sustained-c32` 3,000; cancellation, timeouts, overload (400 req/s, 1,500), recovery-after-overload |
| `vllm-7b-bf16-prod` | vLLM on identical workloads | same `cmp-*`, `scale-*`, `ladder`, `sustained` |
| `14b-nf4-prod` / `vllm-14b-nf4-prod` | 14B NF4, 28 slots, scaled down | the same seven levels with 16/240/240/256/256/512/1,024 requests; ladder 0.5…8 req/s; sustained 800 |
| `faults-7b` | SIGKILL under load, audit accounting, recovery time | 900 requests at 5 req/s |
| `quality-7b`, `quality-14b` | NF4 fidelity | corpus capture |
| `probe-0.5b`, `probe-7b` | divergence explained by logits | 32 prompts |

**Request volume.** The planned TensorPilot 7B stage alone is about 19,900 requests; across all production stages about 44,000 are planned. **The 100,000-request stretch target is not in the
plan:** at the assumed throughput it would take roughly five more hours on the TensorPilot 7B run alone, which together with the vLLM repeat does not fit $9. The acceptance item reports the
executed total and says whether the stretch was met; it will say not met.

**Why queue 1,100.** With queue 192, a closed-loop point at 512 or 1,024 clients mostly measures how fast clients are turned away (a rejected closed-loop client immediately sends its next
request and consumes the plan). Queue 1,100 admits every client, so the sweep measures latency and throughput at that concurrency, and bounded-queue behaviour is the separate `overload`
experiment, which must reject. Timeouts (120 s) and the SLO attainment make saturation visible instead of hiding it.

**Why the 14B set is leaner.** Cancellation, timeouts and overload exercise engine code that does not depend on the model. Repeating them at 14B would spend budget on the same path.

## 6. Budget, sessions and decision gates

`python -m tensorpilot.suite --budget [--price-per-hour P] [--budget-usd 9]`. The price is an **assumption** ($0.90/h, from the earlier plan; the repository records no price). Enter your Pod's real
price and the plan recomputes, scaling budgets down together and reporting any stage the ledger would then refuse.

| Session | Stages (in order) | Expected stage min | `SESSION_BUDGET_MINUTES` | Overhead + slack assumed | Expected | Worst case |
|---|---|---|---|---|---|---|
| **gate** | `0.5b-quick-batch`, `0.5b-quick-batch-packed`, `probe-0.5b` | 17.8 | 36 | 25 + 15 | $0.64 | $1.14 |
| **7b** | `probe-7b`, `7b-bf16-prod-serial`, `7b-bf16-prod`, `faults-7b`, `quality-7b`, `vllm-7b-bf16-prod` | 158.7 | 222 | 75 + 15 | $3.51 | $4.68 |
| **14b** | `14b-nf4-prod`, `quality-14b` | 54.3 | 105 | 83 + 15 | $2.06 | $3.05 |
| **Core total** | | | | | **$6.21** | **$8.87** |
| optional `14b-vllm` | `vllm-14b-nf4-prod` | 39.3 | | 55 + 15 | | +$2.48 (outside the $9) |

Worst case = the session's ledger budget (the ledger refuses any stage whose cap would take used time past it, so stage time cannot exceed it) + the assumed overhead (boot, downloads, model loads,
vLLM install, export) + slack. **Overhead and idle time are not enforced by the ledger**: they are the part of the bound that rests on assumption and on you terminating the Pod. The 14B vLLM
comparison is optional because it needs a fresh Pod, a 28 GB download and a vLLM install; run it only if credits remain.

Production caps are 1.5× each experiment's estimate (older stages use 2.5×). The throughput assumptions (40 tok/s single stream for 7B, 25 for 14B, batching exponent 0.8) are **unmeasured**.
If they are too optimistic, experiments stop at their caps and the acceptance items say INCONCLUSIVE ("cut short by its time cap") instead of the budget being exceeded.

**Decision gates (stop and bring evidence back if any fails):**

1. After `0.5b-quick-batch` and `0.5b-quick-batch-packed`: `quick_compare` as before on the first; the packed stage's `quick-c32` and `quick-c128` tokens/s must be ≥ 0.95× the unpacked ones, its
   invariance exact fraction must not be lower, no failures. If packing fails this gate, run the 7B stages with `PREFILL_PACK_TOKENS=0` (the operator override) and record that packing was not used.
2. `probe-0.5b` must print no `unexplained` divergence. `unexplained` = a possible defect: do not start the 7B session.
3. After `7b-bf16-prod`'s memory gate: `kv_pool.capacity_rows` must equal 48.
4. Read `quality-7b` before the 14B session. If the 14B preflight or smoke fails, terminate rather than debug on a meter.

## 7. Exact commands (nothing below has been run)

Everything that spends money or publishes needs your approval first.

**A. Image (local build done; publishing needs approval).** The immutable tag is `v1.0.0-rc2-1`. The local build is `tensorpilot-exp:v1.0.0-rc2-1`; the registry digest exists only after publishing.

```bash
# publish: pushes the branch and the tag; the experiment-image workflow builds, runs the in-image tests and pushes :v1.0.0-rc2-1
git push origin codex/v10-gpu-scale
git tag exp-v1.0.0-rc2-1 "$(git rev-parse HEAD)" && git push origin exp-v1.0.0-rc2-1
gh run watch                                   # wait for "Experiment image" to go green
docker buildx imagetools inspect ghcr.io/codingaditya17/tensorpilot/cuda-experiment:v1.0.0-rc2-1   # record the digest
```

**B. Pods** (Runpod console, your account). 1 × RTX 4090, public IP, SSH on TCP 22 only, start command empty, image
`ghcr.io/codingaditya17/tensorpilot/cuda-experiment@sha256:<digest from A>`. Environment variables (public values): `TENSORPILOT_IMAGE_REF` = that same `@sha256:` reference (so the evidence
records it), `TENSORPILOT_POD_MAX_MINUTES` = gate **90**, 7b **330**, 14b **230**. Container disk / volume: gate 20/20 GB; 7b 50/120 GB, **driver 580+ for vLLM**; 14b 50/120 GB, RAM ≥ 32 GB.
Confirm the hourly price and re-run `suite --budget --price-per-hour <price>` first.

**C. Gate session** (on the Pod, as user `tensorpilot`, inside `tmux`):

```bash
export TENSORPILOT_API_KEY="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
export SESSION_BUDGET_MINUTES=36 PRICE_PER_HOUR_USD=<price>
scale_session.sh 0.5b-quick-batch
scale_session.sh 0.5b-quick-batch-packed
scale_session.sh probe-0.5b
```

**D. 7B session** (`SESSION_BUDGET_MINUTES=222`; add `PREFILL_PACK_TOKENS=0` only if gate 1 failed):

```bash
scale_session.sh probe-7b
scale_session.sh 7b-bf16-prod-serial
scale_session.sh 7b-bf16-prod
scale_session.sh faults-7b
scale_session.sh quality-7b
deploy/runpod/install_vllm.sh --plan && deploy/runpod/install_vllm.sh
scale_session.sh vllm-7b-bf16-prod
```

**E. 14B session** (fresh Pod, `SESSION_BUDGET_MINUTES=105`): `scale_session.sh 14b-nf4-prod` then `scale_session.sh quality-14b`. Optional, if credits remain (its own Pod, budget ~60):
`install_vllm.sh && scale_session.sh vllm-14b-nf4-prod`.

**F. Export, then terminate the Pod** (terminating, not stopping, ends billing): `scp` the session directory to your Mac, then in the Runpod console terminate every Pod.

**G. Verdict and report** (your Mac; put every Pod's extracted stage directories under one directory, and pass each Pod's `ledger.jsonl`):

```bash
python -m tensorpilot.acceptance v10-evidence --ledger gate/ledger.jsonl --ledger s7b/ledger.jsonl --ledger s14b/ledger.jsonl \
    --archives v10-evidence/archives --report report/report.md --spent-usd <Runpod billing page total> \
    --output verdict.json --markdown verdict.md
python -m tensorpilot.report_v10 v10-evidence --output report --baseline benchmarks/baselines/v0.9.1-4-rtx4090 --acceptance verdict.json
```

Write the `## Why vLLM differs` section of the report from the *Engine profile* table, citing measurements, then run the acceptance command again (it reads the report for item 5).

## 8. Threats to validity

- **One card, one Pod type, one price assumption.** vLLM runs after TensorPilot in the same Pod (never together); thermal and clock drift are recorded, not controlled.
- **The load generator shares the Pod's CPUs with the server.** Mitigated by the dispatch-lag check, not eliminated.
- **Different CUDA stacks** (TensorPilot torch 2.9.1/CUDA 12.8, vLLM 0.31.0 on torch 2.13/CUDA 13) and vLLM's paged KV, CUDA graphs and larger possible batch. The comparison is an engineering baseline, not a fair fight.
- **Fixed-slot KV pool, not paged**: 48 slots at 7B means concurrency above 48 queues by construction. That is the design being measured.
- **Closed-loop clients above the slot count measure queueing**, which is why the SLO-bearing latency criterion uses the open-loop operating point.
- **The sustained-load success criterion needs ≥1,000 requests** to say anything about 99.9%; the sustained run has 3,000.
- **NF4 14B has no bf16 reference on 24 GB.** Its fidelity item is a sanity bound against the 7B reference, said so in the verdict.
- **Not covered by any stage:** multi-GPU, long context beyond 2,048 tokens, speculative decoding, quantized KV, anything above 48 / 28 slots.
