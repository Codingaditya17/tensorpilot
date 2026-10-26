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

`python -m tensorpilot.suite --budget [--price-per-hour P] [--budget-usd B]`. The price is a figure you confirm when renting (**$0.91/h including storage** for the 7B session below; earlier plans used $0.90 and $0.89).
The **ledger counts each stage's whole wall time**: preflight (which loads the model once), the worker's own load, the smoke test, the memory gate and the export all count, and so does the model download in a session's
first stage. The model assumes per-stage start-up minutes (TensorPilot 6, vLLM 7, quality 2, faults 3, probe 2) and a 30 MB/s download. **Those are assumptions**; the first 7B run replaces them with measurements.
`SESSION_BUDGET_MINUTES` is the largest, over the stages, of 1.15 × (expected ledger time used before the stage) + its cap, because the runner refuses a stage if used time plus its cap would exceed the budget.

| Session | Stages (in order) | Expected ledger min | `SESSION_BUDGET_MINUTES` | Outside the ledger (boot, vLLM install, idle) + slack | Expected | Worst case at $0.91/h |
|---|---|---|---|---|---|---|
| gate (done) | `0.5b-quick-batch`, `0.5b-quick-batch-packed`, `probe-0.5b` | | 36 | | | |
| **7b** | `probe-7b`, `7b-bf16-prod-serial`, `7b-bf16-prod`, `faults-7b`, `vllm-7b-bf16-prod`, `quality-7b` | 192.8 | **250** | 40 + 15 | **$3.53** | **$4.63** |
| **14b** (conditional) | `14b-nf4-prod`, `quality-14b` | 78.0 | 105 | 25 + 15 | $1.56 | $2.20 |

An earlier version of this plan used 225 minutes for the 7B session. With the ledger counting start-up and the download, about 117 minutes are used before the vLLM stage, so 117 + its 115-minute cap exceeds 225 and the ledger
would refuse it; 250 passes with a 15% margin. Both sessions together cost at most $6.83 in the worst case; with the $0.43 already spent that stays under the $9.00 cap. Worst case = the ledger budget + the outside-the-ledger
overhead + slack: the ledger bounds stage time, and **boot, the vLLM install and idle time are the part that depends on you terminating the Pod**. The optional `14b-vllm` session is not in these totals.

Production caps are 1.5× each experiment's estimate (older stages use 2.5×). The throughput assumptions (40 tok/s single stream for 7B, 25 for 14B, batching exponent 0.8) are **unmeasured**.
If they are too optimistic, experiments stop at their caps and the acceptance items say INCONCLUSIVE ("cut short by its time cap") instead of the budget being exceeded.

**Decision gates (stop and bring evidence back if any fails):**

1. After `0.5b-quick-batch` and `0.5b-quick-batch-packed`: `python -m tensorpilot.packing_gate UNPACKED_DIR PACKED_DIR` applies a rule fixed in advance. Any regression
   (throughput under 0.95× at any shared point, P95 TTFT over 1.10× at c=32 or c=128, a failed request, an invariance result more than one request worse or under 0.90, a failed memory gate)
   → `REGRESSION`; no regression but no gain (under 1.05× throughput and over 0.9× TTFT at c=128) → `NO_GAIN`. Both mean: run the 7B stages with `PREFILL_PACK_TOKENS=0`.
   Only `USE_PACKING` carries the unmeasured path into the 7B session. An unmeasured code path with no demonstrated gain has no business in a larger run.
2. `probe-0.5b` must print no `unexplained` divergence. `unexplained` = a possible defect: do not start the 7B session.
3. After `7b-bf16-prod`'s memory gate: `kv_pool.capacity_rows` must equal 48.
4. Read `quality-7b` before the 14B session. If the 14B preflight or smoke fails, terminate rather than debug on a meter.

## 6a. Gate session results (2026-10-26, RTX 4090 UUID `GPU-2b01fb0b…`, image `v1.0.0-rc2-1`, evidence checksums verified)

Three stages, all exit 0 (2.5, 2.1 and 0.7 minutes). Unpacked `quick-c128` 746 tok/s, packed 2,257 (**3.03×**); P95 TTFT at c=128 5.72 s → 0.309 s; c=32 636 → 835 tok/s; c=1 and c=8 not worse
(1.03×, 1.08×). `packing_gate` → **USE_PACKING**, reproduced here from the raw reports. The unpacked numbers reproduce the first session on a different Pod (746 vs 759 tok/s at c=128; 20.7 ms decode step;
GPU utilization 18%), so the flat-step finding is not a one-Pod artifact. Packed: tokens per step at c=128 rose from 24.7 to 46.7 and prefill's share of engine time fell from 42% to 22%.

What the gate does **not** settle:

- **Packing makes inter-token latency worse at c=128** (P95 53.7 → 63.0 ms): one packed forward stalls decode longer than one single-prompt forward. On the 0.5B model that is invisible next to the launch overhead.
  On 7B prefill is compute-bound, and an order-of-magnitude estimate (15 GFLOP per token, 50–80 TFLOP/s effective in bf16 on a 4090; **an assumption, not a measurement**) puts a 2,048-token pack near 0.4–0.6 s
  of stall and a 512-token pack near 0.1–0.15 s, against a 150 ms inter-token target. The 7B session therefore overrides the pack budget to **512**. That value is unmeasured on any GPU; the engine-profile table (ms per prefill
  chunk) and the operating-point inter-token latency will show whether it was right.
- **Invariance is at the gate's tolerance edge.** Unpacked 31/32 (request 21, token 7: the same request and token as the first session, so it is deterministic); packed 30/32 (requests 21 and 22, both token 7). 0.9375 equals
  the rule's floor (0.96875 − 1/32). Passing is correct by the rule fixed in advance; it is not comfortable.
- **The probe did not explain those divergences.** It ran with packing on and all 32 prompts inserted at once, so its batch shapes differ from the HTTP run's; it found a different request (index 2, first token,
  fp16 logit gaps of 1 ulp and exactly 0, a genuine tie) and none of 21 or 22. That is a real near-tie, but it is not evidence about 21 or 22. The acceptance check now requires every diverged request in the invariance run to
  be reproduced by a probe run **with the same packing setting**; until then the item is INCONCLUSIVE (it is, for both gate stages). The unpacked HTTP path is deterministic and equals the probe's order, so
  `PREFILL_PACK_TOKENS=0 scale_session.sh probe-0.5b` should reproduce request 21; it is the first stage of the 7B session.
- **The Pod's image reference and Pod id were not recorded**: the entrypoint forwards only a fixed allow-list of variables to SSH sessions, so `TENSORPILOT_IMAGE_REF` (and `RUNPOD_POD_ID`) set in the Runpod
  console never reach `stage-info.json`. Export them in the SSH session before the first stage. The GPU UUID, which proves the physical card, was recorded.

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
UNPACKED="$(ls -d "$TENSORPILOT_V10_SESSION"/0.5b-quick-batch-2*/ | tail -1)"
PACKED="$(ls -d "$TENSORPILOT_V10_SESSION"/0.5b-quick-batch-packed-*/ | tail -1)"
python -m tensorpilot.packing_gate "$UNPACKED" "$PACKED" --output "$TENSORPILOT_V10_SESSION/packing-verdict.json"   # USE_PACKING or not
python3 -c "import glob,json; print(json.load(open(glob.glob('$TENSORPILOT_V10_SESSION/probe-0.5b-2*/divergence-probe.json')[0]))['meaning'])"
```

**D. 7B session, one gated script** (`deploy/runpod/run_7b_session.sh`, also on the Pod under `/opt/tensorpilot/deploy/runpod/` once the image contains it; until then paste it). Export the variables its header lists, then run it with `pipefail` so a failure survives `tee`:

```bash
set -o pipefail
bash ~/run7b.sh 2>&1 | tee ~/run7b.log ; echo "exit status: ${PIPESTATUS[0]}"
```

The script (a test asserts this copy equals the file):

```bash
#!/usr/bin/env bash
# Run the whole 7B BF16 production session as one gated chain, on the Pod, as user tensorpilot, inside tmux.
#
#   set -o pipefail
#   bash run_7b_session.sh 2>&1 | tee ~/run7b.log ; echo "exit status: ${PIPESTATUS[0]}"
#
# Order: probe-7b (mandatory correctness gate) -> 7b-bf16-prod-serial -> 7b-bf16-prod -> faults-7b -> vLLM install -> vllm-7b-bf16-prod
# -> quality-7b -> (optional) the unpacked 0.5B divergence probe, only if the ledger budget still allows it.
#
# It stops at the first failure of anything: a stage, the probe gate, the vLLM installation or a check below. Every stage archives its own
# evidence whether it passes or fails, so a stop loses nothing. THE POD KEEPS BILLING until you terminate it.
#
# Environment it needs. You export: TENSORPILOT_API_KEY, TENSORPILOT_IMAGE_REF, RUNPOD_POD_ID, SESSION_BUDGET_MINUTES (250 to 260),
# PREFILL_PACK_TOKENS (512); PRICE_PER_HOUR_USD is optional (cost lines). The login profile gives it: PATH (scale_session.sh,
# install_vllm.sh), TENSORPILOT_POD_MAX_MINUTES (330 or more) and TENSORPILOT_RESULTS_ROOT. Nothing else is read.
set -Eeuo pipefail

EXPECTED_DIGEST="sha256:145879a6fac04f1a673fc18885aec082942c15245e497fe7ec82e4bf953488cc"
STAGE="startup"
trap 'echo >&2; echo "*** STOPPED at: ${STAGE}. Evidence for every stage that ran, including this one, is archived under ${SESSION:-the session directory}." >&2; echo "*** THE POD IS STILL BILLING: copy the evidence, then TERMINATE the Pod." >&2' ERR

# ----- checks the stage runner does not make (it checks the GPU, disk, lifetime and budget per stage, and the vLLM driver only
# ----- when the vLLM stage starts, about two hours in; this fails early instead) -----
fail() { echo "refused: $*" >&2; exit 2; }
[[ -z "${V10_TEST_MODE:-}" ]] || fail "V10_TEST_MODE is set; that mode skips GPU checks and its results are not evidence"
for name in TENSORPILOT_API_KEY RUNPOD_POD_ID; do [[ -n "${!name:-}" ]] || fail "$name is not exported"; done
[[ "${TENSORPILOT_IMAGE_REF:-}" == *"@${EXPECTED_DIGEST}" ]] || fail "TENSORPILOT_IMAGE_REF must end in @${EXPECTED_DIGEST}"
[[ "${PREFILL_PACK_TOKENS:-}" == 512 ]] || fail "PREFILL_PACK_TOKENS must be 512 for this session"
budget="${SESSION_BUDGET_MINUTES:-}"; pod_max="${TENSORPILOT_POD_MAX_MINUTES:-}"
[[ "$budget" =~ ^[0-9]+$ ]] && (( budget >= 250 && budget <= 260 )) ||
  fail "SESSION_BUDGET_MINUTES must be 250 to 260 (225 would have the ledger refuse the vLLM stage; more than 260 spends past the plan)"
[[ "$pod_max" =~ ^[0-9]+$ ]] && (( pod_max >= 330 )) ||
  fail "TENSORPILOT_POD_MAX_MINUTES must be at least 330 in the Pod's environment variables"
command -v scale_session.sh >/dev/null || fail "scale_session.sh is not on PATH (log in again so the profile is read)"
command -v install_vllm.sh >/dev/null || fail "install_vllm.sh is not on PATH (log in again so the profile is read)"
driver="$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
echo "GPU: ${driver}"
major="${driver##*, }"; major="${major%%.*}"
[[ "$major" =~ ^[0-9]+$ ]] && (( major >= 580 )) || fail "the driver must be 580 or newer for vLLM (this host: ${driver})"

export SESSION_BUDGET_MINUTES PREFILL_PACK_TOKENS TENSORPILOT_API_KEY
SESSION="${TENSORPILOT_V10_SESSION:-${TENSORPILOT_RESULTS_ROOT:-/workspace/results}/v10-session}"
export TENSORPILOT_V10_SESSION="$SESSION"

run() { STAGE="$1"; echo; echo "=== $(date -u +%H:%M:%S) stage $1"; scale_session.sh "$1"; }

ledger_used() {
  python3 - "$SESSION/ledger.jsonl" <<'PY'
import json, sys
print(round(sum(json.loads(l)["elapsed_min"] for l in open(sys.argv[1]) if l.strip()), 1))
PY
}

# ----- 1. the mandatory correctness gate -----
run probe-7b
STAGE="probe-7b gate"
python3 - "$SESSION" <<'PY'
import glob, json, sys
r = json.load(open(sorted(glob.glob(sys.argv[1] + "/probe-7b-2*/divergence-probe.json"))[-1]))
s = r["settings"]
ok = (r["passed"] and r["unexplained"] == 0 and r["prompts"] == 32 and s["granted_capacity"] == s["capacity"] == 32
      and s["prefill_pack_tokens"] == 512)
print(f"probe-7b: {r['exact_matches']}/32 identical, {r['near_tie']} near-tie, {r['unexplained']} unexplained, "
      f"{s['granted_capacity']}/{s['capacity']} slots, packing {s['prefill_pack_tokens']}: GATE {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
PY

# ----- 2-4. serial baseline; continuous batching (its memory gate must grant all 48 slots); crash recovery -----
run 7b-bf16-prod-serial
run 7b-bf16-prod
run faults-7b

# ----- 5. vLLM: installed into its own virtualenv (the 7B weights are already in the shared cache, so nothing is downloaded twice) -----
STAGE="vLLM plan"
install_vllm.sh --plan
STAGE="vLLM installation"
install_vllm.sh
run vllm-7b-bf16-prod

# ----- 6. quality -----
run quality-7b

# ----- 7. optional: the unpacked 0.5B probe that explains the gate-session divergence (about 2 minutes), only if the ledger allows -----
used="$(ledger_used)"
if python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) + 12 <= float(sys.argv[2]) else 1)" "$used" "$SESSION_BUDGET_MINUTES"; then
  STAGE="probe-0.5b (optional)"
  echo; echo "=== optional unpacked 0.5B probe (ledger ${used} of ${SESSION_BUDGET_MINUTES} minutes used)"
  PREFILL_PACK_TOKENS=0 scale_session.sh probe-0.5b
else
  echo; echo "optional unpacked 0.5B probe skipped: ${used} of ${SESSION_BUDGET_MINUTES} ledger minutes used, its 12-minute cap would not fit"
fi

STAGE="finished"
echo; echo "=== ALL STAGES DONE ($(date -u +%H:%M:%S))"
cat "$SESSION/ledger.jsonl"
ls -lh "$SESSION"/*.tar.gz
echo "*** COPY THE EVIDENCE TO YOUR MAC, THEN TERMINATE THE POD (terminating, not stopping, ends billing)."
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
