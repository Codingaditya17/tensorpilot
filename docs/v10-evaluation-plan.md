# v1.0 GPU scale and reliability evaluation: plan

Nothing in this plan has been run on a GPU. Every number here is either a computed estimate (labeled) or a stated assumption
(labeled). The experiment hard caps, not the estimates, bound the cost.

## The questions and the experiments that answer them

| # | Question | Experiments | Stage(s) |
|---|---|---|---|
| 1 | How does it behave with 1, 8, 16, 32, 64 and 128 clients? | `sweep-c{1,8,16,32,64,128}`, closed loop, mixed lengths | `*-batch`, `vllm-*` |
| 2 | Maximum sustainable rate under a latency and error objective? | `ladder-poisson` (rising Poisson arrivals), `bursty` | same |
| 3 | Does continuous batching beat the serial worker? | `serial-c1`, `serial-c8` against `sweep-c1`, `sweep-c8` | `0.5b-serial`/`0.5b-batch`, `7b-bf16-serial`/`7b-bf16-batch` |
| 4 | How does it change from 0.5B to 7B to 14B? | the same sweep per model | `0.5b-batch`, `7b-bf16-batch`, `14b-nf4-batch` |
| 5 | What happens under memory, queue and deadline stress? | `overload`, `recovery-after-overload`, `timeouts`, `long-c*` | `7b-bf16-batch`, `*-long`, `14b-nf4-batch` |
| 6 | Does it survive a worker crash with its accounting and authentication intact? | `faults-*`: SIGKILL under load, then an audit of the request journal | `faults-0.5b`, `faults-7b` |
| 7 | How does it compare with vLLM on the same card? | identical workloads against vLLM | `vllm-7b-bf16`, `vllm-14b-nf4` |
| – | Is batched generation correct? | `batch_invariance` (solo vs together vs staggered), `quality` capture and compare | batching stages, `quality-*` |
| – | What does the telemetry itself cost? | `overhead-telemetry-on/off`, identical workload | `7b-bf16-batch`, `0.5b-batch` |
| – | Cold start? | `cold-start`: the first requests after readiness, before any warmup | every stage |

Cancellation while queued and while decoding is the `cancellation` experiment (30% of clients abandon their request, by token
count or by time). The brief's "control-plane or data-plane disruption" is covered by tests (authentication, redirect refusal,
host allow-listing, worker credential, crash recovery) rather than a GPU stage: it does not depend on the GPU, so it is not a
good use of rented time.

## Models

Chosen for being openly licensed, reproducible at a pinned commit, and the same architecture family so the engine's allowlist applies.
All Apache-2.0. Estimates come from the checkpoint's `config.json` and file sizes (`benchmarks/models/model-matrix-rtx4090.json`).

| Role | Model @ revision | Precision | Weights resident | KV per token | Slots at 1,536-token rows | Verdict (estimated) |
|---|---|---|---|---|---|---|
| Regression and performance baseline | Qwen2.5-0.5B-Instruct @ 7ae5576… | fp16 | 0.92 GiB | 12 KiB | 128 | fits |
| **Primary larger model** | **Qwen2.5-7B-Instruct @ a09a354…** | **bf16** | **14.19 GiB** | **56 KiB** | **64 (limit: KV budget 5.5 GiB)** | fits, with about 0.15 GiB to spare at 64 slots |
| **Stretch model** | **Qwen2.5-14B-Instruct @ cf98f3b…** | **NF4** | **9.25 GiB** (27.51 GiB download) | **192 KiB** | **34 (limit: VRAM)** | fits, with fewer slots |
| (reference only) | Qwen2.5-14B-Instruct | bf16 | 27.51 GiB | 192 KiB | 0 | **does not fit** (cap is 21.17 GiB) |
| (refused) | Qwen3-8B | any | – | – | – | unsupported architecture (the forward pass does not implement Qwen3's QK-norm) |

Why 7B bf16 is the primary: it is the largest model that fits at full precision, so it isolates batching from quantization. Why
14B NF4 is still included: you asked for it, and it is the only way to exercise a model that exceeds the card.

**What 14B NF4 can and cannot do on 24 GB (estimated):** 105 concurrent sequences at 512-token context, 52 at 1,024, 25 at 2,048, 12 at
4,096 and 5 at 8,192. It cannot hold 128 concurrent sequences at any of those lengths, so its 128-client point is queue-bound by
construction, and a 32K-context row (6 GiB) does not fit beside its weights at all. The 7B bf16 holds 99 sequences at 1,024 tokens,
49 at 2,048 and 24 at 4,096. These are estimates until the first load reports `cuda_peak_reserved_bytes`.

Long-context prompts are capped at **2,048 tokens**, the most the service accepts through the word-built prompt generator (its
16,384-byte prompt limit). The suite refuses to generate a longer prompt; an earlier draft declared 4,096 and would have silently
produced 2,700.

## Stages, expected time and hard caps

Expected minutes come from a stated model: single-stream decode rate × a batching-efficiency exponent of 0.8. The 0.5B rate (58 tok/s)
was measured; **the 7B (40) and 14B (25) rates are assumptions.** The cap per experiment is 2.5× its expectation (plus two minutes).

| Stage | Experiments | Expected min | Hard cap min |
|---|---|---|---|
| `0.5b-serial` | 3 | 5.0 | 21 |
| `0.5b-batch` | 15 | 32.6 | 124 |
| `7b-bf16-serial` | 3 | 5.5 | 22 |
| `7b-bf16-batch` | 15 | 47.7 | 160 |
| `7b-bf16-long` | 5 | 12.6 | 44 |
| `14b-nf4-batch` | 13 | 71.7 | 214 |
| `14b-nf4-long` | 4 | 14.4 | 47 |
| `vllm-7b-bf16` | 9 | 36.8 | 115 |
| `vllm-14b-nf4` | 9 | 62.8 | 179 |
| `quality-7b` / `quality-14b` | – | 14 / 12 | 40 / 45 |
| `faults-0.5b` / `faults-7b` | – | 5 / 12 | 20 / 50 |
| **Total** | | **332** | **1,081** |

Regenerate with `python -m tensorpilot.suite --plan`. The batching stages include a ten-minute batch-invariance check inside their caps.

## Cost (assumption-based; confirm the price when you rent)

At the roughly $0.90/hour you quoted for an RTX 4090 Pod (not a price recorded anywhere in this repository), in two sessions. Each vLLM stage runs in the **same Pod as
its TensorPilot counterpart**, so the engine comparison is made on one physical card; vLLM is installed into an isolated virtualenv inside the session (about 12 minutes,
repeated in session B because a Pod's volume does not survive termination).

| Session | Stages | Stage minutes (expected) | Overhead assumed | Expected | Recommended `SESSION_BUDGET_MINUTES` | Bounded worst case |
|---|---|---|---|---|---|---|
| A | `0.5b-serial`, `0.5b-batch`, `7b-bf16-serial`, `7b-bf16-batch`, `quality-7b`, `faults-0.5b`, `faults-7b`, then `vllm-7b-bf16` | 159 | 63 min (boot, 7B download, eight loads, vLLM install, export) | 3.7 h, **$3.3** | 360 | ≤ 7.4 h, **$6.7** |
| B | `7b-bf16-long`, `14b-nf4-batch`, `14b-nf4-long`, `quality-14b`, then `vllm-14b-nf4` | 174 | 87 min (7B and 14B downloads, slow 14B loads, vLLM install, export) | 4.3 h, **$3.9** | 360 | ≤ 7.8 h, **$7.0** |
| | | | | **8.0 h, about $7.2** | | **≤ 15.2 h, about $13.7** |

The ledger refuses a stage if the stage time already used plus that stage's hard cap would exceed the session budget, so stage time cannot exceed the budget. It does **not**
bound time outside stages (boot, downloads, idle, SSH). The overhead column is an assumption; the worst-case column adds that overhead and 20 minutes of slack to the budget.
**Stopping or exiting the engine does not stop billing: terminate the Pod.**

If that is too much, the cheapest high-value subset is session A's `0.5b-serial`, `0.5b-batch` and `7b-bf16-batch` (about 85 expected minutes of stage time): it answers whether batching is
correct and faster on a real GPU before anything larger is attempted.

**Disk:** the model cache holds 0.9 + 14.2 + 27.5 GiB, the vLLM environment about 15 GiB, and the image about 7 GiB: a 120 GB volume and a 50 GB container disk.
RAM of at least 32 GB is recommended for loading the 14B shards.

## Metric definitions

- **Wall-clock output tokens/s:** completed output tokens divided by the wall time of the whole run, including ramp-up and drain.
- **Active-window tokens/s:** completed output tokens divided by the time from the first token produced to the last. Excludes the ramp and tail.
- **Active decode tokens/s (GPU):** decode tokens divided by synchronized decode execution seconds, from the worker's own counters over the window.
  Excludes idle time and prefill. The only measure that approximates what the GPU does while decoding.
- **TTFT:** request dispatch to the first output token the client receives (includes queue wait and prefill).
- **TPOT:** a request's mean inter-token gap. **ITL:** pooled per-token gaps.
- **Goodput:** completed requests (and their tokens) per second that met **every** declared threshold (TTFT, end-to-end, TPOT). A request that completed
  but was too slow is throughput, not goodput.
- **SLO attainment:** the fraction of non-cancelled requests that met the thresholds.
- **Maximum sustainable rate:** the highest offered rate such that it and every lower rate had attainment ≥ 95% and error rate ≤ 1%.
- **Outcomes:** each request ends as exactly one of completed, rejected, timed_out, failed or client_cancelled. A deliberate client cancellation is not an error.

SLO thresholds are **declared** (0.5B: TTFT 1 s, end-to-end 10 s, TPOT 50 ms; 7B: 2 s, 20 s, 150 ms; 14B: 4 s, 40 s, 350 ms), not derived from user requirements.
Percentiles are computed per repeat and reported as the median of repeats with the range; a point whose smallest repeat completed fewer than 100 requests is
labeled small-sample, and the c=1 points are small by design.

## The vLLM comparison: what is matched and what is not

Matched: the model and exact revision, the workload specs and seeds, request counts, arrival rates, warmup policy, timeouts, SLOs, the load generator, the metric
definitions, greedy sampling, and `max_model_len` equal to TensorPilot's row length. Run sequentially on the same GPU type, never together.

Not matched, and recorded rather than hidden:

- **Different CUDA stacks:** TensorPilot runs torch 2.9.1 on CUDA 12.8; vLLM 0.31.0 runs torch 2.13.0 on CUDA 13.0 in its own virtualenv, and needs driver ≥ 580.
- **Different batching:** vLLM pages its KV cache and can hold more sequences than TensorPilot's fixed slots, and uses CUDA graphs; TensorPilot does neither.
- **Same physical card, different times:** each vLLM stage runs in the same Pod as its TensorPilot stage, after it. Clock, temperature and co-tenancy drift between the two
  are recorded by telemetry but not controlled. (Comparisons across Pods, such as 7B in session A against 14B in session B, are not controlled for the card.)
- **Token accounting:** vLLM's token arrivals are server-sent-event chunks and its token count is the server's usage figure; TensorPilot emits one event per token.
- **Sampling:** vLLM would apply the model repository's `generation_config.json` (Qwen2.5 ships `repetition_penalty` 1.05, `top_k` 20, `top_p` 0.8) even at temperature 0. The client sends every
  override explicitly and the server is started with `--generation-config vllm`; the stage refuses to run if the installed vLLM lacks that flag.
- **14B quantization:** vLLM loads bitsandbytes NF4 through its own implementation: the same format, not the same code.

It is acceptable, and expected, for TensorPilot to be slower. The comparison is an engineering baseline for locating bottlenecks.

## Fault and security validation

- **Crash:** SIGKILL one supervised worker mid-load; audit the request journal (one record per request, one terminal outcome each, nothing completed across the crash
  or during the outage, exact token counts, requests after recovery complete), re-check authentication after the restart, and verify no process is left behind.
- **Authentication and forwarding (tested on CPU, over real HTTP):** invalid client key, missing or wrong internal worker credential, worker-host allow-list, redirect refusal on
  all three credentialed calls, secret redaction and refusal to archive evidence containing a key.
- **Overload:** arrivals far above capacity; rejection is expected and measured; the worker must serve normally again afterwards (`recovery-after-overload`).
- **Never forced:** no experiment tries to cause an out-of-memory error. Admission and the fixed pool bound KV memory; an OOM event during any window aborts the experiment.

## Stop conditions

Each experiment has its own time cap and an abort rule (error rate above its limit, an OOM event, or the worker not ready afterwards). Each stage has a hard cap with a graceful
stop one minute earlier. A failed preflight, a failed readiness gate or a failed smoke test stops the stage before any stress testing. Evidence is archived and the ledger written
however the stage ends.

## Recommendation

**GO for a first, bounded GPU session after your approval, and NO-GO for running everything at once.** The code is as verified as it can be without a GPU
(tests locally and inside the Linux image, mutation-tested guards), but it has never executed on CUDA, so the first stage is the cheapest one that tests the riskiest assumption.
Start with `0.5b-serial` (regression against the immutable baseline), then `0.5b-batch` (is batching correct and faster on a real GPU?). Proceed to 7B only if those pass, and to 14B and
vLLM only after 7B behaves. Publishing the experiment image and renting hardware each need your explicit approval.

## Known limitations

- Nothing has run on a GPU; bf16/fp16 numerics, CUDA kernels, bitsandbytes' CUDA path and real memory use are all unverified.
- Time and cost estimates rest on assumed single-stream rates for 7B and 14B and an assumed batching exponent.
- The batching engine's Python event loop and per-token JSON streaming may bottleneck at high batch width before the GPU does; the telemetry is designed to show it, but it is a hypothesis.
- 14B has no bf16 reference on this card; its fidelity is bounded by the 7B comparison and by literature.
- A single repeat or a small request count gives wide percentile uncertainty; the report labels it rather than hiding it.
