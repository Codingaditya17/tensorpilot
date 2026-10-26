# v1.0 architecture: continuous batching and NF4 quantization on CUDA

Status: implemented and tested on CPU and inside the Linux image. **Not yet run on a GPU.** Every performance statement
below is a design reason or a hypothesis, not a measurement. See [`v10-evaluation-plan.md`](v10-evaluation-plan.md) for how
it will be measured.

## What changed, in one paragraph

The CUDA worker used to run one sequence at a time. It now has an optional continuous-batching mode: every decode step is one
forward pass over all active sequences, sequences join and leave between steps, and a finished request frees its slot for the
next waiting one. A new explicit `--quantization nf4` mode loads weights as 4-bit NormalFloat so a 14B model can fit a 24 GB
card. The serial backend is unchanged and remains selectable, so the two can be A/B tested.

## Request path

```
client ──HTTP──▶ data plane ──▶ control plane (placement) ──▶ worker
                  (auth, forward)   (capacity-aware routing)     │
                                                                 ▼
                                                     BatchEngine  (queue, deadlines, cancellation,
                                                          │        per-sequence bounded buffers, health)
                                                          ▼
                                                 CUDABatchAdapter  (model thread only)
                                                   ├─ prefill queue ──▶ chunked prefill on the staging row
                                                   ├─ decode slots  ──▶ one batched forward per step
                                                   └─ KV slot pool  ──▶ fixed, preallocated, in-place writes
```

`BatchEngine` was already adapter-driven for MLX. The CUDA work is a new adapter behind the same four calls
(`insert`, `step`, `remove`, `close`), plus three small generalizations in the engine: it honors the adapter's own width limit
(previously a hard-coded 8), it adopts the capacity the adapter actually grants, and it takes an admission budget sized in real
bytes.

## The batched forward pass (`tensorpilot/cuda_batch.py`)

Written for the Qwen2 and Llama decoder layouts only; any other architecture (Qwen3, Gemma, MoE, sliding-window layers) is refused at
construction instead of being assumed to work.

- **Fixed slot pool.** `capacity + 1` rows of `row_tokens` KV positions per layer, allocated once at startup and zero-filled
  (a masked column is multiplied by probability 0, and `0 × NaN` would poison the output if fresh memory held NaN bit patterns).
  KV memory is therefore known in advance: there is no per-step allocation, no `torch.cat`, no padding shuffle. It is **not
  paged**: every active sequence reserves a whole row, so short sequences waste the unused tail of theirs.
- **Per-row positions.** Each decode row has its own length. New keys and values are written at that row's own position with
  one advanced-index assignment; the attention mask is `column <= length` per row, so rows of different lengths share a step.
- **Compact decode rows.** Live rows stay in `[0, n)`. When sequences finish, surviving tail rows move into the holes (only
  their filled prefix is copied), so a step reads a contiguous slice instead of gathering.
- **Staging row.** Prefill runs on a dedicated extra row; the finished row is copied into a decode slot. Waiting sequences hold
  no KV until their turn.
- **Chunked prefill, interleaved.** A prompt is processed in `prefill_step_size` chunks, and prefill and decode steps alternate while
  both have work, so a long prompt delays running streams by one chunk, not by its whole length.
- **Grouped-query attention without copying K/V.** The query-head group is folded into the query-length dimension, so scaled
  dot-product attention reads each KV head once. (Transformers' default path expands K/V by the group factor, 7× for
  Qwen2.5-7B.)
- **One host synchronization per step.** Greedy `argmax` for the whole batch, one `.tolist()`.
- **Incremental detokenization.** Only a short window of token ids is decoded per token, with incomplete UTF-8 held back
  (tested with emoji and CJK on the real Qwen tokenizer), instead of re-decoding the whole output each step.

### Decisions and what they cost

| Decision | Why | Cost |
|---|---|---|
| Own forward pass instead of Transformers' cache | per-row positions, in-place KV, no K/V expansion, explicit architecture allowlist | duplicates model code for two layouts; each new architecture needs its own verification |
| Fixed slot pool instead of paged KV | simple, verifiable on CPU, fails at startup not mid-run | wastes the tail of every row; concurrency is bounded by slots, not by tokens in use |
| Fail the whole worker on a step error | a half-failed shared batch cannot be trusted; the supervisor restarts it | an OOM in a step fails every in-flight request (requests are never silently retried) |
| Prefill and decode alternate, not mixed in one pass | keeps phase timing separable and the code small | decode throughput halves while a prefill is pending |
| Admission sized from the real KV bytes per token | the old flat 16 KiB/token guess is 3.5× off for 14B | admission is still an estimate; real memory is reported separately |

### Not implemented, and not claimed

Paged KV cache, prefix caching, CUDA graphs, speculative decoding, FlashAttention-style custom kernels, tensor parallelism,
quantized KV cache, any quantization other than NF4. Each is a plausible next step; none exists. CUDA graphs and paged attention
in particular are what a production engine such as vLLM uses to avoid the per-step Python and kernel-launch overhead and the
row-tail waste, so a gap to vLLM is expected and is the point of the comparison.

## NF4 quantization (`tensorpilot/quantization.py`)

- **An explicit mode, never a fallback.** `--quantization nf4` requires `bitsandbytes` and `accelerate`; if either is missing or
  cannot run on the host, loading fails with the reason and no model is loaded in any precision. A test asserts the model loader
  is never called in that case.
- **Fixed recipe** so evidence is comparable: NF4, double quantization, compute dtype equal to `--dtype` (bf16 or fp16). Weights
  are quantized from the original checkpoint at load time, shard by shard, straight onto the device.
- **What is quantized:** the attention and MLP projections (7 per layer). The embedding, `lm_head`, norms and biases stay in the
  compute dtype. For 14B that is about 2.9 GiB of the 9.25 GiB resident total, so the saving is about 3×, not 4×.
- **Honest accounting.** `parameters()` undercounts a quantized model because bitsandbytes keeps its scale tables elsewhere; the
  backend reports the true resident bytes. An analytic estimator (used by the planner and by preflight) matches a real load of
  Qwen2.5-0.5B within 0.04% and matches the published bf16 sizes of 7B and 14B within 0.5%.
- **KV cache is unchanged by quantization**, which is why 14B at 192 KiB/token leaves far fewer slots than 7B at 56 KiB/token.

### Lower memory is not higher speed

A bitsandbytes 4-bit matmul with more than one token per step dequantizes each weight matrix to the compute dtype before
multiplying. For a batched decode step that adds memory traffic bf16 does not pay: it writes and re-reads a bf16 copy of every
weight each step. Whether NF4 is slower or faster than bf16 at a given batch width is a measurement the plan includes; the
mechanism above is a hypothesis for why it may be slower. NF4 also does not make 14B immune to running out of room: at 8,192-token
context only about 5 sequences fit, and 128 concurrent sequences do not fit at any tested context length.

### Fidelity

There are three different questions, and only two have a reference on a 24 GB card (`tensorpilot/quality.py`):

1. *Is the engine correct on these weights?* Batched must equal serial on the same weights. Tested on CPU for NF4 and 16-bit,
   for both layouts, down to per-step logits against the Transformers forward pass.
2. *How much does NF4 change a model whose bf16 version fits?* Measured on the 7B pair (teacher-forced perplexity ratio, top-1
   agreement, top-5 overlap, KL on the reference's top-k support, greedy prefix agreement).
3. *How good is 14B NF4?* It cannot be compared against its own bf16 weights here. Only absolute numbers are available, and the
   7B comparison is the nearest evidence about NF4 fidelity, not a substitute. The first CPU check of Qwen2.5-0.5B already
   shows a 0.5B model changes its top-1 prediction under NF4; 14B is expected to fare much better, but that is literature, not a
   measurement of this stack.

## Observability added

Continuous, timestamped GPU sampling (utilization, memory, power, temperature, clocks, throttle reasons) and server sampling
(queue depth, batch width, KV pool, allocator, OOM counter, decode counters) with per-experiment window statistics, boundary
samples so a short window is never empty, and the sampler's own call counts. An **active decode rate** (decode tokens per
synchronized decode second) is reported separately from wall-clock throughput, which includes idle time and prefill.
Telemetry overhead is measured by an on/off A/B experiment, not assumed.

## Defects found while building this (all fixed, each with a regression test)

| Found by | Defect |
|---|---|
| Mutation testing | the first batching tests did not exercise EOS; a mutation counting EOS as output survived until a deterministic EOS test was added |
| Real PyTorch validator | `set_per_process_memory_fraction` rejects a bare `cuda` device (found on the first RTX 4090 run; reproduced on CPU) |
| First RTX 4090 gateway run | the data plane never sent the worker credential, so every gateway request was a relayed 401 |
| Tests against a redirecting worker | services followed worker redirects to hosts never checked against the allow-list |
| A flaky full-suite run | a telemetry window shorter than the sampling interval held zero samples |
| Container test under emulation | a handler timeout firing before the job's deadline recorded the same expiry as `cancelled` or `timed_out` depending on timing |
| Load-generator test | a pooled keep-alive connection reused after the server force-closed it failed an unrelated request |
| Suite validation | the long-context stage declared 4,096-token prompts while the generator silently clamped to 2,700 |
| Looking at a rendered chart | coarse axis ticks and a subtitle referring to a line that was not drawn |
| Report test | small-sample flagging used the pooled count although percentiles are per repeat |

## What is verified where

| | CPU (tiny models) | Linux image (emulated amd64) | Real GPU |
|---|---|---|---|
| Batched equals serial, per-step logits equal Transformers, joins, removal, EOS, chunked prefill | yes | yes | **no** |
| NF4 load, accounting, estimator, fidelity tool | yes (bitsandbytes CPU backend) | yes | **no** |
| HTTP concurrency, cancellation, deadlines, overload, failure handling | yes | yes | **no** |
| Worker crash and recovery audit (real process killed) | yes | yes | **no** |
| CUDA kernels, bf16/fp16 numerics, VRAM use, NF4 kernels | n/a | n/a | **no** |
| Throughput, latency, goodput, GPU utilization, vLLM comparison | n/a | n/a | **no** |
