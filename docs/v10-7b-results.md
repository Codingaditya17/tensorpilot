# 7B BF16 session results (rc2-3), the vLLM failure, and the re-run plan

Evidence: [`benchmarks/baselines/v1.0.0-rc2-3-7b-rtx4090/`](../benchmarks/baselines/v1.0.0-rc2-3-7b-rtx4090/) (Pod `wydrcdr7xnqbwf`, one RTX 4090,
driver 595.91.07, image `cuda-experiment@sha256:7e65d7ca…c446e56`, 2026-10-27 UTC). Five archives, hashes below. Every number in this document is
read from them by `tensorpilot.report_v10` / `tensorpilot.acceptance` or by the one-off derivations marked *derived*. Nothing here compares
TensorPilot with vLLM, because no vLLM measurement exists.

| archive | SHA-256 | bytes |
|---|---|---|
| `probe-7b-20261026T235916Z` | `f1bb16882eacb13d974c07ba9b873c990665de544cb7660ce1a091a5acc58c4d` | 2,265 |
| `7b-bf16-prod-serial-20261027T000948Z` | `212228521371edf310b51e675770b1f81f0bede7d621d3fea5963f990e1b1f74` | 3,129,096 |
| `7b-bf16-prod-20261027T001850Z` | `ce682b9335014ac8cf3e7339e95881813173e8749856d7707f30be911dc25a0c` | 161,839,424 |
| `faults-7b-20261027T010216Z` | `2d0256805f0f7d22e9f6aaff2c7129a8e9f780c7b3c8eb2af3b3db12be8fad78` | 1,768,954 |
| `vllm-7b-bf16-prod-20261027T012546Z` | `43486b8a57eb989a82202e938a20a895d2019de9b715b89a29f0bd17c82e9709` | 50,815 |

## 1. What the session established

| gate | result |
|---|---|
| probe-7b (rc2-3's fixed diagnostic) | 20/32 identical, 12 near-tie, **0 unexplained**; 32/32 slots, packing 512; 118 of 1,488 solo steps within one ulp of a tie |
| batch invariance in the stage (`invariance.json`) | 20/32 exact together-vs-solo and staggered-vs-solo; every diverged request is one the probe explains as a near-tie |
| memory gate | 48 slots granted; 2.76 GiB free of a 21.17 GiB allocator cap at peak reserved 18.40 GiB |
| crash recovery | worker killed at 25.0 s; ready again 16.3 s later; 693 requests dispatched after recovery, all completed (the stage summary counts 691 with a slightly different cut-off) |
| serial and batching stages | every planned experiment completed |
| vLLM stage | **failed before serving a request** (section 4) |
| quality-7b | not run (the chain stops at the first failure) |

Automated verdict on this evidence (`acceptance-verdict.md`): **evaluation FAIL**, caused by one item, `3.overload`; the rest are PASS or INCONCLUSIVE
for missing evidence (14B, vLLM, quality, the 0.5B probe, ledger and billing). It is not a pass, and it is not a statement about vLLM.

## 2. Serial against continuous batching, matched workloads (7B BF16, one RTX 4090)

Medians over three repeats (short prompts; the serial stage has no c128). TTFT and inter-token values are client-side p95.

| clients | engine | tok/s | TTFT p95 | inter-token p95 | e2e p95 | errors | GPU util | GPU mem |
|---|---|---|---|---|---|---|---|---|
| 1 | serial | 53.9 | 0.023 s | 19 ms | 1.15 s | 0 | 93% | 15,228 MiB |
| 1 | batch | 53.4 (**0.99×**) | 0.024 s | 19 ms | 1.17 s | 0 | 95% | 19,416 MiB |
| 8 | serial | 53.8 | 6.18 s | 19 ms | 7.01 s | 0 | 93% | 15,248 MiB |
| 8 | batch | 321.7 (**5.98×**) | 0.063 s | 42 ms | 1.46 s | 0 | 94% | 19,416 MiB |
| 32 | serial | 53.7 | 24.9 s | 19 ms | 25.7 s | 0 | 93% | 15,248 MiB |
| 32 | batch | 832.7 (**15.5×**) | 0.218 s | 48 ms | 2.17 s | 0 | 91% | 19,436 MiB |

- The serial worker is flat at 53.8 tok/s at every concurrency: extra clients only queue (TTFT p95 6 s at c8, 25 s at c32, the whole queue wait).
- Single-stream decode is bandwidth-bound, not launch-bound as it was for 0.5B: 18.3 ms per step against 15.2 ms to read the 15.29 GB of weights once at the
  card's published 1,008 GB/s (*derived*: 83% of that floor), with 93–95% GPU utilization.
- Batching costs a single client about 1% (53.4 against 53.9 tok/s; the repeat ranges 53.3–53.5 and 53.7–54.3 do not overlap).
- Batching raises the p95 inter-token latency from 19 ms to 42 ms (c8) and 48 ms (c32), 2.2× and 2.5×, because a decode step now shares the GPU with prefill.
  Per-stream speed falls as throughput rises; at c32 each stream gets 26 tok/s against 54 for a lone one.

## 3. Scale, saturation and queueing (batching stage; mixed prompt and output lengths, 1,500–2,000 requests per point, one run each)

| clients | tok/s | TTFT p50 / p95 | e2e p95 | max queue | active | SLO attainment (TTFT ≤ 2 s) | errors |
|---|---|---|---|---|---|---|---|
| 1 | 52.1 | 0.023 / 0.088 s | 3.9 s | 0 | 1 | 100% | 0 |
| 32 | 822.9 | 0.075 / 0.329 s | 8.2 s | 3 | 32 | 100% | 0 |
| 64 | 1,006.4 | 1.43 / 1.91 s | 11.2 s | 16 | 48 | 97% | 0 |
| 128 | 1,005.5 | 6.75 / 7.74 s | 16.6 s | 80 | 48 | 2% | 0 |
| 256 | 1,012.4 | 17.3 / 19.0 s | 27.3 s | 208 | 48 | 1.5% | 0 |
| 512 | **1,013.6** | 37.8 / 40.3 s | 47.9 s | 464 | 48 | 1.5% | 0 |
| 1024 | 1,010.2 | 78.5 / 82.7 s | 89.3 s | 976 | 48 | 1.3% | 0 |

- **Peak: about 1,010 tok/s** (1,013.6 at c512 is the highest single run; c64 to c1024 agree within 0.8%, so the difference is noise).
  **Saturation: the 48-slot pool at about 64 clients.** c32 delivers 82% of peak with 32 active; c64 reaches 99%. Nothing after that adds throughput.
- **Queueing is exactly what a full pool predicts.** Past c64 the extra clients wait in the queue: TTFT p50 is the queue depth over the completion rate
  (c1024: 976 queued ÷ 11.8 requests/s ≈ 83 s, measured p50 78.5 s, p95 82.7 s). Throughput, step time (24 ms) and GPU memory (19.4 GiB) are the same at c64 and c1024.
- **What caps the plateau** (engine counters, *derived*): at saturation about 39% of engine time is prefill (347k prompt tokens in 63.7 s, 5.4k tokens/s) and the decode step
  is 24 ms for about 41 tokens, against 18 ms for one. Decode scales almost free of width; prefill is the compute-bound part. This is a statement about this workload mix, not a ceiling for the hardware.
- Open-loop (Poisson, mixed lengths, 240 requests per rate): the objective (≥95% SLO attainment, ≤1% errors) holds up to **12 requests/s** (766 tok/s, TTFT p95 0.43 s).
  At 16/s attainment is 81% and TTFT p95 2.4 s; at 24/s and 32/s it is 39% and 31%. Throughput flattens at about 855 tok/s there, below the closed-loop plateau,
  so 12/s is a measured, short-window figure (27 s runs), not a sustained rating. The only sustained evidence is `sustained-c32`: 3,000 of 3,000 requests, 839 tok/s, TTFT p95 0.30 s.
- GPU: 90–92% utilization, 338 W peak mean power, 69 °C maximum; the `sw_power_cap` throttle flag appeared in 24 reports and the lowest SM clock was 2,535 MHz against 2,640, a loss of at most 4%.
- Memory is flat from c1 to c1024 (19,416 to 19,452 MiB): the KV pool is fixed-slot and preallocated (4.0 GiB for 48 × 1,536 tokens), so concurrency cannot cause an out-of-memory. Zero OOM events in 3,095 samples.

## 4. Do the stress results support production-readiness claims?

**Supported, for one RTX 4090 and this model:**
- No request failed at any concurrency from 1 to 1,024 when the queue admits them all (about 20,000 requests overall), and memory did not move.
- 3,000-request sustained run at c32: 100% success.
- A SIGKILLed worker was serving again in 16 s, and everything dispatched afterwards completed.
- Continuous batching is 6× (c8) and 15.5× (c32) the serial worker at identical workloads.

**Not supported, and not claimable from this data:**
1. **Overload behavior (a real failure).** At 400 requests/s for 1,500 requests the bounded queue (1,100) never filled: the server accepted all 1,500 (peak queue 976), rejected none, and
   318 (21.2%) hit their 30 s deadline, 104 of them before a first token and the rest mid-answer, after 6,429 output tokens had already been spent on requests that then failed.
   The server has no load shedding that looks at expected wait; the queue bound protects memory, not latency. The deadline cut itself is also silent: the handler closes the stream at the
   deadline with no terminal event (`tensorpilot/server.py`, by design), so the client records `incomplete_stream` and can tell a timeout from a crash only by the elapsed time.
   Whether shedding or a terminal `deadline_exceeded` event is the right fix is a product decision; this report does not change the server.
2. **Latency at high concurrency.** 1,024 clients are served without error but with a 78 s median first-token wait. The usable range is up to about 48–64 concurrent streams, or about 12 requests/s.
3. **Failures during a crash are not hidden by the server**: 87 of 900 requests failed (5 in flight at the kill, 82 during the 16 s outage). There is no retry or standby, so this is a restart, not high availability.
4. **Output depends on batch composition.** 12 of 32 prompts produce a different continuation when batched than solo; all 12 are explained as bf16 near-ties (logit gap within the dtype's rounding), but a client
   cannot assume the same text for the same prompt under different load.
5. **Everything else not measured**: 14B, NF4 quality (the stage was skipped), a second GPU, prompts above 1,024 tokens, runs longer than a few minutes, 100,000 requests (20,568 were executed).
6. **Samples are small.** Scale points are single runs; only the comparison points have three repeats. Pod-to-Pod variance is unknown.

**Regressions.** There is no earlier 7B measurement to regress against. The within-session findings above are: c1 about 1% slower than serial, inter-token p95 2.2–2.5× higher under batching,
SLO attainment collapsing beyond about 64 clients, and the overload behavior. Against the 0.5B quick session: speedup at c8 is 5.98× on 7B against 5.42× on 0.5B.

## 5. Root cause of the vLLM failure

**Symptom.** `RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}`. This final line says nothing; the cause is earlier in `engine.log`.

**Earliest relevant error** (line 44, 01:29:05, 18 seconds after the 14.29 GiB of weights were on the GPU):

```
torch._inductor.exc.InductorError: RuntimeError: Failed to find C compiler. Please specify via CC environment variable or set triton.knobs.build.impl.
```

The stack: `_initialize_kv_caches → determine_available_memory → profile_run → _dummy_run → model forward → aot_compile (torch.compile) → Inductor → Triton
driver → CudaUtils() → compile_module_from_src → _build`. Triton builds a small C helper module with the system C compiler the first time it touches the driver; vLLM's
compilation path needs it during the memory-profiling pass, before the KV cache is allocated and before the server opens its port.

**Why the image has no compiler.** The experiment image is `python:3.11-slim-bookworm`-based. Checked in the published rc2-3 image: `cc`, `gcc`, `g++`, `clang`, `make` are absent, and so are `/usr/include/stdio.h`
and `/usr/lib/x86_64-linux-gnu/crt1.o` (`Python.h` is present). In that image, Triton 3.5.1's `CudaUtils()` fails with the same message; after `apt-get install gcc libc6-dev` the helper module builds.
TensorPilot never needed a compiler (no `torch.compile`, no Triton), which is why nothing in the 878-test image suite caught it.

**Ruled out** (each from the archive):
- *Driver or CUDA 13 mismatch*: driver 595.91.07 ≥ 580, `torch.cuda` initialized, the weights loaded to the GPU, FlashAttention 2 was selected.
- *Dependencies*: `vllm-env.txt` matches the hash-locked `requirements-vllm.txt` (vllm 0.31.0, torch 2.13.0, triton 3.7.1, transformers 5.17.0).
- *Memory*: no allocation error; it failed before the KV cache was sized, on a 24 GB card with 14.29 GiB used.
- *Download or revision*: `Loading weights took 1.87 seconds` from the shared cache; revision `a09a3545…` as in the TensorPilot stages.
- *Launch flags*: the logged non-default arguments are exactly `--model --revision --dtype bfloat16 --max-model-len 1536 --max-num-seqs 48 --gpu-memory-utilization 0.9 --generation-config vllm --download-dir`.

**Why it was found two hours in.** Nothing checked for the toolchain before the vLLM stage. The installer verified only that vllm and torch imported and that CUDA was visible.

**Not proven:** that vLLM then runs end to end. The first compile may meet a second environment gap (for example an FlashInfer JIT step, which needs `nvcc`; the lock includes `nvidia-cuda-nvcc` and `ninja`,
and greedy requests normally do not take that path). The new Triton kernel smoke covers Triton, not every JIT. The rerun stage is capped and stops on the first failure.

## 6. Changes made (local commits only; nothing pushed, no image published)

| change | where | test |
|---|---|---|
| install `gcc` and `libc6-dev` in the experiment image | `deploy/docker/Dockerfile.experiment` | `tests/test_vllm_toolchain.py::test_the_experiment_image_installs_a_c_compiler_and_libc_headers` |
| `check_toolchain.sh`: compile and link a Python-header shared object; `--gpu` also compiles and launches one Triton kernel (`triton_smoke.py`) | `deploy/runpod/` | `tests/test_vllm_toolchain.py` (missing compiler, compiler that cannot build, smoke wiring) |
| `install_vllm.sh --plan` and the install refuse without a toolchain, before any download; the install ends with the GPU kernel smoke | `deploy/runpod/install_vllm.sh` | `test_the_installer_plan_and_install_refuse_before_downloading_anything` |
| the vLLM stage guard runs the same check before the engine starts | `deploy/runpod/scale_session.sh` | `test_the_vllm_stage_guard_*` |
| vLLM is launched with `--no-enable-prefix-caching` and the guard requires the switch (see section 7) | `deploy/runpod/scale_session.sh` | `test_the_vllm_command_turns_prefix_caching_off_*` |
| lean stages `vllm-7b-bf16-cmp` and `7b-bf16-cmp`, the re-run session and `--rerun-budget` | `tensorpilot/suite.py`, `benchmarks/gpu/v10-*-cmp.json` | `tests/test_production_suite.py` (lean stages copy the prod definitions unchanged) |
| the re-run chain | `deploy/runpod/run_vllm_rerun.sh` (refuses until the rc2-4 digest is pinned) | `tests/test_run_vllm_rerun.py` |
| the acceptance checker pairs vLLM with the same-card control, and no longer reads a vLLM stage that never started as a hardware mismatch | `tensorpilot/acceptance.py` | `tests/test_acceptance.py` |

TensorPilot's batching implementation, the server and the acceptance thresholds are unchanged. `3.overload` stays FAIL.

## 7. Limits on an apples-to-apples TensorPilot vs vLLM comparison

1. **Different software stacks, so a gap has no single cause.** TensorPilot: torch 2.9.1 + CUDA 12.8, Transformers eager attention, a fixed-slot KV pool, a Python scheduler. vLLM: torch 2.13.0 + CUDA 13.0,
   `torch.compile`, CUDA graphs, FlashAttention 2, paged KV, chunked prefill. The comparison measures products, not a controlled variable.
2. **Prefix caching.** Every repeat of an experiment replays the same prompts (verified: cmp-c128 repeats 1 and 2 have identical prompts). vLLM's default prefix cache would serve repeats 2 and 3 from cache.
   The rerun disables it; vLLM with shipped defaults is a different, also valid, measurement that this plan does not make.
3. **Same card needs a control.** The archived TensorPilot run was on another Pod. The rerun adds a short TensorPilot control (comparison points only) on the vLLM Pod, which also measures Pod-to-Pod variance against the archive.
4. **Matched settings are limited to what is shared**: model revision, dtype, 48 sequences, 1,536 token context, greedy sampling, the same client and workload hashes. Unmatched and unpinned: vLLM's token budget per step and
   scheduler policy (its defaults), KV memory (vLLM takes 90% of VRAM, TensorPilot a 4 GiB pool, so VRAM columns are not comparable), start-up cost (compile and graph capture are not in throughput).
5. **Wire formats differ** (NDJSON against OpenAI SSE); inter-token times are taken per received chunk in both.
6. **Output length can differ** between engines when greedy paths diverge at a near-tie, so tokens per request are reported and tok/s is not a per-request-equal comparison. No cross-engine output equality is asserted.
7. **Sample size.** Three repeats at the comparison points, single runs on the plateau points, one GPU model, short prompts.
8. **Out of scope for vLLM**: cancellation, timeouts, overload and crash recovery experiments have no vLLM counterpart in this suite.

## 8. The cheapest valid next GPU experiment

Rerun only what is missing, on a fresh Pod, from a new image that contains the compiler. Do not repeat the serial stage, the batching stage, the crash experiment or the probe.

| step | purpose | expected |
|---|---|---|
| `check_toolchain.sh` | compiler present, in seconds | < 1 min |
| `install_vllm.sh --plan` / install | vLLM venv; ends with a real Triton kernel on the GPU | about 15–20 min (not in the ledger) |
| `vllm-7b-bf16-cmp` | cold start, cmp-c1/c8/c32/c128 × 3 repeats, scale-c64/256/1024; includes the 14 GB download | stage cap 49 min |
| `7b-bf16-cmp` | TensorPilot control, same Pod and card: the comparison points only | stage cap 26 min |
| `quality-7b` | the skipped NF4 vs bf16 evaluation | stage cap 30 min |

At $0.91/h (`python -m tensorpilot.suite --rerun-budget --price-per-hour 0.91`): ledger budget 84 minutes, expected total about 98 minutes ≈ **$1.48**, worst case 134 minutes ≈ **$2.03**,
against $4.63 for the full 7B chain. Prerequisite met: rc2-4 (`sha256:92721fab66c72deb382cf21248ca19cb43e18dc5147fc9a25deacaddf888612f`) is published and contains `gcc`.

The control makes the pair acceptable to `5.same-hardware-7b`. Without it, vLLM could only be paired with the archived run from another card, and that item would stay FAIL (different card), as it should.

Stop conditions: if `check_toolchain.sh` or the install smoke fails, stop and terminate (the cost so far is boot plus a short install). If `vllm-7b-bf16-cmp` fails again, stop, archive, terminate: do not run the control or quality alone.
If the vLLM numbers come back, the `## Why vLLM differs` section of the report must cite measurements (`5.gaps-explained`); until then no statement about relative speed is supported.

### The re-run script (copy of `deploy/runpod/run_vllm_rerun.sh`; a test keeps it identical)

The script is pinned to the published rc2-4 digest. Exports: `TENSORPILOT_API_KEY`, `TENSORPILOT_IMAGE_REF` (ending in that digest), `RUNPOD_POD_ID`, `SESSION_BUDGET_MINUTES=84`,
`PREFILL_PACK_TOKENS=512`; the Pod environment needs `TENSORPILOT_POD_MAX_MINUTES=150`.

```bash
#!/usr/bin/env bash
# Re-run only what the 7B session lost, on a fresh Pod, as user tensorpilot, inside tmux:
#   the vLLM comparison (vllm-7b-bf16-cmp), a TensorPilot control on the same card (7b-bf16-cmp), and quality-7b.
#
#   set -o pipefail
#   bash run_vllm_rerun.sh 2>&1 | tee ~/rerun.log ; echo "exit status: ${PIPESTATUS[0]}"
#
# Order: toolchain check -> vLLM install (plan, install, GPU kernel smoke) -> vllm-7b-bf16-cmp -> 7b-bf16-cmp -> quality-7b.
# vLLM goes first because it is what failed; everything before it takes seconds or is the install itself. It stops at the first
# failure of anything; every stage archives its evidence whether it passes or fails. THE POD KEEPS BILLING until you terminate it.
#
# Environment it needs. You export: TENSORPILOT_API_KEY, TENSORPILOT_IMAGE_REF, RUNPOD_POD_ID, SESSION_BUDGET_MINUTES (84 to 95),
# PREFILL_PACK_TOKENS (512); PRICE_PER_HOUR_USD is optional. The login profile gives it: PATH (scale_session.sh, install_vllm.sh,
# check_toolchain.sh), TENSORPILOT_POD_MAX_MINUTES (150 or more) and TENSORPILOT_RESULTS_ROOT. Nothing else is read.
set -Eeuo pipefail

# The digest of the published image that carries gcc and libc6-dev (v1.0.0-rc2-4). The script refuses to run until it is pinned.
EXPECTED_DIGEST="sha256:92721fab66c72deb382cf21248ca19cb43e18dc5147fc9a25deacaddf888612f"
STAGE="startup"
trap 'echo >&2; echo "*** STOPPED at: ${STAGE}. Evidence for every stage that ran, including this one, is archived under ${SESSION:-the session directory}." >&2; echo "*** THE POD IS STILL BILLING: copy the evidence, then TERMINATE the Pod." >&2' ERR

fail() { echo "refused: $*" >&2; exit 2; }
[[ -z "${V10_TEST_MODE:-}" ]] || fail "V10_TEST_MODE is set; that mode skips GPU checks and its results are not evidence"
[[ "$EXPECTED_DIGEST" =~ ^sha256:[0-9a-f]{64}$ ]] || fail "this script has no image digest pinned yet (EXPECTED_DIGEST); publish the image and pin it"
for name in TENSORPILOT_API_KEY RUNPOD_POD_ID; do [[ -n "${!name:-}" ]] || fail "$name is not exported"; done
[[ "${TENSORPILOT_IMAGE_REF:-}" == *"@${EXPECTED_DIGEST}" ]] || fail "TENSORPILOT_IMAGE_REF must end in @${EXPECTED_DIGEST}"
[[ "${PREFILL_PACK_TOKENS:-}" == 512 ]] || fail "PREFILL_PACK_TOKENS must be 512: the control must run TensorPilot as the archived 7b-bf16-prod did"
budget="${SESSION_BUDGET_MINUTES:-}"; pod_max="${TENSORPILOT_POD_MAX_MINUTES:-}"
[[ "$budget" =~ ^[0-9]+$ ]] && (( budget >= 84 && budget <= 95 )) ||
  fail "SESSION_BUDGET_MINUTES must be 84 to 95 (less and the ledger refuses a stage; more spends past the plan)"
[[ "$pod_max" =~ ^[0-9]+$ ]] && (( pod_max >= 150 )) ||
  fail "TENSORPILOT_POD_MAX_MINUTES must be at least 150 in the Pod's environment variables"
for tool in scale_session.sh install_vllm.sh check_toolchain.sh; do
  command -v "$tool" >/dev/null || fail "$tool is not on PATH (log in again so the profile is read)"
done
driver="$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
echo "GPU: ${driver}"
major="${driver##*, }"; major="${major%%.*}"
[[ "$major" =~ ^[0-9]+$ ]] && (( major >= 580 )) || fail "the driver must be 580 or newer for vLLM (this host: ${driver})"

export SESSION_BUDGET_MINUTES PREFILL_PACK_TOKENS TENSORPILOT_API_KEY
SESSION="${TENSORPILOT_V10_SESSION:-${TENSORPILOT_RESULTS_ROOT:-/workspace/results}/v10-session}"
export TENSORPILOT_V10_SESSION="$SESSION"

run() { STAGE="$1"; echo; echo "=== $(date -u +%H:%M:%S) stage $1"; scale_session.sh "$1"; }

# ----- 1. the thing that failed last time, proven in seconds, before any billed download -----
STAGE="toolchain check"
check_toolchain.sh

# ----- 2. vLLM: its own virtualenv; the installer repeats the toolchain check and launches a real Triton kernel on the GPU -----
STAGE="vLLM plan"
install_vllm.sh --plan
STAGE="vLLM installation"
install_vllm.sh

# ----- 3. the comparison: vLLM first, then TensorPilot on the same card, then the skipped quality evaluation -----
run vllm-7b-bf16-cmp
run 7b-bf16-cmp
run quality-7b

STAGE="finished"
echo; echo "=== ALL STAGES DONE ($(date -u +%H:%M:%S))"
cat "$SESSION/ledger.jsonl"
ls -lh "$SESSION"/*.tar.gz
echo "*** COPY THE EVIDENCE (the archives, ledger.jsonl and rerun.log) TO YOUR MAC, THEN TERMINATE THE POD (terminating, not stopping, ends billing)."
```
