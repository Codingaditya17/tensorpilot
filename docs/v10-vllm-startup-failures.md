# vLLM start-up failures on the RTX 4090 Pod: two root causes, the fixes, and what is and is not proven

Both failed attempts died before serving a request. Each had a different cause, and each was invisible to the tests that existed.

| attempt | image | where it died | first underlying error |
|---|---|---|---|
| 1 (7B session) | rc2-3 | `torch.compile`, in the memory-profiling pass | `Failed to find C compiler` (Triton cannot build its CUDA helper) |
| 2 (re-run) | rc2-4 | kernel warm-up, after the KV cache was sized | `Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist` (FlashInfer) |

Section 1 is attempt 1 (fixed and confirmed on a GPU by attempt 2 getting past it). Everything after is attempt 2.

## 1. Attempt 1, for context
`docs/v10-7b-results.md` section 5. The slim base image had no C compiler; rc2-4 added `gcc`. Attempt 2's archive confirms that fix on the GPU:
`toolchain.txt` shows `triton_smoke: triton 3.7.1 compiled and ran a kernel on NVIDIA GeForce RTX 4090`, and the engine log shows
`torch.compile took 16.65 s in total` and `Capturing CUDA graphs ... finished in 2 secs`.

## 2. Attempt 2: evidence

Archive `vllm-7b-bf16-cmp-20261027T030707Z.tar.gz`, SHA-256 `6205f5fe1aad695a082517aeb72670fc1d5cc7cc4ed201709f3709320dc08259` (verified), Pod GPU RTX 4090, driver 595.91.07, image
`cuda-experiment@sha256:92721fab…888612f`. The first error is at `engine.log` line 54 (`EngineCore failed to start`); the innermost exception is at line 150:

```
File ".../vllm/v1/worker/gpu/warmup.py", line 373, in _warmup_kernels      worker_sample_tokens(grammar_output)
File ".../vllm/v1/worker/gpu/sample/sampler.py", line 336, in _sample_random   flashinfer_sample(processed_logits, top_k, top_p)
File ".../flashinfer/sampling.py", line 68, in get_sampling_module           module = gen_sampling_module().build_and_load()
File ".../flashinfer/jit/cpp_ext.py", line 61, in get_cuda_path
RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist
```

What the log establishes, line by line, before the failure: the weights loaded (14.29 GiB, 24 s), Dynamo and Inductor compiled the model (16.65 s), piecewise and full CUDA graphs were captured, the
KV cache was sized (5.57 GiB, 104,304 tokens, 67.9× concurrency at 1,536 tokens) and `JIT kernel warmup starting`. The very next step, the engine's warm-up sampling pass, asks FlashInfer for its top-k/top-p
kernel; FlashInfer compiles that with `nvcc` the first time, and there is no `nvcc` anywhere the way FlashInfer looks (`CUDA_HOME`, `which nvcc`, `/usr/local/cuda`).

Ruled out from the same log: Triton and Inductor (both succeeded), CUDA graph capture, GPU memory (sized with 5.57 GiB free for KV), driver and CUDA 13 compatibility, multiprocessing and shared memory
(the engine core process started and ran its warm-up; nothing about `/dev/shm` or spawn appears), download (14 s), and the launch flags (all accepted).

Why the benchmark would not have noticed: the benchmark is greedy, and a greedy request does not take FlashInfer's sampler. The engine's **warm-up** does (it samples with top-k and top-p), so the failure
is at start-up regardless. A later random-sampling request in production would have hit the same compile.

## 3. Why "install nvcc" alone was not enough: a third failure, found offline

The lock already contains `nvidia-cuda-nvcc`, so the obvious repair is to point `CUDA_HOME` at it. It is not sufficient, and this was found by actually running FlashInfer's build, not by reading code.

I installed the real hash-locked vLLM environment (`install_vllm.sh`, 199 packages, 4 minutes) in an emulated linux/amd64 container (no GPU; nvcc compiles without one) and reproduced both stages:

1. With no `CUDA_HOME`, `gen_sampling_module().build()` fails with the archive's exact message.
2. With a `CUDA_HOME` assembled from the venv's wheels, nvcc was found and started compiling, and then failed:
   ```
   .../flashinfer/data/cccl/libcudacxx/include/cuda/std/__cccl/cuda_toolkit.h:41:8: error: #error
   "CUDA compiler and CUDA toolkit headers are incompatible, please check your include paths"
   ```
   The lock had `nvidia-cuda-nvcc` **13.4.92** (pulled in unpinned by `humming-kernels` and `tilelang`) beside `nvidia-cuda-runtime` **13.0.96**, whose headers define `CUDART_VERSION 13000`. torch's own `cuda-toolkit 13.0.3.0`
   pins the compiler set to 13.0.88. CCCL refuses the mix, correctly.
3. After pinning `nvidia-cuda-nvcc`, `nvidia-cuda-crt` and `nvidia-nvvm` to 13.0.88 (a three-line re-lock; the diff against the old lock is exactly those three versions), the same build succeeded:
   `BUILD OK in 49 s` for the sampling module for `sm_89`, including the link against a `libcuda` stub.

## 4. The permanent fix

| change | why |
|---|---|
| `requirements-vllm.in` / `.txt`: nvcc, crt, nvvm pinned to 13.0.88 | one coherent CUDA 13.0 compiler set matching the runtime headers; hash-locked like everything else |
| `deploy/runpod/cuda_home.sh` | assembles `CUDA_HOME` from symlinks into the venv's `nvidia/cu13` tree (bin, include, nvvm, `lib64`, an unversioned `libcudart.so`, `lib64/stubs/libcuda.so` pointing at the driver). Nothing is downloaded and nothing outside the venv is touched. Refuses if a component is missing |
| `Dockerfile.experiment`: `g++` | nvcc compiles the host half of every `.cu` file with a C++ compiler, and FlashInfer builds `.cpp` with `c++`. Debian's `gcc` package does not include `g++`; only the nvcc build, not the C check, exposes that |
| `check_toolchain.sh` | also checks `c++` with the standard library; with `--gpu` assembles `CUDA_HOME`, compiles a CUDA shared object for sm_89 with nvcc (the whole pipeline: cicc, ptxas, host compiler, linker), then runs the Triton and FlashInfer smoke kernels (`flashinfer_smoke.py` runs the engine warm-up's exact sampling op, and its first run populates the JIT cache the engine then reuses) |
| `scale_session.sh` | the vLLM guard builds `CUDA_HOME`; the engine is started with `CUDA_HOME` set and the venv's `bin` (ninja) and `$CUDA_HOME/bin` on `PATH` |
| stage `vllm-startup-0.5b` + `tensorpilot/vllm_startup_check.py` | the early GPU validation: the benchmark's exact vLLM command on Qwen2.5-0.5B (about 1 GB), the 5-request smoke, then four sampling requests (greedy, top-k+top-p, top-k, top-p) so the sampler path is exercised after start-up. Archived like any stage; the chain stops on its failure before the 7B download |
| `run_vllm_rerun.sh` | runs: toolchain check, install (with GPU smokes), early startup stage, vLLM 7B comparison, TensorPilot control, quality. Budget 97–105 ledger minutes |
| `suite.py` | the 7B download is charged to the first stage that uses the 7B model (it was the session's first stage before) |

TensorPilot's batching code, the benchmark definitions and the acceptance thresholds are unchanged. vLLM is not run in a reduced mode: the FlashInfer sampler stays on, and prefix caching stays off for the reason in `docs/v10-7b-results.md` section 7.

## 5. Tests

`tests/test_vllm_toolchain.py` (compiler, C++, CUDA_HOME layout and failure modes, smoke wiring, engine environment), `tests/test_vllm_environment.py` (the compiler set is one release and matches the runtime headers and
the toolkit pin; a relock cannot drift it), `tests/test_vllm_startup_check.py` (a stand-in OpenAI server: which answers pass, a sampler-path server error fails the check, the key is never an argument),
`tests/test_run_vllm_rerun.py` (order, early stop after the startup stage, budget, pinning), `tests/test_production_suite.py` (the startup stage uses the benchmark's vLLM settings; per-model download).

## 6. What is verified and what is not

**Verified without a GPU:**
- the archive's exact error, reproduced from the real locked vLLM environment (section 3.1);
- the CCCL incompatibility, reproduced, and cleared by the 13.0.88 pins (3.2, 3.3);
- nvcc, cicc, ptxas, `g++` and the linker build a CUDA shared object for sm_89 in the image (`check_toolchain.sh --gpu` reaches `nvcc: builds a CUDA shared object for sm_89` and stops at the Triton smoke only because the machine has no GPU);
- FlashInfer's sampling module compiles and links with the assembled `CUDA_HOME`.

**NOT verified, because they need a GPU:** that the compiled sampling module loads and runs (it links `-lcuda` against the driver and needs `libcudart.so.13` resolved at load time), that `flashinfer_smoke.py` passes, that
the engine now completes its warm-up, that the early stage passes, and everything after it. A fourth start-up failure after the warm-up is possible. The early stage is designed so that it would cost minutes on a 0.5B model rather than the 7B run.
Treat the vLLM start-up fix as **unproven on a GPU until the next run reports**.

## 7. The next run

Image: `ghcr.io/codingaditya17/tensorpilot/cuda-experiment@sha256:25908f3ae3a51f7d2c3cc54be5ba50583e8ad8277b95a784ba9b69b3abfdd644` (v1.0.0-rc2-6, revision 0f4f665; an rc2-5 tag exists on GitHub but its image job failed before publishing, so no rc2-5 image exists). Pod: 1 × RTX 4090, driver 580+, TCP 22, container disk 50 GB, volume 120 GB, RAM ≥ 31 GB, `TENSORPILOT_POD_MAX_MINUTES=165`.

At $0.91/h (`python -m tensorpilot.suite --rerun-budget --price-per-hour 0.91`): ledger budget 97 minutes, expected total about 109 minutes, **about $1.65**; worst case 147 minutes, **about $2.23**. Fail-fast points and what each costs:
the toolchain check (seconds), the install with its GPU smokes (about 15–20 minutes of install, then the sm_89 FlashInfer compile, 1–3 minutes on the Pod's CPUs), then the 0.5B startup stage (cap 12 minutes, expected 4).
If any of them fails the chain stops, archives that stage, and prints a reminder to terminate. Do not continue by hand.

Exports before the script: `TENSORPILOT_API_KEY`, `TENSORPILOT_IMAGE_REF` (ending in the pinned digest), `RUNPOD_POD_ID`, `SESSION_BUDGET_MINUTES=97`, `PREFILL_PACK_TOKENS=512`. Then `set -o pipefail; bash ~/rerun.sh 2>&1 | tee ~/rerun.log; echo "exit status: ${PIPESTATUS[0]}"`.
Copy the archives, `ledger.jsonl` and `rerun.log` before terminating the Pod.

### The re-run script (copy of `deploy/runpod/run_vllm_rerun.sh`; a test keeps it identical)

```bash
#!/usr/bin/env bash
# Re-run only what the 7B session lost, on a fresh Pod, as user tensorpilot, inside tmux:
#   the vLLM comparison (vllm-7b-bf16-cmp), a TensorPilot control on the same card (7b-bf16-cmp), and quality-7b.
#
#   set -o pipefail
#   bash run_vllm_rerun.sh 2>&1 | tee ~/rerun.log ; echo "exit status: ${PIPESTATUS[0]}"
#
# Order: toolchain check -> vLLM install (plan, install, GPU smoke: Triton and FlashInfer kernels) -> vllm-startup-0.5b (the benchmark's
# exact vLLM command on a small model, plus top-k/top-p sampling requests) -> vllm-7b-bf16-cmp -> 7b-bf16-cmp -> quality-7b.
# vLLM goes first because it is what failed twice; the early startup stage means a third start-up failure costs minutes, not the 7B run. It stops at the first
# failure of anything; every stage archives its evidence whether it passes or fails. THE POD KEEPS BILLING until you terminate it.
#
# Environment it needs. You export: TENSORPILOT_API_KEY, TENSORPILOT_IMAGE_REF, RUNPOD_POD_ID, SESSION_BUDGET_MINUTES (97 to 105),
# PREFILL_PACK_TOKENS (512); PRICE_PER_HOUR_USD is optional. The login profile gives it: PATH (scale_session.sh, install_vllm.sh,
# check_toolchain.sh), TENSORPILOT_POD_MAX_MINUTES (165 or more) and TENSORPILOT_RESULTS_ROOT. Nothing else is read.
set -Eeuo pipefail

# The digest of the published image that carries gcc, libc6-dev and the CUDA_HOME assembly (v1.0.0-rc2-6). The script refuses to run until it is pinned.
EXPECTED_DIGEST="sha256:25908f3ae3a51f7d2c3cc54be5ba50583e8ad8277b95a784ba9b69b3abfdd644"
STAGE="startup"
trap 'echo >&2; echo "*** STOPPED at: ${STAGE}. Evidence for every stage that ran, including this one, is archived under ${SESSION:-the session directory}." >&2; echo "*** THE POD IS STILL BILLING: copy the evidence, then TERMINATE the Pod." >&2' ERR

fail() { echo "refused: $*" >&2; exit 2; }
[[ -z "${V10_TEST_MODE:-}" ]] || fail "V10_TEST_MODE is set; that mode skips GPU checks and its results are not evidence"
[[ "$EXPECTED_DIGEST" =~ ^sha256:[0-9a-f]{64}$ ]] || fail "this script has no image digest pinned yet (EXPECTED_DIGEST); publish the image and pin it"
for name in TENSORPILOT_API_KEY RUNPOD_POD_ID; do [[ -n "${!name:-}" ]] || fail "$name is not exported"; done
[[ "${TENSORPILOT_IMAGE_REF:-}" == *"@${EXPECTED_DIGEST}" ]] || fail "TENSORPILOT_IMAGE_REF must end in @${EXPECTED_DIGEST}"
[[ "${PREFILL_PACK_TOKENS:-}" == 512 ]] || fail "PREFILL_PACK_TOKENS must be 512: the control must run TensorPilot as the archived 7b-bf16-prod did"
budget="${SESSION_BUDGET_MINUTES:-}"; pod_max="${TENSORPILOT_POD_MAX_MINUTES:-}"
[[ "$budget" =~ ^[0-9]+$ ]] && (( budget >= 97 && budget <= 105 )) ||
  fail "SESSION_BUDGET_MINUTES must be 97 to 105 (less and the ledger refuses a stage; more spends past the plan)"
[[ "$pod_max" =~ ^[0-9]+$ ]] && (( pod_max >= 165 )) ||
  fail "TENSORPILOT_POD_MAX_MINUTES must be at least 165 in the Pod's environment variables"
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

# ----- 2. vLLM: its own virtualenv; the installer repeats the toolchain check, assembles a CUDA_HOME from the venv's CUDA wheels, and runs a
# -----    Triton kernel and FlashInfer's sampling kernel (nvcc-compiled) on the GPU -----
STAGE="vLLM plan"
install_vllm.sh --plan
STAGE="vLLM installation"
install_vllm.sh

# ----- 3. the early validation (a 0.5B download, minutes), then the comparison: vLLM first, then TensorPilot on the same card, then the
# -----    skipped quality evaluation. A failure at any step stops the chain with that stage's evidence archived. -----
run vllm-startup-0.5b
run vllm-7b-bf16-cmp
run 7b-bf16-cmp
run quality-7b

STAGE="finished"
echo; echo "=== ALL STAGES DONE ($(date -u +%H:%M:%S))"
cat "$SESSION/ledger.jsonl"
ls -lh "$SESSION"/*.tar.gz
echo "*** COPY THE EVIDENCE (the archives, ledger.jsonl and rerun.log) TO YOUR MAC, THEN TERMINATE THE POD (terminating, not stopping, ends billing)."
```
