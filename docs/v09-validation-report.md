# v0.9 local validation report

Scope: everything that can be verified on an Apple M5 MacBook Pro (16 GB) without
an NVIDIA GPU and without renting infrastructure. Nothing here is a CUDA
performance measurement. The GPU session in
[`v09-runpod-runbook.md`](v09-runpod-runbook.md) is the next step.

Environment: macOS 26.6.2 (arm64), Python 3.14.6 in the project venv, PyTorch 2.9.1
(CPU and MPS build, CUDA not available), Transformers 5.20.1, MLX 0.32.4 and
mlx-lm, `uv` for builds. No GPU-resident model process was started.

## Summary

| check | result |
|---|---|
| Full test suite | **202 passed** (154 existing, 48 new), 0 failed |
| Ruff (`tensorpilot`, `tests`) | clean |
| Real pinned checkpoint on CPU matches transformers greedy | **12 / 12 tokens identical** |
| Graceful SIGTERM during live decode | exit 0, explicit `shutting_down` terminal event |
| Wheel and sdist build | `tensorpilot-0.9.0-py3-none-any.whl`, `tensorpilot-0.9.0.tar.gz` |
| Independent install (fresh venv, wheel only) | imports from site-packages, version 0.9.0 |
| Test suite run from the sdist against the installed wheel | **202 passed** |
| Compose file | `docker compose config` valid |
| Dockerfile | **not built; BuildKit check did not complete** (registry access hung) |
| Preflight on this Mac | **not ready**, as expected (no NVIDIA device) |
| Benchmark runner end to end against a CPU worker | evidence written, manifest and provenance verified |

## Test inventory

New in v0.9 (48 tests):

| module | tests | what it covers |
|---|---:|---|
| `test_cuda_backend.py` | 11 | greedy equivalence with transformers; EOS accounting; finish reasons; detokenization; prompt limits; cancellation and reuse; OOM containment and counting; unmeasured CPU memory; provenance; device and configuration rules; engine cancellation on the real decode loop |
| `test_cuda_cluster.py` | 9 | CUDA worker behind real control and data planes over loopback HTTP: placement and streaming with validated events; model and artifact compatibility; client disconnect cancels decode and releases capacity; connection failure after placement; mid-stream fault with no replay; drain and removal; restart with a new instance identity; the integrated proxy disconnect path; API registration |
| `test_gpu_telemetry.py` | 7 | nvidia-smi parsing; `[N/A]` as null; shape rejection; unavailable and error states; sampler faults that do not stop sampling |
| `test_gpu_preflight.py` | 6 | ready only when every required check passes; no-GPU result explains itself; VRAM threshold; unpinned revision rejected; smoke failures report only the error type; skipped smoke cannot pass |
| `test_gpu_bench.py` | 4 | config rejects unsafe shapes; plan and budget; refuses unpinned revision and planned kinds; end-to-end run writes per-repeat reports with server provenance |
| `test_server_cli.py` | 7 | CUDA refuses the MLX default model, batch execution, MLX-only profiling and bad memory fractions; environment selection; prompt cap; non-loopback bind requires a key |
| `test_graceful_shutdown.py` | 1 | SIGTERM during decode in a real subprocess |
| `test_backend_contract.py` | 3 | both backends implement the serving contract; the protocol declares the engine's members |

Existing tests (154) still pass unchanged. That includes the MLX engine, batch,
adaptive, routing, data-plane and control-plane suites.

## Defects found and fixed

1. **Client disconnect marked a healthy worker as failed (data plane).** Confirmed.
   The default route outcome was `worker_transport_error`, so a client that closed
   its stream mid-decode made the control plane mark the worker unready until the
   next heartbeat. The regression test `test_client_disconnect_cancels_decode_releases_capacity_and_worker_recovers`
   fails on the v0.8.4 proxy code and passes on v0.9. This affects MLX deployments
   too, because the proxy is shared.
2. **Integrated proxy misclassified a client-side write error (latent).** A
   `ConnectionError` while writing to a departed client was counted as a worker
   failure. The test added for this path passes on the old code as well, so the path
   was not reproduced. The change is a hardening fix. It is not claimed as a confirmed
   defect.
3. **The sdist omitted the shared test fixture.** `tests/tiny_model.py` was not in
   the source distribution, so the suite could not run from the sdist. Found by
   installing the sdist and running its tests. Fixed with `MANIFEST.in`, which also
   ships the benchmark configs and deployment files.
4. **Tokenizer resolution in the fixture.** An early test fixture used a Qwen-type
   configuration, which made `AutoTokenizer` select a BPE class that needs vocabulary
   files. The fixture now uses a Llama-type configuration that loads `tokenizer.json`.
   This was a test-fixture issue, not a backend bug. Real checkpoints ship a matching
   tokenizer, and the real-checkpoint check above confirms the backend path.
5. **Invalid pip flag in the Dockerfile.** I introduced `--require-hashes-not-needed`,
   which is not a pip option, and corrected it before any build. Tooling did not catch
   it. Review did, and the image still has not been built.

## Real-checkpoint check (CPU, not CUDA)

Model: `Qwen/Qwen2.5-0.5B-Instruct`, revision `7ae557604adf67be50417f59c2c2f167def9a775`,
artifact SHA-256 `4df441e19d21a857681efbb77b9a9dcbf27f6a468edb09d196161b54e978b366`,
loaded through `CUDABackend(device="cpu", dtype="float32")`.

- Load: 25.8 s, including the first download into the local Hugging Face cache.
- Twelve generated tokens from the prompt "Explain GPU memory bandwidth in one sentence.":
  the streamed token IDs equal `model.generate(do_sample=False)` exactly.
- EOS IDs taken from the model's generation configuration: `[151643, 151645]`.
- Provenance recorded the requested and resolved revision, the artifact hash, the
  PyTorch and Transformers versions, and device `cpu`.

This establishes that the Transformers path, tokenizer, generation config, and
revision recording are correct. It says nothing about CUDA kernels, VRAM use or speed.

## Preflight on this Mac (expected failures)

```
[FAIL] nvidia_smi_visible: nvidia-smi not found
[FAIL] torch_cuda: PyTorch CUDA unavailable (cuda_available is false)
[FAIL] vram: no CUDA device reported by PyTorch
[FAIL] pinned_revision: a 40-character Hugging Face commit SHA is required ...
[FAIL] model_smoke: skipped by --skip-load; a smoke inference is required
ready: false
```

The gate fails safely. Every failure carries its own reason, and the exit status is 1.

## Benchmark plan (planning estimate, not measurement)

`python -m tensorpilot.gpu_bench plan benchmarks/gpu/v09-single-gpu.json`:
experiments 69.0 min, session overhead 45 min, session estimate 114 min against a
300-minute budget. The estimate depends on placeholder assumptions of 60 decode
tokens/s and 2000 prefill tokens/s. Both must be replaced with preflight numbers
before the budget is relied on.

## Distributed behavior over local HTTP

The cluster tests run the production worker app, control plane and data plane on
loopback TCP, with a real CUDA-interface backend on CPU. Each test asserts the
streamed events against the shared `StreamValidator`, so the protocol checks are
the ones a real client applies. The mid-stream test verifies that the request reached
the worker exactly once, so no replay occurred.

## What cannot be verified without NVIDIA hardware

- CUDA execution: kernels, dtype behavior (`float16`, `bfloat16`), and numerical
  agreement between CUDA and CPU outputs.
- Memory: real VRAM occupancy, OOM behavior under the 90% allocator cap, and peak
  reserved bytes under load.
- Telemetry: nvidia-smi output on a real driver, including ECC fields and the
  `[N/A]` pattern on consumer GPUs.
- Throughput, TTFT, inter-token latency and P50/P95/P99 on a GPU. Every value in this
  repository is either a CPU test result or a planning estimate.
- The container NVIDIA runtime, the healthcheck against a loaded model, and the compose
  stack with a GPU reservation. (The image itself builds and passes its CPU tests; see
  docs/v09-cuda-runtime.md.)
- The Runpod session script end to end. Only its `--plan` mode was exercised here.
- The CI job `cuda-backend-cpu` in `.github/workflows/tests.yml`. It was not run on
  GitHub, and its CPU PyTorch index has not been checked from CI.
- Multi-GPU behavior. It is not claimed. `deploy/runpod/two_gpu_session.sh` is prepared and
  checked only in `--plan` mode; see docs/v09-two-gpu-experiment.md.
- Batched CUDA execution. It is not implemented, and batch-width experiments are
  listed as planned.

## Preserved artifacts and release checksums

Earlier evidence is untouched. `results/` (git-ignored) and the v0.8.4 wheel and
sdist in `dist/` were not modified. The v0.9.0 artifacts sit beside them.

| artifact | sha256 |
|---|---|
| `dist/tensorpilot-0.9.0-py3-none-any.whl` | `fc071913faa431851eb7a1f32c3dcea9f1dab3da685a08b16a59ea9ad8f02c87` |
| `dist/tensorpilot-0.9.0.tar.gz` | `e8d6a53c83b9e295a672e57eddf2950c99b72c20c08b5285b3b21ee7de20c801` |

Both were verified after the final build: the wheel was installed into a fresh
environment outside the repository, and the sdist's test suite (202 tests) passed
against that installed wheel.
