# Current engineering state

TensorPilot v0.3.0 tested checkpoint. Durable history: WORK_LOG.md.

- Source: 74 tests passed, no warnings; Ruff clean.
- Final installed wheel: all 74 tests passed outside checkout without MLX.
- Real studies: 540 scheduler requests, 15 measured cache-profile requests,
  182-request soak, interrupted-journal recovery, final harness/API checks.
- FIFO retained after measured shortest-budget tail-latency regressions.
- Service running: 127.0.0.1:8080, FIFO, queue 8, 6 GiB allocator limit,
  profiling disabled. Runtime log: results/runtime_v03_final.log.
- Final validation: results/final_validation_v03.json. Raw evidence and matching
  source snapshots remain under results/. Exported artifacts are in chat outputs.
- Dedicated local Git checkpoint; no remote publication or push.

Before experiments: stop the existing server so the orchestrator owns one model
process/port at a time. Keep source fingerprints distinct across future upgrades.
Next substantive work: backend conformance and an opt-in real MLX batching design
with independent per-sequence cache/cancellation; measure correctness, capacity
and memory before enabling it. Also evaluate process supervision and quality gates.
