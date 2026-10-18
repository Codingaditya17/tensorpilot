# Current engineering state

TensorPilot v0.4.0 adds opt-in real MLX continuous batching. Serial/FIFO default.
95 source tests pass; Ruff clean. See WORK_LOG.md and docs/batching-design.md.

Real evidence: adapter token equality, HTTP cancellation/overload/recovery, and
three 30-request runs per mode: 180 successful, zero failures/rejections. Batch
width two: +56.2% throughput, -37.7% P95 on this model/workload. Full evidence
under results/batch_execution_seed42 and results/batch_http_validation.json.

Earlier v0.3 release exports remain unchanged. Current exports are separate.
Nothing remotely published. Before a new experiment stop the local server so
one process owns the model/GPU at a time. Remaining work: larger contexts, wider
batches, sustained batch memory testing, quality evaluation, process supervision,
resource-aware admission and tenant isolation.

Final wheel: 95 CPU-only tests passed outside checkout. Local server restored
to serial/FIFO, 127.0.0.1:8080, queue 8, memory limit 6 GiB. Runtime log
results/runtime_v04_serial.log. Batch soak: 288 successful, zero failures/rejections.
