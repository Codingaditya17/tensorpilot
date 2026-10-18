# Actual cache-array and allocator profiling

Profiling is opt-in because collecting host-side samples adds overhead:

```bash
tensorpilot-server --offline --profile-memory
```

In another terminal:

```bash
tensorpilot-profile --repeat 3 --output results/cache_profile.json
```

This experiment uses five controlled prompt lengths, one excluded warmup per
length, three measured repetitions, and a 16-token output cap. Actual encoded
prompt-token counts are recorded rather than inferred from text length.

The backend creates a fresh MLX prompt cache for each request and passes it to
the real generator. Each sampled token records the sum of cache objects' `nbytes`
and the current MLX active allocation. Requests never reuse one another's cache.
Final NDJSON events contain prompt count, finish reason and bounded profile
samples. A cache type without byte inspection reports **unknown**, not zero.

Cache-array bytes include spare capacity allocated by cache growth. They are
not necessarily unique physical device allocations when views/aliasing exist,
not process RSS, and not system unified-memory pressure. Process-lifetime MLX
allocation peaks include loading/warmup. Profile timing includes collection
overhead and must not be compared to unprofiled timing as an optimization.

Prometheus exposes last-sampled MLX active bytes, process-lifetime allocation
peak, and sample age. Samples are collected on the sole model thread after
startup/generation. A long-running call can make them stale; sample age makes
that visible without concurrently calling the GPU backend from a metrics handler.

The cache profiling study is a separate real-hardware artifact; CPU stub tests
only verify accounting and protocol behavior.
