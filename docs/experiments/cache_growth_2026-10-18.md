# KV-cache growth study — 2026-10-18

Real MLX inference on the local Apple M5 / 16 GiB machine, with the same pinned
Qwen2.5-0.5B-Instruct 4-bit artifact as the scheduling study. Profiling was
explicitly enabled; these timings include profiling overhead.

Five context lengths, three measured requests each, one excluded warmup per
context, 16-token output cap. Fifteen measured requests completed successfully.

| Actual prompt tokens | Median peak cache-array bytes | Median client latency |
|---:|---:|---:|
| 19 | 3 MiB | 64.5 ms |
| 75 | 3 MiB | 71.9 ms |
| 267 | 6 MiB | 97.7 ms |
| 523 | 9 MiB | 102.3 ms |
| 1035 | 15 MiB | 132.4 ms |

The cache grows in allocation steps rather than one exact-sized allocation for
every token. The installed MLX KVCache uses a 256-token growth step; this explains
why 19- and 75-token prompts share the same cache-array footprint, while longer
contexts move to larger allocations. Output-generation tokens also occupy cache
space. These observations describe this model/cache implementation only.

Cache-owned array nbytes includes spare capacity and is not total physical GPU
usage or process RSS. No memory savings or throughput improvement is claimed.
The longer-context latency includes prompt processing, decode, HTTP and sampling
of profile metadata; it is not an isolated prefill benchmark.

Raw samples, actual prompt counts, finish reasons, model/source/runtime provenance
and device metadata are in `results/cache_profile_v03.json`.
