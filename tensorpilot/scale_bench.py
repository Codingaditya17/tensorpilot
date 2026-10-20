"""Benchmark TensorPilot registry scaling without loading a model or allocating GPU memory."""
import argparse
from collections import Counter, deque
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import time
import tracemalloc

from .reporting import write_json
from .routing import WorkerRegistry


BASELINE_COMMIT = "675d1fba65092183467b86dbe5b525ccdb2a871a"


def percentile(values, quantile):
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def worker_health(index, capacity):
    return {
        "ready": True,
        "active_sequences": 0,
        "queue_depth": 0,
        "execution_concurrency": capacity,
        "queue_capacity": capacity,
        "admission": {
            "estimated_kv_reserved_bytes": 0,
            "estimated_kv_budget_bytes": 8 * 1024**3,
        },
        "model": "synthetic-model",
        "artifact_sha256": "0" * 64,
        "service_version": "0.7.0",
        "instance_id": f"synthetic-{index}",
        "execution_mode": "synthetic",
    }


def run_benchmark(*, workers, placements, in_flight, policy="capacity-aware"):
    if not 1 <= workers <= 10_000:
        raise ValueError("workers must be in 1..10000")
    if not 1 <= placements <= 1_000_000:
        raise ValueError("placements must be in 1..1000000")
    if not 1 <= in_flight <= placements:
        raise ValueError("in_flight must be in 1..placements")

    registry = WorkerRegistry(stale_after_s=3600, policy=policy)
    capacity = max(8, math.ceil(in_flight / workers) + 2)
    tracemalloc.start()
    registration_start = time.perf_counter()
    for index in range(workers):
        worker_id = f"worker-{index:05d}"
        registry.register(worker_id, f"http://127.0.0.1:{10_000 + index}")
        registry.update(worker_id, worker_health(index, capacity))
    registration_s = time.perf_counter() - registration_start

    heartbeat_start = time.perf_counter()
    for index, worker_id in enumerate(registry.workers):
        registry.update(worker_id, worker_health(index, capacity))
    heartbeat_s = time.perf_counter() - heartbeat_start

    latencies_ns = []
    selected = Counter()
    active = deque()
    wall_start = time.perf_counter()
    for index in range(placements):
        started = time.perf_counter_ns()
        placement = registry.select(1024, model="synthetic-model",
                                    artifact_sha256="0" * 64,
                                    request_id=f"synthetic-{index}")
        latencies_ns.append(time.perf_counter_ns() - started)
        if placement is None:
            raise RuntimeError("synthetic benchmark unexpectedly exhausted capacity")
        selected[placement.worker_id] += 1
        active.append(placement.worker_id)
        if len(active) >= in_flight:
            registry.release(active.popleft())
    while active:
        registry.release(active.popleft())
    placement_wall_s = time.perf_counter() - wall_start

    snapshot_start = time.perf_counter()
    snapshot = registry.snapshot()
    snapshot_json = json.dumps(snapshot, separators=(",", ":"), allow_nan=False).encode()
    snapshot_s = time.perf_counter() - snapshot_start
    current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    latencies_us = [value / 1000 for value in latencies_ns]
    counts = list(selected.values())
    mean_count = statistics.mean(counts)
    return {
        "schema_version": 1,
        "benchmark": "registry-placement-saturation",
        "baseline": {"release": "v0.7.0", "commit": BASELINE_COMMIT},
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {"workers": workers, "placements": placements,
                          "max_simultaneous_in_flight": in_flight, "policy": policy,
                          "gpu_allocated": False, "network_used": False},
        "results": {
            "registration_s": registration_s,
            "heartbeat_sweep_s": heartbeat_s,
            "heartbeat_updates_per_s": workers / heartbeat_s,
            "placement_wall_s": placement_wall_s,
            "placements_per_s": placements / placement_wall_s,
            "placement_latency_us": {
                "p50": percentile(latencies_us, .50),
                "p95": percentile(latencies_us, .95),
                "p99": percentile(latencies_us, .99),
                "max": max(latencies_us),
            },
            "snapshot_s": snapshot_s,
            "snapshot_bytes": len(snapshot_json),
            "traced_current_bytes": current_bytes,
            "traced_peak_bytes": peak_bytes,
            "selected_workers": len(selected),
            "assignments_per_worker": {
                "min": min(counts), "max": max(counts), "mean": mean_count,
                "population_stddev": statistics.pstdev(counts),
            },
        },
        "interpretation": (
            "This isolates in-process v0.7 registry and placement cost. It does not measure HTTP, "
            "networking, model inference, GPU scaling, or end-to-end request capacity."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=100)
    parser.add_argument("--placements", type=int, default=10_000)
    parser.add_argument("--in-flight", type=int, default=1_000)
    parser.add_argument("--policy", choices=WorkerRegistry.POLICIES, default="capacity-aware")
    parser.add_argument("--output", default="results/v08_registry_baseline.json")
    args = parser.parse_args()
    try:
        result = run_benchmark(workers=args.workers, placements=args.placements,
                               in_flight=args.in_flight, policy=args.policy)
    except ValueError as exc:
        parser.error(str(exc))
    path = Path(args.output)
    if path.exists():
        parser.error(f"output already exists: {path}")
    write_json(path, result)
    print(json.dumps(result["results"], indent=2))
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
