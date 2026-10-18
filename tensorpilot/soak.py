"""Guarded, reproducible 15/30/60-minute soak runner."""
import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from types import SimpleNamespace
import uuid

import aiohttp

from .journal import recover
from .load import run as run_load
from .reporting import write_json
from .workloads import build_stress_workload


async def guarded_run(args):
    output = Path(args.output)
    workload_path = output.with_suffix(".workload.json")
    journal_path = output.with_suffix(".requests.jsonl")
    plan = build_stress_workload(args.plan_size, prompt_repeats=args.prompt_repeats,
                                 output_tokens=args.output_tokens)
    write_json(workload_path, plan)
    load_args = SimpleNamespace(
        url=args.url, requests=args.plan_size, concurrency=args.concurrency,
        rate=args.rate, max_tokens=args.output_tokens, timeout=args.timeout,
        output=str(output), workload="uniform", workload_file=str(workload_path), seed=42,
        duration_s=args.minutes * 60, max_requests=args.max_requests,
        journal=str(journal_path), journal_fsync=args.journal_fsync,
        compact=True, overwrite=False,
    )
    state = {"schema_version": 1, "status": "running",
             "started_at_utc": datetime.now(timezone.utc).isoformat(),
             "guard": {"max_mlx_active_bytes": args.max_mlx_active_mib * 1024**2,
                       "max_memory_sample_age_s": args.max_memory_sample_age,
                       "poll_seconds": args.poll_seconds}, "samples": []}
    guard_path = output.with_suffix(".guard.json")
    write_json(guard_path, state)
    load_task = asyncio.create_task(run_load(load_args))
    reason = None
    key = os.environ.get("TENSORPILOT_API_KEY")
    headers = {"Authorization": "Bearer " + key} if key else {}
    async with aiohttp.ClientSession(headers=headers) as session:
        while not load_task.done():
            await asyncio.sleep(args.poll_seconds)
            if load_task.done():
                break
            try:
                async with session.get(args.url.rstrip("/") + "/health",
                                       timeout=aiohttp.ClientTimeout(total=5)) as response:
                    health = await response.json()
                sample = {"at_s": len(state["samples"]) * args.poll_seconds,
                          "ready": health.get("ready"), "queue_depth": health.get("queue_depth"),
                          "active_sequences": health.get("active_sequences"),
                          "last_measured_memory": health.get("last_measured_memory"),
                          "memory_sample_age_s": health.get("memory_sample_age_s"),
                          "estimated_kv_reserved_bytes": health.get("admission", {}).get("estimated_kv_reserved_bytes")}
                state["samples"].append(sample)
                active = (sample["last_measured_memory"] or {}).get("mlx_active_bytes", 0)
                if not sample["ready"]:
                    reason = "service_not_ready"
                elif active >= state["guard"]["max_mlx_active_bytes"]:
                    reason = "mlx_active_memory_guard"
                elif (sample["memory_sample_age_s"] is None or
                      sample["memory_sample_age_s"] > args.max_memory_sample_age):
                    reason = "memory_sample_stale"
            except (aiohttp.ClientError, TimeoutError, ValueError):
                reason = "health_probe_failed"
            write_json(guard_path, state)
            if reason:
                load_task.cancel()
                await asyncio.gather(load_task, return_exceptions=True)
                break
    if reason:
        report = recover(journal_path)
        report["metadata"]["safety_abort"] = reason
        state["status"], state["stop_reason"] = "safety_aborted", reason
    else:
        report = await load_task
        state["status"] = "completed"
    state["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(output, report)
    write_json(guard_path, state)
    return report, state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--minutes", type=int, choices=(15, 30, 60), default=15)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--rate", type=float, default=0)
    parser.add_argument("--prompt-repeats", type=int, default=32)
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--plan-size", type=int, default=20)
    parser.add_argument("--max-requests", type=int, default=100000)
    parser.add_argument("--max-mlx-active-mib", type=int, default=4096)
    parser.add_argument("--poll-seconds", type=float, default=5)
    parser.add_argument("--max-memory-sample-age", type=float, default=30,
                        help="abort when allocator observations become too stale")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--journal-fsync", action="store_true")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parser.add_argument("--output", default=f"results/soak_{stamp}_{uuid.uuid4().hex[:6]}.json")
    args = parser.parse_args()
    if not 1 <= args.concurrency <= 32 or args.max_mlx_active_mib < 256:
        parser.error("concurrency must be in 1..32; memory guard must be at least 256 MiB")
    if not 0.25 <= args.poll_seconds <= 60 or args.plan_size < 1 or args.max_requests < 1:
        parser.error("poll interval must be .25..60 seconds; plan size and request cap must be positive")
    if args.max_memory_sample_age <= args.poll_seconds:
        parser.error("max memory sample age must exceed the polling interval")
    try:
        report, state = asyncio.run(guarded_run(args))
    except (ValueError, OSError, RuntimeError, aiohttp.ClientError) as exc:
        parser.exit(1, f"Soak failed: {exc}\n")
    print(json.dumps({"status": state["status"], **report["summary"]}, indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
