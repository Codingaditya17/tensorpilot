"""Controlled opt-in KV-cache profiling against an already-running MLX service."""
import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
import statistics
import time

import aiohttp

from .client import Client
from .reporting import write_json


def summarize_profiles(records):
    grouped = {}
    for record in records:
        grouped.setdefault(record["prompt_repeats"], []).append(record)
    summary = []
    for repeats, group in sorted(grouped.items()):
        peaks = [max((sample["cache_allocated_bytes"] for sample in record["server"]["memory_profile"]
                      if sample["cache_allocated_bytes"] is not None), default=None) for record in group]
        supported = all(value is not None for value in peaks)
        summary.append({"prompt_repeats": repeats, "runs": len(group),
                        "prompt_tokens": group[0]["server"]["prompt_tokens"],
                        "median_peak_cache_array_bytes": statistics.median(peaks) if supported else None,
                        "median_client_elapsed_s": statistics.median(record["client_elapsed_s"] for record in group),
                        "cache_profile_supported": supported})
    return summary


async def run(url, *, repetitions=3):
    key = os.environ.get("TENSORPILOT_API_KEY")
    headers = {"Authorization": "Bearer " + key} if key else {}
    async with aiohttp.ClientSession(headers=headers) as session:
        async with session.get(url.rstrip("/") + "/v1/model", timeout=aiohttp.ClientTimeout(total=10)) as response:
            response.raise_for_status()
            server = await response.json()
    if server.get("backend") != "mlx" or not server.get("memory_profiling"):
        raise ValueError("start the MLX server with --profile-memory before running this experiment")
    records = []
    async with Client(url, api_key=key) as client:
        for repeats in (1, 8, 32, 64, 128):
            prompt = "Summarize the repeated observation in one sentence:\n" + "GPU memory bandwidth can limit transformer inference. " * repeats
            # One excluded warmup per context size, then measured repetitions.
            for iteration in range(repetitions + 1):
                started = time.perf_counter()
                terminal = None
                async for event in client.stream(prompt, max_tokens=16):
                    if event["type"] == "done":
                        terminal = event
                if terminal is None or not terminal.get("memory_profile"):
                    raise ValueError("service returned no memory profile")
                if iteration:
                    records.append({"prompt_repeats": repeats, "iteration": iteration,
                                    "client_elapsed_s": time.perf_counter() - started,
                                    "prompt": prompt, "server": terminal})
    return {"schema_version": 1, "measurement_kind": "real_mlx_opt_in_cache_array_profile",
            "metadata": {"run_at_utc": datetime.now(timezone.utc).isoformat(), "server": server,
                         "repetitions": repetitions, "excluded_warmups_per_context": 1,
                         "limits": "Cache array nbytes includes spare capacity, is not unique physical GPU allocation, and profiling adds overhead. No unprofiled throughput comparison is implied."},
            "summary": summarize_profiles(records), "requests": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--output", default="results/cache_profile.json")
    args = parser.parse_args()
    if not 1 <= args.repeat <= 10:
        parser.error("repeat must be in 1..10")
    result = asyncio.run(run(args.url, repetitions=args.repeat))
    write_json(args.output, result)
    print(json.dumps(result["summary"], indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
