"""Bounded async workload runner; one JSON report includes raw request records."""
import argparse
import asyncio
import json
import math
import os
import platform
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import aiohttp

from .workloads import build_workload, fingerprint, validate_workload
from .reporting import write_json
from .journal import Journal
from .protocol import StreamProtocolError, StreamValidator, decode_event
from .environment import source_fingerprint


def percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    rank = (len(values) - 1) * fraction
    low = int(rank)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (rank - low)


def summarize(records, wall_s):
    if wall_s <= 0:
        raise ValueError("wall_s must be positive")
    completed = [r for r in records if r["ok"]]
    rejected = sum(r.get("error_code") == "overloaded" for r in records)
    latencies = [r["client_elapsed_s"] for r in completed]
    tokens = sum(r["server"]["generated_tokens"] for r in completed)
    inter_token = [value for record in completed
                   for value in record["server"].get("inter_token_latency_s", [])]
    phase_profiles = [record["server"].get("phase_profile", {}) for record in completed]
    result = {
        "successful_requests": len(completed), "failed_requests": len(records) - len(completed),
        "rejected_requests": rejected,
        "error_rate": (len(records) - len(completed)) / len(records) if records else None,
        "rejection_rate": rejected / len(records) if records else None,
        "wall_s": wall_s, "successful_requests_per_s": len(completed) / wall_s,
        "completed_output_tokens": tokens, "completed_output_tokens_per_s": tokens / wall_s,
        **{f"p{p}_client_latency_s": percentile(latencies, p / 100) for p in (50, 95, 99)},
        "p95_client_ttft_s": percentile([r["client_ttft_s"] for r in completed
                                         if r["client_ttft_s"] is not None], .95),
        "p95_queue_wait_s": percentile([r["server"]["queue_wait_s"] for r in completed], .95),
        "p95_inter_token_latency_s": percentile(inter_token, .95),
        "median_request_observed_prefill_execution_s": percentile(
            [profile["prefill_execution_s"] for profile in phase_profiles
             if profile.get("prefill_execution_s") is not None], .5),
        "median_request_observed_decode_execution_s": percentile(
            [profile["decode_execution_s"] for profile in phase_profiles
             if profile.get("decode_execution_s") is not None], .5),
        "max_queue_wait_s": max((r["server"]["queue_wait_s"] for r in completed), default=None),
        "p95_dispatch_lag_s": percentile([r.get("dispatch_lag_s", 0) for r in records], .95),
        "partial_output_tokens_on_failure": sum(r.get("received_output_tokens", 0)
                                                 for r in records if not r["ok"]),
    }
    return result


def class_summaries(records, wall_s):
    return {label: summarize([record for record in records if record["class"] == label], wall_s)
            for label in sorted({record["class"] for record in records})}


def runtime_profile_delta(before, after):
    """Return batch-wide synchronized totals for only this load window."""
    if not isinstance(before, dict) or not isinstance(after, dict):
        return None
    numeric = ("prefill_steps", "decode_steps", "prefill_execution_s",
               "decode_execution_s", "occupancy_sum", "steps")
    if not all(type(before.get(key)) in (int, float) and type(after.get(key)) in (int, float)
               for key in numeric):
        return None
    delta = {key: after[key] - before[key] for key in numeric}
    delta["mean_occupancy"] = delta["occupancy_sum"] / delta["steps"] if delta["steps"] else None
    delta["measurement_source"] = "difference_of_synchronized_runtime_counters"
    return delta


async def request_one(session, url, index, args, scheduled, experiment_start, item):
    started = time.perf_counter()
    record = {"index": index, "ok": False, "prompt": item["prompt"],
              "class": item["class"], "max_tokens": item["max_tokens"],
              "scheduled_offset_s": scheduled - experiment_start,
              "dispatch_offset_s": started - experiment_start,
              "dispatch_lag_s": max(0, started - scheduled),
              "client_ttft_s": None, "client_token_times_s": [], "received_output_tokens": 0}
    try:
        async with session.post(url + "/v1/generate", json={"prompt": record["prompt"],
                                "max_tokens": item["max_tokens"], "timeout_s": args.timeout},
                                timeout=aiohttp.ClientTimeout(total=args.timeout + 5)) as response:
            record["http_status"] = response.status
            record["request_id"] = response.headers.get("X-Request-ID")
            if response.status != 200:
                record["error"] = (await response.text())[:500]
                try:
                    record["error_code"] = json.loads(record["error"])["error"]["code"]
                except (ValueError, KeyError, TypeError):
                    record["error_code"] = "http_error"
            else:
                validator = StreamValidator(record["request_id"], item["max_tokens"])
                async for line in response.content:
                    event = decode_event(line)
                    validator.feed(event)
                    if event["type"] == "token" and event["output"]:
                        elapsed = time.perf_counter() - started
                        if record["client_ttft_s"] is None:
                            record["client_ttft_s"] = elapsed
                        record["client_token_times_s"].append(elapsed)
                        record["received_output_tokens"] += 1
                    elif event["type"] == "error":
                        record["error"] = event["error"]
                        record["error_code"] = event.get("code", "inference_error")
                        break
                    elif event["type"] == "done":
                        record["server"] = event
                        record["ok"] = event["generated_tokens"] == record["received_output_tokens"]
                        if not record["ok"]:
                            record["error"] = "client/server output token counts disagree"
                            record["error_code"] = "token_count_mismatch"
                        break
                if not record["ok"] and "error" not in record:
                    record["error"] = "stream ended without done event"
                    record["error_code"] = "incomplete_stream"
    except StreamProtocolError as exc:
        record["error"], record["error_code"] = str(exc), exc.code
    except (aiohttp.ClientError, TimeoutError, ValueError, KeyError, TypeError) as exc:
        record["error"] = repr(exc)
        record["error_code"] = "transport_error" if isinstance(exc, (aiohttp.ClientError, TimeoutError)) else "invalid_stream"
    record["client_elapsed_s"] = time.perf_counter() - started
    times = record["client_token_times_s"]
    record["client_inter_token_s"] = [b - a for a, b in zip(times, times[1:])]
    return record


def validate_config(args):
    for name in ("requests", "concurrency", "max_tokens"):
        if type(getattr(args, name)) is not int or getattr(args, name) < 1:
            raise ValueError(f"{name} must be a positive integer")
    if args.requests > 100000 or args.concurrency > 256 or args.max_tokens > 512:
        raise ValueError("limits: requests <= 100000, concurrency <= 256, max_tokens <= 512")
    if not math.isfinite(args.rate) or args.rate < 0 or not math.isfinite(args.timeout) or not 0 < args.timeout <= 120:
        raise ValueError("rate must be finite and nonnegative; timeout must be in (0, 120]")
    duration = getattr(args, "duration_s", None)
    if duration is not None and (not math.isfinite(duration) or not 0 < duration <= 3600):
        raise ValueError("duration must be finite and in (0, 3600] seconds")
    cap = getattr(args, "max_requests", 10000)
    if type(cap) is not int or not 1 <= cap <= 100000:
        raise ValueError("max_requests must be in 1..100000")
    if duration is None and args.requests > cap:
        raise ValueError("requests exceeds the configured max_requests safety cap")
    if getattr(args, "compact", False) and not getattr(args, "journal", None):
        raise ValueError("compact reports require a raw request journal")


async def run(args):
    validate_config(args)
    workload_path = getattr(args, "workload_file", None)
    if workload_path:
        plan = validate_workload(json.loads(Path(workload_path).read_text()))
        if len(plan) != args.requests:
            raise ValueError("--requests must match workload-file request count")
    else:
        plan = validate_workload(build_workload(args.requests, args.max_tokens,
                                              kind=getattr(args, "workload", "uniform"),
                                              seed=getattr(args, "seed", 42)))
    url = args.url.rstrip("/")
    connector = aiohttp.TCPConnector(limit=args.concurrency)
    api_key = os.environ.get("TENSORPILOT_API_KEY")
    headers = {"Authorization": "Bearer " + api_key} if api_key else {}
    async with aiohttp.ClientSession(connector=connector, headers=headers) as session:
        async with session.get(url + "/health", timeout=aiohttp.ClientTimeout(total=10)) as response:
            response.raise_for_status()
            server = await response.json()
            if not server.get("ready"):
                raise RuntimeError("server is not ready")
        duration = getattr(args, "duration_s", None)
        limit = args.requests if duration is None else getattr(args, "max_requests", 10000)
        metadata = {"run_id": uuid.uuid4().hex, "run_at_utc": datetime.now(timezone.utc).isoformat(),
                    "client_source_sha256": source_fingerprint(),
                    "client_platform": platform.platform(), "client_python": platform.python_version(),
                    "server": server, "config": vars(args),
                    "workload_sha256": fingerprint(plan), "workload": plan,
                    "workload_repeats": duration is not None,
                    "arrival_mode": "paced, concurrency-capped; inspect dispatch lag" if args.rate else "closed loop",
                    "request_records": "compact; full records in journal" if getattr(args, "compact", False) else "full",
                    "timings": "monotonic clock; client includes HTTP; server TTFT includes queue",
                    "throughput": "successful output tokens / measured workload wall time, including drain",
                    "limitations": "client chunks include network buffering; MLX memory is not total system pressure; inspect artifact provenance; journals contain prompt text"}
        journal_path = getattr(args, "journal", None)
        journal = Journal(journal_path, metadata, overwrite=getattr(args, "overwrite", False),
                          sync=getattr(args, "journal_fsync", False)) if journal_path else None
        # Fixed workers and a request cap bound task/record growth during soak runs.
        next_index = 0
        records = []
        start = time.perf_counter()
        stop_at = start + duration if duration is not None else math.inf

        async def worker():
            nonlocal next_index
            while next_index < limit and time.perf_counter() < stop_at:
                index = next_index
                next_index += 1
                scheduled = start + index / args.rate if args.rate else time.perf_counter()
                if scheduled >= stop_at:
                    return
                await asyncio.sleep(max(0, scheduled - time.perf_counter()))
                if time.perf_counter() >= stop_at:
                    return
                record = await request_one(session, url, index, args, scheduled, start, plan[index % len(plan)])
                if journal:
                    journal.record(record)
                if getattr(args, "compact", False):
                    for key in ("client_token_times_s", "client_inter_token_s"):
                        record.pop(key, None)
                    if "server" in record:
                        record["server"].pop("token_times_s", None)
                        record["server"].pop("memory_profile", None)
                records.append(record)

        tasks = [asyncio.create_task(worker()) for _ in range(args.concurrency)]
        status = "completed"
        try:
            await asyncio.gather(*tasks)
        except BaseException as exc:
            status = "interrupted" if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt)) else "failed"
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        finally:
            wall_s = time.perf_counter() - start
            if journal:
                journal.finish(status, wall_s)
        metadata["stop_reason"] = ("request_count_completed" if duration is None else
                                   "request_cap_reached" if len(records) >= limit else "duration_elapsed")
        try:
            async with session.get(url + "/health", timeout=aiohttp.ClientTimeout(total=10)) as response:
                response.raise_for_status()
                metadata["server_after"] = await response.json()
        except (aiohttp.ClientError, TimeoutError, ValueError):
            metadata["server_after"] = None
    metadata["runtime_profile_delta"] = runtime_profile_delta(
        server.get("runtime_profile"),
        (metadata.get("server_after") or {}).get("runtime_profile"))
    return {"schema_version": 3, "metadata": metadata,
            "summary": summarize(records, wall_s), "by_class": class_summaries(records, wall_s),
            "requests": sorted(records, key=lambda r: r["index"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--requests", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--rate", type=float, default=0, help="target arrivals/s; 0 selects closed loop")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=60)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parser.add_argument("--output", default=f"results/load_{stamp}_{uuid.uuid4().hex[:6]}.json")
    parser.add_argument("--workload", choices=("uniform", "mixed"), default="uniform")
    parser.add_argument("--workload-file", help="validated JSON plan; --requests must match its length")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duration", dest="duration_s", type=float,
                        help="repeat the request plan for this many seconds, then drain active requests")
    parser.add_argument("--max-requests", type=int, default=10000, help="safety cap, including soak mode")
    parser.add_argument("--journal", help="raw JSONL path; defaults beside the output report")
    parser.add_argument("--journal-fsync", action="store_true", help="fsync each record; changes workload overhead")
    parser.add_argument("--compact", action="store_true", help="keep token arrays only in the raw journal")
    parser.add_argument("--overwrite", action="store_true", help="explicitly replace an existing report/journal")
    args = parser.parse_args()
    path = Path(args.output)
    args.journal = args.journal or str(path.with_suffix(".requests.jsonl"))
    if Path(args.journal).resolve() == path.resolve():
        parser.error("journal and report must use different paths")
    if not args.overwrite and path.exists():
        parser.error("output already exists; choose a new path or use --overwrite")
    try:
        validate_config(args)
        result = asyncio.run(run(args))
    except KeyboardInterrupt:
        parser.exit(130, f"Interrupted; completed request records remain in {args.journal}\n")
    except (ValueError, OSError, RuntimeError, aiohttp.ClientError) as exc:
        parser.exit(1, f"Load failed: {exc}\n")
    write_json(path, result)
    print(json.dumps(result["summary"], indent=2))
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
