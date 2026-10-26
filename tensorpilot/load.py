"""Bounded async workload runner; one JSON report includes raw request records."""
import argparse
import asyncio
import json
import math
import os
import platform
import random
import statistics
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import aiohttp

from .workloads import build_spec_workload, build_workload, fingerprint, validate_workload
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


def outcome_of(record):
    """The single terminal outcome of a request. Records from older runs have only ``ok``."""
    return record.get("outcome") or ("completed" if record["ok"] else "failed")


def _mean_itl(record):
    return record.get("client_tpot_s")


def summarize(records, wall_s, slo=None):
    if wall_s <= 0:
        raise ValueError("wall_s must be positive")
    # A request the client cancelled on purpose is not an error; it is counted on its own.
    cancelled = [r for r in records if outcome_of(r) == "client_cancelled"]
    counted = [r for r in records if outcome_of(r) != "client_cancelled"]
    completed = [r for r in counted if r["ok"]]
    rejected = sum(r.get("error_code") == "overloaded" for r in counted)
    latencies = [r["client_elapsed_s"] for r in completed]
    tokens = sum(r["server"]["generated_tokens"] for r in completed)
    inter_token = [value for record in completed
                   for value in record["server"].get("inter_token_latency_s", [])]
    client_itl = [value for record in completed for value in record.get("client_inter_token_s", [])]
    ttfts = [r["client_ttft_s"] for r in completed if r["client_ttft_s"] is not None]
    # Only TensorPilot reports queue wait; an OpenAI-compatible server does not.
    queue_waits = [r["server"]["queue_wait_s"] for r in completed if r["server"].get("queue_wait_s") is not None]
    phase_profiles = [record["server"].get("phase_profile", {}) for record in completed]
    # Active window: first token produced to last token produced, across all completed requests. Unlike
    # wall time it excludes ramp-up (queue fill) and the drain tail, so it approximates steady throughput.
    firsts = [r["dispatch_offset_s"] + r["client_ttft_s"] for r in completed
              if r.get("dispatch_offset_s") is not None and r.get("client_ttft_s") is not None]
    lasts = [r["dispatch_offset_s"] + r["client_last_token_s"] for r in completed
             if r.get("dispatch_offset_s") is not None and r.get("client_last_token_s") is not None]
    window = (max(lasts) - min(firsts)) if firsts and lasts else None
    result = {
        "requests_total": len(records),
        "successful_requests": len(completed), "failed_requests": len(counted) - len(completed),
        "rejected_requests": rejected,
        "client_cancelled_requests": len(cancelled),
        "timed_out_requests": sum(outcome_of(r) == "timed_out" for r in counted),
        "outcomes": dict(sorted(Counter(outcome_of(r) for r in records).items())),
        "errors_by_code": dict(sorted(Counter(r.get("error_code") for r in counted if not r["ok"]).items(),
                                      key=lambda item: str(item[0]))),
        "error_rate": (len(counted) - len(completed)) / len(counted) if counted else None,
        "rejection_rate": rejected / len(counted) if counted else None,
        "wall_s": wall_s, "successful_requests_per_s": len(completed) / wall_s,
        "completed_output_tokens": tokens, "completed_output_tokens_per_s": tokens / wall_s,
        "active_window_s": window,
        "active_window_output_tokens_per_s": tokens / window if window and window > 0 else None,
        "mean_output_tokens_per_request": tokens / len(completed) if completed else None,
        **{f"p{p}_client_latency_s": percentile(latencies, p / 100) for p in (50, 95, 99)},
        **{f"p{p}_client_ttft_s": percentile(ttfts, p / 100) for p in (50, 95, 99)},
        **{f"p{p}_client_tpot_s": percentile([v for v in map(_mean_itl, completed) if v is not None], p / 100)
           for p in (50, 95, 99)},
        **{f"p{p}_client_inter_token_s": percentile(client_itl, p / 100) for p in (50, 95, 99)},
        "p95_queue_wait_s": percentile(queue_waits, .95),
        "p99_queue_wait_s": percentile(queue_waits, .99),
        "p95_inter_token_latency_s": percentile(inter_token, .95),
        "median_request_observed_prefill_execution_s": percentile(
            [profile["prefill_execution_s"] for profile in phase_profiles
             if profile.get("prefill_execution_s") is not None], .5),
        "median_request_observed_decode_execution_s": percentile(
            [profile["decode_execution_s"] for profile in phase_profiles
             if profile.get("decode_execution_s") is not None], .5),
        "max_queue_wait_s": max(queue_waits, default=None),
        "p95_dispatch_lag_s": percentile([r.get("dispatch_lag_s", 0) for r in records], .95),
        "partial_output_tokens_on_failure": sum(r.get("received_output_tokens", 0)
                                                 for r in counted if not r["ok"]),
    }
    if slo:
        def good(record):
            if not record["ok"]:
                return False
            if "ttft_s" in slo and (record["client_ttft_s"] is None or record["client_ttft_s"] > slo["ttft_s"]):
                return False
            if "e2e_s" in slo and record["client_elapsed_s"] > slo["e2e_s"]:
                return False
            if "tpot_s" in slo and (record.get("client_tpot_s") or 0) > slo["tpot_s"]:
                return False
            return True

        winners = [r for r in counted if good(r)]
        result["slo"] = {"definition": dict(slo), "good_requests": len(winners),
                         "attainment": len(winners) / len(counted) if counted else None,
                         "goodput_requests_per_s": len(winners) / wall_s,
                         "goodput_output_tokens_per_s":
                             sum(r["server"]["generated_tokens"] for r in winners) / wall_s,
                         "note": "a request is good only if it completed and met every declared threshold"}
    return result


def class_summaries(records, wall_s, slo=None):
    return {label: summarize([record for record in records if record["class"] == label], wall_s, slo)
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
    for field in ("prompt_tokens", "cancel_after_tokens", "cancel_after_s"):
        if field in item:
            record[field] = item[field]
    timeout_s = item.get("timeout_s", args.timeout)
    cancel_tokens, cancel_s = item.get("cancel_after_tokens"), item.get("cancel_after_s")
    scope = None
    try:
        # Client-side cancellation timer: it can fire while the request is queued or while it decodes.
        async with asyncio.timeout(cancel_s) as scope:
            async with session.post(url + "/v1/generate", json={"prompt": record["prompt"],
                                    "max_tokens": item["max_tokens"], "timeout_s": timeout_s},
                                    timeout=aiohttp.ClientTimeout(total=timeout_s + 5)) as response:
                record["http_status"] = response.status
                record["request_id"] = response.headers.get("X-Request-ID")
                record["worker_id"] = response.headers.get("X-TensorPilot-Worker")
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
                            if cancel_tokens and record["received_output_tokens"] >= cancel_tokens:
                                record["cancelled"] = True  # leaving the stream closes the connection
                                break
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
                    if not record["ok"] and "error" not in record and not record.get("cancelled"):
                        record["error"] = "stream ended without done event"
                        record["error_code"] = "incomplete_stream"
    except StreamProtocolError as exc:
        record["error"], record["error_code"] = str(exc), exc.code
    except (aiohttp.ClientError, TimeoutError, ValueError, KeyError, TypeError) as exc:
        if isinstance(exc, TimeoutError) and scope is not None and scope.expired():
            record["cancelled"] = True
        else:
            record["error"] = repr(exc)
            record["error_code"] = ("transport_error" if isinstance(exc, (aiohttp.ClientError, TimeoutError))
                                    else "invalid_stream")
            record["client_timeout"] = isinstance(exc, TimeoutError)
    return _finish_record(record, timeout_s, started)


def _finish_record(record, timeout_s, started):
    record["client_elapsed_s"] = time.perf_counter() - started
    times = record["client_token_times_s"]
    record["client_inter_token_s"] = [b - a for a, b in zip(times, times[1:])]
    record["client_tpot_s"] = statistics.fmean(record["client_inter_token_s"]) if record["client_inter_token_s"] else None
    record["client_last_token_s"] = times[-1] if times else None
    if record["ok"]:
        record["outcome"] = "completed"
    elif record.get("cancelled"):
        record["outcome"] = "client_cancelled"
        record.pop("error", None)
        record.pop("error_code", None)
    elif record.get("error_code") == "overloaded":
        record["outcome"] = "rejected"
    elif (record.get("error_code") == "deadline_exceeded" or record.get("client_timeout")
          or (record.get("error_code") == "incomplete_stream" and record["client_elapsed_s"] >= 0.9 * timeout_s)):
        # The server's handler closes a stream at the deadline without a terminal event, so a stream that
        # ends near the client's own deadline is a timeout, not a transport failure.
        record["outcome"] = "timed_out"
    else:
        record["outcome"] = "failed"
    return record


OPENAI_SAMPLING = {"temperature": 0, "top_p": 1.0, "top_k": -1, "repetition_penalty": 1.0}


def _openai_error(record, status, text):
    record["http_status"] = status
    record["error"] = text[:500]
    try:
        body = json.loads(text)["error"]
        record["error_code"] = str(body.get("code") or body.get("type") or "http_error")
    except (ValueError, KeyError, TypeError, AttributeError):
        record["error_code"] = "http_error"
    if status == 503 or record["error_code"] in ("overloaded", "server_busy"):
        record["error_code"] = "overloaded"


async def request_openai(session, url, index, args, scheduled, experiment_start, item):
    """The same request against an OpenAI-compatible server (vLLM), producing the same record shape.

    Sampling is pinned to greedy and every sampling override is explicit: vLLM otherwise applies the model
    repository's generation_config.json (Qwen2.5-Instruct ships repetition_penalty 1.05, top_k 20, top_p 0.8),
    which would make the two engines generate different text. Token arrivals are SSE chunks that carry text;
    the generated-token count is the server's own usage figure.
    """
    started = time.perf_counter()
    record = {"index": index, "ok": False, "prompt": item["prompt"], "class": item["class"],
              "max_tokens": item["max_tokens"], "scheduled_offset_s": scheduled - experiment_start,
              "dispatch_offset_s": started - experiment_start, "dispatch_lag_s": max(0, started - scheduled),
              "client_ttft_s": None, "client_token_times_s": [], "received_output_tokens": 0}
    for field in ("prompt_tokens", "cancel_after_tokens", "cancel_after_s"):
        if field in item:
            record[field] = item[field]
    timeout_s = item.get("timeout_s", args.timeout)
    cancel_tokens, cancel_s = item.get("cancel_after_tokens"), item.get("cancel_after_s")
    body = {"model": args.model_name, "prompt": item["prompt"], "max_tokens": item["max_tokens"], "stream": True,
            "stream_options": {"include_usage": True}, **OPENAI_SAMPLING}
    usage, finish, done, scope = None, None, False, None
    try:
        async with asyncio.timeout(cancel_s) as scope:
            async with session.post(url + "/v1/completions", json=body,
                                    timeout=aiohttp.ClientTimeout(total=timeout_s + 5)) as response:
                record["http_status"] = response.status
                record["request_id"] = response.headers.get("X-Request-ID")
                if response.status != 200:
                    _openai_error(record, response.status, await response.text())
                else:
                    async for raw in response.content:
                        line = raw.decode("utf-8", "replace").strip()
                        if not line.startswith("data:"):
                            continue
                        payload = line[5:].strip()
                        if payload == "[DONE]":
                            done = True
                            break
                        event = json.loads(payload)
                        if event.get("error"):
                            record["error"] = str(event["error"])[:500]
                            record["error_code"] = "inference_error"
                            break
                        if event.get("usage"):
                            usage = event["usage"]
                        for choice in event.get("choices") or []:
                            if choice.get("finish_reason"):
                                finish = choice["finish_reason"]
                            if choice.get("text"):
                                elapsed = time.perf_counter() - started
                                if record["client_ttft_s"] is None:
                                    record["client_ttft_s"] = elapsed
                                record["client_token_times_s"].append(elapsed)
                                record["received_output_tokens"] += 1
                                if cancel_tokens and record["received_output_tokens"] >= cancel_tokens:
                                    record["cancelled"] = True
                                    break
                        if record.get("cancelled"):
                            break
                    if not record.get("cancelled") and "error" not in record:
                        if done and finish and usage:
                            record["ok"] = True
                            record["server"] = {"generated_tokens": usage["completion_tokens"],
                                                "prompt_tokens": usage.get("prompt_tokens"), "finish_reason": finish,
                                                "queue_wait_s": None, "api": "openai"}
                        else:
                            record["error"] = "stream ended without finish, usage and [DONE]"
                            record["error_code"] = "incomplete_stream"
    except (aiohttp.ClientError, TimeoutError, ValueError, KeyError, TypeError) as exc:
        if isinstance(exc, TimeoutError) and scope is not None and scope.expired():
            record["cancelled"] = True
        else:
            record["error"] = repr(exc)
            record["error_code"] = ("transport_error" if isinstance(exc, (aiohttp.ClientError, TimeoutError))
                                    else "invalid_stream")
            record["client_timeout"] = isinstance(exc, TimeoutError)
    return _finish_record(record, timeout_s, started)


def arrival_offsets(args, count):
    """Scheduled arrival offsets in seconds for open-loop runs; None for closed loop."""
    rate = args.rate
    if not rate:
        return None
    mode = getattr(args, "arrival", "uniform")
    if mode == "uniform":
        return [index / rate for index in range(count)]
    if mode == "poisson":
        rng, clock, offsets = random.Random(getattr(args, "seed", 42)), 0.0, []
        for _ in range(count):
            offsets.append(clock)
            clock += rng.expovariate(rate)
        return offsets
    if mode == "burst":
        size = getattr(args, "burst_size", 8)
        return [(index // size) * (size / rate) for index in range(count)]
    raise ValueError("arrival must be uniform, poisson or burst")


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
    if getattr(args, "api", "tensorpilot") not in ("tensorpilot", "openai"):
        raise ValueError("api must be tensorpilot or openai")
    if getattr(args, "api", "tensorpilot") == "openai" and not getattr(args, "model_name", None):
        raise ValueError("--model-name is required with --api openai")
    if getattr(args, "arrival", "uniform") not in ("uniform", "poisson", "burst"):
        raise ValueError("arrival must be uniform, poisson or burst")
    if getattr(args, "arrival", "uniform") != "uniform" and not args.rate:
        raise ValueError("poisson and burst arrivals need a nonzero --rate")
    if type(getattr(args, "burst_size", 8)) is not int or not 1 <= getattr(args, "burst_size", 8) <= 1024:
        raise ValueError("burst_size must be in 1..1024")
    slo = getattr(args, "slo", None)
    if slo is not None:
        if not isinstance(slo, dict) or not slo or set(slo) - {"ttft_s", "e2e_s", "tpot_s"}:
            raise ValueError("slo must be an object with ttft_s, e2e_s and/or tpot_s")
        if not all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in slo.values()):
            raise ValueError("slo thresholds must be positive finite numbers")


async def _server_state(session, url, api):
    """TensorPilot's /health, or for an OpenAI-compatible server its model list (it has no health document)."""
    timeout = aiohttp.ClientTimeout(total=10)
    if api != "openai":
        async with session.get(url + "/health", timeout=timeout) as response:
            response.raise_for_status()
            return await response.json()
    async with session.get(url + "/v1/models", timeout=timeout) as response:
        response.raise_for_status()
        models = (await response.json()).get("data") or []
    return {"ready": bool(models), "api": "openai", "service": "openai-compatible",
            "models": [{"id": m.get("id"), "max_model_len": m.get("max_model_len")} for m in models]}


async def run(args):
    validate_config(args)
    workload_path = getattr(args, "workload_file", None)
    spec_path = getattr(args, "workload_spec", None)
    if spec_path:
        plan = validate_workload(build_spec_workload(json.loads(Path(spec_path).read_text()), args.requests))
    elif workload_path:
        plan = validate_workload(json.loads(Path(workload_path).read_text()))
        if len(plan) != args.requests:
            raise ValueError("--requests must match workload-file request count")
    else:
        plan = validate_workload(build_workload(args.requests, args.max_tokens,
                                              kind=getattr(args, "workload", "uniform"),
                                              seed=getattr(args, "seed", 42)))
    url = args.url.rstrip("/")
    api = getattr(args, "api", "tensorpilot")
    request_client = request_openai if api == "openai" else request_one
    # No keep-alive: the server force-closes a connection when a request hits its deadline, and a pooled
    # connection reused right after would fail the next, unrelated request. Each request pays one loopback connect.
    connector = aiohttp.TCPConnector(limit=args.concurrency, force_close=True)
    api_key = os.environ.get("TENSORPILOT_API_KEY")
    headers = {"Authorization": "Bearer " + api_key} if api_key else {}
    async with aiohttp.ClientSession(connector=connector, headers=headers) as session:
        server = await _server_state(session, url, api)
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
                    "arrival_mode": (f"{getattr(args, 'arrival', 'uniform')} arrivals, concurrency-capped; inspect "
                                     "dispatch lag" if args.rate else "closed loop"),
                    "request_records": "compact; full records in journal" if getattr(args, "compact", False) else "full",
                    "timings": "monotonic clock; client includes HTTP and a fresh connection per request; server TTFT includes queue",
                    "throughput": "successful output tokens / measured workload wall time, including drain",
                    "limitations": "client chunks include network buffering; MLX memory is not total system pressure; inspect artifact provenance; journals contain prompt text"}
        journal_path = getattr(args, "journal", None)
        journal = Journal(journal_path, metadata, overwrite=getattr(args, "overwrite", False),
                          sync=getattr(args, "journal_fsync", False)) if journal_path else None
        # Fixed workers and a request cap bound task/record growth during soak runs.
        next_index = 0
        records = []
        offsets = arrival_offsets(args, limit)
        start = time.perf_counter()
        stop_at = start + duration if duration is not None else math.inf

        async def worker():
            nonlocal next_index
            while next_index < limit and time.perf_counter() < stop_at:
                index = next_index
                next_index += 1
                scheduled = start + offsets[index] if offsets else time.perf_counter()
                if scheduled >= stop_at:
                    return
                await asyncio.sleep(max(0, scheduled - time.perf_counter()))
                if time.perf_counter() >= stop_at:
                    return
                record = await request_client(session, url, index, args, scheduled, start, plan[index % len(plan)])
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
            metadata["server_after"] = await _server_state(session, url, api)
        except (aiohttp.ClientError, TimeoutError, ValueError):
            metadata["server_after"] = None
    metadata["runtime_profile_delta"] = runtime_profile_delta(
        server.get("runtime_profile"),
        (metadata.get("server_after") or {}).get("runtime_profile"))
    return {"schema_version": 3, "metadata": metadata,
            "summary": summarize(records, wall_s, getattr(args, "slo", None)),
            "by_class": class_summaries(records, wall_s, getattr(args, "slo", None)),
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
    parser.add_argument("--api", choices=("tensorpilot", "openai"), default="tensorpilot",
                        help="openai drives an OpenAI-compatible server (vLLM) with the same workload and metrics")
    parser.add_argument("--model-name", help="model id sent to an OpenAI-compatible server")
    parser.add_argument("--workload-spec", help="declarative workload spec (JSON); builds --requests items")
    parser.add_argument("--arrival", choices=("uniform", "poisson", "burst"), default="uniform",
                        help="open-loop arrival process when --rate is nonzero")
    parser.add_argument("--burst-size", type=int, default=8, help="requests per burst for --arrival burst")
    parser.add_argument("--slo-ttft", type=float, help="goodput: max time to first token, seconds")
    parser.add_argument("--slo-e2e", type=float, help="goodput: max end-to-end latency, seconds")
    parser.add_argument("--slo-tpot", type=float, help="goodput: max mean inter-token latency, seconds")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duration", dest="duration_s", type=float,
                        help="repeat the request plan for this many seconds, then drain active requests")
    parser.add_argument("--max-requests", type=int, default=10000, help="safety cap, including soak mode")
    parser.add_argument("--journal", help="raw JSONL path; defaults beside the output report")
    parser.add_argument("--journal-fsync", action="store_true", help="fsync each record; changes workload overhead")
    parser.add_argument("--compact", action="store_true", help="keep token arrays only in the raw journal")
    parser.add_argument("--overwrite", action="store_true", help="explicitly replace an existing report/journal")
    args = parser.parse_args()
    slo = {key: value for key, value in (("ttft_s", args.slo_ttft), ("e2e_s", args.slo_e2e),
                                         ("tpot_s", args.slo_tpot)) if value is not None}
    args.slo = slo or None
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
