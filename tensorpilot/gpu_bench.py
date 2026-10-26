"""Configurable single-GPU experiments: plan GPU time, then run them against a live worker.

``plan`` never touches a GPU. It converts the configuration into request counts and
a rough GPU-time estimate, labeled as an assumption. ``run`` drives the production
worker through the existing load runner and writes one report per repeat, with
the server's own health snapshot (model revision, artifact hash, GPU identity,
software versions, and nvidia-smi telemetry) captured before and after.

Experiments that cannot run in this release are listed with status ``planned``.
``run`` refuses them rather than producing a result that looks like evidence.
"""
import argparse
import asyncio
import hashlib
import json
import math
import os
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from . import load, telemetry
from .reporting import write_json
from .workloads import build_spec_workload, build_stress_workload, validate_workload

SCHEMA = "tensorpilot.gpu-experiments/v1"
REVISION = re.compile(r"[0-9a-f]{40}")
RUNNABLE_KINDS = ("closed_loop", "open_loop", "saturation", "length_matrix")
PLANNED_KINDS = ("recovery_manual", "batch_width_planned", "two_gpu_planned")
ALL_KINDS = RUNNABLE_KINDS + PLANNED_KINDS
EXPERIMENT_KEYS = {"id", "kind", "status", "description", "requests", "concurrency", "prompt_repeats",
                   "max_tokens", "repeats", "warmup_requests", "timeout_s", "rates", "steps", "reason",
                   "estimated_minutes", "minimum_gpus", "max_minutes", "abort_error_rate", "workload_spec",
                   "arrival", "burst_size", "slo", "telemetry"}
LIMITATIONS = (
    "The worker executes one sequence at a time; concurrency above one measures queueing, not batched throughput.",
    "Client timings include loopback HTTP and streaming parse time on the load-generating host.",
    "One model on one GPU; no multi-GPU or scaling claim.",
    "CUDA peak memory is process-lifetime from the PyTorch caching allocator, not total device pressure.",
    "Planning estimates are assumptions and are never reported as measurements.",
    "GPU clocks, temperature and co-tenancy are recorded by nvidia-smi samples, not controlled.",
)
BATCH_LIMITATIONS = (
    "The worker batches sequences in one forward pass over a fixed-slot KV pool (not paged); each active sequence "
    "reserves a full row, so concurrency is bounded by slots rather than by tokens in use.",
    "Client timings include loopback HTTP, a fresh connection per request and streaming parse time on the "
    "load-generating host.",
    "One model on one GPU; no multi-GPU or scaling claim.",
    "CUDA peak memory is process-lifetime from the PyTorch caching allocator, not total device pressure.",
    "Planning estimates are assumptions and are never reported as measurements.",
    "GPU clocks, temperature and co-tenancy are recorded by nvidia-smi samples, not controlled.",
)
SMALL_SAMPLE = 100  # below this many completed requests, p95 and p99 are effectively the maximum
COUNTERS = ("tensorpilot_jobs_total", "tensorpilot_rejected_requests_total", "tensorpilot_admission_decisions_total",
            "tensorpilot_output_tokens_total", "tensorpilot_worker_stalls_total")
DEFAULT_BUDGET_MINUTES = 300
# Preflight, model load, warmup, evidence export and teardown, excluding experiment time.
DEFAULT_SESSION_OVERHEAD_MINUTES = 45


def load_config(path):
    config = json.loads(Path(path).read_text())
    validate_config(config)
    return config


def validate_config(config):
    if not isinstance(config, dict) or config.get("schema") != SCHEMA:
        raise ValueError(f"config schema must be {SCHEMA}")
    target = config.get("target")
    if not isinstance(target, dict) or not isinstance(target.get("model"), str) or not target["model"]:
        raise ValueError("target.model is required")
    assumptions = config.get("assumptions")
    if not isinstance(assumptions, dict):
        raise ValueError("assumptions is required; planning numbers are labeled as assumptions")
    for name in ("decode_tokens_per_s", "prefill_tokens_per_s"):
        if not _positive_number(assumptions.get(name)):
            raise ValueError(f"assumptions.{name} must be a positive number (planning only)")
    if assumptions.get("status") != "placeholder_unmeasured":
        raise ValueError("assumptions.status must be placeholder_unmeasured until a preflight measures them")
    if "telemetry" in config:
        sampling = config["telemetry"]
        if not isinstance(sampling, dict) or set(sampling) - {"interval_s"} or (
                "interval_s" in sampling and not (_positive_number(sampling["interval_s"]) and sampling["interval_s"] <= 60)):
            raise ValueError("telemetry must be an object with interval_s in (0, 60]")
    if "max_session_minutes" in config and not _positive_number(config["max_session_minutes"]):
        raise ValueError("max_session_minutes must be a positive number")
    if "session_overhead_minutes" in config and not _positive_number(config["session_overhead_minutes"]):
        raise ValueError("session_overhead_minutes must be a positive number")
    experiments = config.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("experiments must be a nonempty list")
    seen = set()
    for experiment in experiments:
        _validate_experiment(experiment)
        if experiment["id"] in seen:
            raise ValueError(f"duplicate experiment id {experiment['id']}")
        seen.add(experiment["id"])


def _validate_experiment(experiment):
    if not isinstance(experiment, dict):
        raise ValueError("each experiment must be an object")
    unknown = set(experiment) - EXPERIMENT_KEYS
    if unknown:
        raise ValueError(f"unknown experiment fields: {sorted(unknown)}")
    identifier = experiment.get("id")
    if not isinstance(identifier, str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,62}", identifier):
        raise ValueError("experiment id must be lowercase letters, digits and hyphens")
    kind = experiment.get("kind")
    if kind not in ALL_KINDS:
        raise ValueError(f"unknown kind for {identifier}: {kind}")
    if kind in PLANNED_KINDS:
        if experiment.get("status") != "planned" or not experiment.get("reason"):
            raise ValueError(f"{identifier}: {kind} must declare status planned with a reason")
        if kind == "two_gpu_planned" and not _bounded_int(experiment.get("minimum_gpus"), 2, 8):
            raise ValueError(f"{identifier}: minimum_gpus must be 2..8")
        return
    if experiment.get("status") not in (None, "runnable"):
        raise ValueError(f"{identifier}: runnable experiments may not declare status {experiment['status']}")
    checks = {"requests": (1, 20000), "concurrency": (1, 256), "prompt_repeats": (1, 256),
              "max_tokens": (1, 512), "repeats": (1, 10), "warmup_requests": (0, 50),
              "timeout_s": (1, 120), "max_minutes": (1, 240)}
    if "abort_error_rate" in experiment:
        rate = experiment["abort_error_rate"]
        if type(rate) not in (int, float) or not 0 <= rate <= 1:
            raise ValueError(f"{identifier}: abort_error_rate must be in [0, 1]")
    if "max_minutes" not in experiment and kind in RUNNABLE_KINDS:
        raise ValueError(f"{identifier}: max_minutes is required; every experiment has a time cap")
    for field, (low, high) in checks.items():
        if field in experiment and not _bounded_int(experiment[field], low, high):
            raise ValueError(f"{identifier}: {field} must be an integer in {low}..{high}")
    if "workload_spec" in experiment:
        try:
            build_spec_workload(experiment["workload_spec"], 16)
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"{identifier}: invalid workload_spec: {exc}") from exc
        if not _positive_number(experiment.get("estimated_minutes")):
            raise ValueError(f"{identifier}: a workload_spec experiment must declare estimated_minutes")
        required = ("requests",)
    else:
        required = ("requests", "prompt_repeats", "max_tokens")
    for name in required:
        if name not in experiment:
            raise ValueError(f"{identifier}: {name} is required")
    if "arrival" in experiment:
        if experiment["arrival"] not in ("uniform", "poisson", "burst") or kind not in ("open_loop", "saturation"):
            raise ValueError(f"{identifier}: arrival is uniform, poisson or burst and only valid for open loops")
    if "burst_size" in experiment and not _bounded_int(experiment["burst_size"], 1, 1024):
        raise ValueError(f"{identifier}: burst_size must be an integer in 1..1024")
    if "slo" in experiment:
        slo = experiment["slo"]
        if not isinstance(slo, dict) or not slo or set(slo) - {"ttft_s", "e2e_s", "tpot_s"} or not all(
                _positive_number(v) for v in slo.values()):
            raise ValueError(f"{identifier}: slo needs positive ttft_s, e2e_s and/or tpot_s")
    if "telemetry" in experiment and type(experiment["telemetry"]) is not bool:
        raise ValueError(f"{identifier}: telemetry must be true or false")
    if kind == "open_loop" or kind == "saturation":
        rates = experiment.get("rates")
        if not isinstance(rates, list) or not rates or len(rates) > 8 or not all(
                _positive_number(rate) and rate <= 50 for rate in rates):
            raise ValueError(f"{identifier}: rates must be 1..8 positive numbers up to 50 per second")
    elif "rates" in experiment:
        raise ValueError(f"{identifier}: rates is only valid for open_loop and saturation")


def _bounded_int(value, low, high):
    return type(value) is int and low <= value <= high


def _positive_number(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def estimate(config, experiment):
    """Planning-only GPU minutes from declared assumptions; no GPU is touched."""
    assumptions = config["assumptions"]
    if experiment["kind"] in PLANNED_KINDS:
        minutes = experiment.get("estimated_minutes", 0) if experiment["kind"] == "recovery_manual" else 0
        return {"minutes": float(minutes), "note": "manual step; not run by this command"}
    if "estimated_minutes" in experiment:
        return {"minutes": float(min(experiment["estimated_minutes"], experiment.get("max_minutes",
                                                                                  experiment["estimated_minutes"]))),
                "note": "declared in the config; an assumption, not a measurement"}
    prompt_tokens = experiment["prompt_repeats"] * 17  # roughly 17 tokens per stress observation
    prefill_s = prompt_tokens / assumptions["prefill_tokens_per_s"]
    service_s = prefill_s + experiment["max_tokens"] / assumptions["decode_tokens_per_s"]
    repeats = experiment.get("repeats", 1)
    warmup = experiment.get("warmup_requests", 0)
    if experiment["kind"] in ("open_loop", "saturation"):
        rates = experiment["rates"]
        seconds = sum(repeats * max(experiment["requests"] / rate, experiment["requests"] * service_s)
                      for rate in rates)
    else:
        seconds = repeats * experiment["requests"] * service_s
    seconds += warmup * service_s
    minutes = round(seconds / 60 + 1, 2)
    if "max_minutes" in experiment and minutes > experiment["max_minutes"]:
        minutes = float(experiment["max_minutes"])  # The runner enforces this cap.
    return {"minutes": minutes, "note": "ESTIMATE from placeholder assumptions; not a measurement"}


def plan(config, budget_minutes=DEFAULT_BUDGET_MINUTES):
    rows, total = [], 0.0
    for experiment in config["experiments"]:
        cost = estimate(config, experiment)
        total += cost["minutes"]
        rows.append({"id": experiment["id"], "kind": experiment["kind"],
                     "status": experiment.get("status", "runnable"),
                     "estimated_gpu_minutes": cost["minutes"], "note": cost["note"]})
    overhead = config.get("session_overhead_minutes", DEFAULT_SESSION_OVERHEAD_MINUTES)
    session_minutes = total + overhead
    return {"schema": SCHEMA, "rows": rows, "experiment_minutes_estimate": round(total, 2),
            "assumptions_status": config["assumptions"]["status"],
            "session_overhead_minutes": overhead, "session_minutes_estimate": round(session_minutes, 2),
            "budget_minutes": budget_minutes, "within_budget": session_minutes <= budget_minutes,
            "note": "Estimates depend on the declared assumptions and are not measurements."}


def config_fingerprint(config):
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_plan_for(experiment):
    if "workload_spec" in experiment:
        return validate_workload(build_spec_workload(experiment["workload_spec"], experiment["requests"]))
    return validate_workload(build_stress_workload(
        experiment["requests"], prompt_repeats=experiment["prompt_repeats"],
        output_tokens=experiment["max_tokens"]))


def _namespace(url, plan_path, journal, *, requests, concurrency, rate, max_tokens, timeout, arrival="uniform",
               burst_size=8, slo=None, seed=42):
    return argparse.Namespace(
        url=url, requests=requests, concurrency=concurrency, rate=rate, max_tokens=max_tokens,
        timeout=timeout, workload="uniform", workload_file=str(plan_path), seed=seed, duration_s=None,
        max_requests=requests, journal=str(journal), journal_fsync=False, compact=False,
        overwrite=True, arrival=arrival, burst_size=burst_size, slo=slo)


def limitations_for(server):
    """The limitation text that matches how the worker actually executes; serial keeps its original wording."""
    return list(BATCH_LIMITATIONS if (server or {}).get("execution_mode") in ("batch", "adaptive") else LIMITATIONS)


def statistics_caveat(summary):
    completed = summary.get("successful_requests", 0)
    if completed < SMALL_SAMPLE:
        return (f"small sample: {completed} completed requests (< {SMALL_SAMPLE}); p95 and p99 are effectively the "
                "maximum and must not be quoted as tail-latency estimates")
    return f"{completed} completed requests; percentiles are estimates from a single repeat unless repeats > 1"


def parse_counters(text):
    """Selected Prometheus counters as {'name{label=value}': float}; anything else is ignored."""
    from prometheus_client.parser import text_string_to_metric_families
    values = {}
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            if sample.name in COUNTERS:
                labels = ",".join(f"{k}={v}" for k, v in sorted(sample.labels.items()))
                values[f"{sample.name}{{{labels}}}" if labels else sample.name] = sample.value
    return values


async def scrape_counters(url, api_key):
    import aiohttp
    headers = {"Authorization": "Bearer " + api_key} if api_key else {}
    try:
        async with aiohttp.ClientSession(headers=headers, timeout=aiohttp.ClientTimeout(total=5)) as session:
            async with session.get(url + "/metrics") as response:
                response.raise_for_status()
                return parse_counters(await response.text())
    except Exception:
        return None


def counter_delta(before, after):
    if before is None or after is None:
        return None
    return {key: after[key] - before.get(key, 0) for key in after if after[key] != before.get(key, 0)}


def _write_plan(directory, name, plan):
    path = directory / f"{name}.workload.json"
    path.write_text(json.dumps(plan, indent=2) + "\n")
    return path


DEFAULT_ABORT_ERROR_RATE = .05


def _abort_reason(result, abort_error_rate):
    """Return why an experiment must stop now, or None. Checked after every report."""
    after = result["metadata"].get("server_after")
    if not after or not after.get("ready"):
        return "worker_not_ready_after_run"
    before = result["metadata"].get("server") or {}
    oom_before = (before.get("last_measured_memory") or {}).get("cuda_oom_events") or 0
    oom_after = ((after.get("last_measured_memory") or {}).get("cuda_oom_events")) or 0
    if oom_after > oom_before:
        return "cuda_out_of_memory"
    error_rate = result["summary"]["error_rate"]
    if error_rate is not None and error_rate > abort_error_rate:
        return f"error_rate_{error_rate:.3f}_above_{abort_error_rate}"
    return None


async def run_experiment(config, experiment, *, url, output_dir, config_sha256, allowed_seconds):
    """Run one experiment under its own cap; the session budget is passed in as allowed_seconds."""
    experiment_dir = Path(output_dir) / experiment["id"]
    experiment_dir.mkdir(parents=True, exist_ok=False)
    timeout = experiment.get("timeout_s", 60)
    concurrency = experiment.get("concurrency", 1)
    max_tokens = experiment.get("max_tokens", 512)
    abort_rate = experiment.get("abort_error_rate", DEFAULT_ABORT_ERROR_RATE)
    cap_s = min(experiment["max_minutes"] * 60, allowed_seconds)
    started = time.monotonic()
    artifacts, status, reason = [], "completed", None
    plan = build_plan_for(experiment)
    api_key = os.environ.get("TENSORPILOT_API_KEY")
    extras = {"arrival": experiment.get("arrival", "uniform"), "burst_size": experiment.get("burst_size", 8),
              "slo": experiment.get("slo")}
    collector, overhead = None, None
    if config.get("telemetry") and experiment.get("telemetry", True):
        collector = telemetry.Collector(experiment_dir / "telemetry", url=url, api_key=api_key,
                                        interval_s=config["telemetry"].get("interval_s", 1.0))
        await collector.start()
    try:
        if experiment.get("warmup_requests", 0):
            if collector:
                collector.mark("warmup_start")
            warm_path = _write_plan(experiment_dir, "warmup", build_plan_for(
                {**experiment, "requests": experiment["warmup_requests"]}))
            journal = experiment_dir / "warmup.requests.jsonl"
            await asyncio.wait_for(load.run(_namespace(
                url, warm_path, journal, requests=experiment["warmup_requests"], concurrency=1, rate=0,
                max_tokens=max_tokens, timeout=timeout)), cap_s)
            if collector:
                collector.mark("warmup_end")
        if experiment["kind"] in ("open_loop", "saturation"):
            labels = [(f"rate{str(rate).replace('.', 'p')}", rate) for rate in experiment["rates"]]
        else:
            labels = [("closed", 0)]
        done = False
        for repeat in range(1, experiment.get("repeats", 1) + 1):
            for label, rate in labels:
                remaining = cap_s - (time.monotonic() - started)
                if remaining <= 0:
                    status, reason, done = "aborted", "time_cap", True
                    break
                name = f"repeat{repeat:02d}_{label}"
                plan_path = _write_plan(experiment_dir, name, plan)
                journal = experiment_dir / f"{name}.requests.jsonl"
                args = _namespace(url, plan_path, journal, requests=len(plan), concurrency=concurrency,
                                  rate=rate, max_tokens=max_tokens, timeout=timeout, **extras)
                counters_before = await scrape_counters(url, api_key)
                window_start = collector.now() if collector else None
                if collector:
                    collector.mark(f"{name}_start")
                try:
                    result = await asyncio.wait_for(load.run(args), remaining)
                except TimeoutError:
                    status, reason, done = "aborted", "time_cap", True
                    if collector:
                        collector.mark(f"{name}_aborted", reason="time_cap")
                    break
                if collector:
                    collector.mark(f"{name}_end")
                    result["telemetry"] = collector.window(window_start, collector.now())
                result["server_counters_delta"] = counter_delta(counters_before, await scrape_counters(url, api_key))
                result["experiment"] = {"id": experiment["id"], "kind": experiment["kind"], "repeat": repeat,
                                        "arrival": label, "rate_per_s": rate or None,
                                        "arrival_process": extras["arrival"] if rate else "closed_loop",
                                        "description": experiment.get("description", "")}
                result["config_sha256"] = config_sha256
                result["target"] = config["target"]
                result["limitations"] = limitations_for(result["metadata"].get("server"))
                result["statistics_caveat"] = statistics_caveat(result["summary"])
                report = experiment_dir / f"{name}.json"
                write_json(report, result)
                artifacts.append(report.name)
                reason = _abort_reason(result, abort_rate)
                if reason:
                    status, done = "aborted", True
                    break
            if done:
                break
    finally:
        if collector:
            overhead = await collector.stop()
    outcome = {"artifacts": artifacts, "status": status, "abort_reason": reason,
               "elapsed_s": round(time.monotonic() - started, 1), "cap_s": round(cap_s, 1)}
    if overhead is not None:
        outcome["telemetry_overhead"] = overhead
    return outcome


def _server_health(url):
    """One synchronous /health read so the manifest records the worker's mode and precision at the start."""
    import urllib.request
    key = os.environ.get("TENSORPILOT_API_KEY")
    request = urllib.request.Request(url.rstrip("/") + "/health", headers={"Authorization": "Bearer " + key} if key else {})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.loads(response.read())
    except Exception:
        return None


def run(config, *, url, output_dir, only=None, config_path=None, session_minutes=None):
    validate_config(config)
    revision = config["target"].get("revision")
    if not isinstance(revision, str) or not REVISION.fullmatch(revision):
        raise ValueError("target.revision must be a 40-character commit SHA before any GPU run")
    chosen = [experiment for experiment in config["experiments"] if not only or experiment["id"] in only]
    if only:
        missing = set(only) - {experiment["id"] for experiment in config["experiments"]}
        if missing:
            raise ValueError(f"unknown experiment ids: {sorted(missing)}")
    for experiment in chosen:
        if experiment["kind"] in PLANNED_KINDS:
            raise ValueError(f"{experiment['id']} is {experiment['kind']} and cannot run in this release")
    budget_min = session_minutes or config.get("max_session_minutes")
    if not budget_min:
        raise ValueError("a session budget is required: set max_session_minutes or pass --session-minutes")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    digest = config_fingerprint(config)
    server_at_start = _server_health(url)
    session_started = time.monotonic()
    manifest = {"schema": SCHEMA, "run_id": uuid.uuid4().hex,
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "config_path": str(config_path) if config_path else None, "config_sha256": digest,
                "target": config["target"], "session_budget_minutes": budget_min,
                "assumptions_status": config["assumptions"]["status"],
                "experiments": {}, "limitations": limitations_for(server_at_start),
                "server_at_start": server_at_start}
    manifest_path = output / "manifest.json"
    for experiment in chosen:
        remaining = budget_min * 60 - (time.monotonic() - session_started)
        if remaining <= 0:
            manifest["experiments"][experiment["id"]] = {"status": "skipped", "abort_reason": "session_budget"}
        else:
            outcome = asyncio.run(run_experiment(
                config, experiment, url=url.rstrip("/"), output_dir=output,
                config_sha256=digest, allowed_seconds=remaining))
            manifest["experiments"][experiment["id"]] = outcome
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan_parser = commands.add_parser("plan", help="estimate GPU time without touching a GPU")
    plan_parser.add_argument("config")
    plan_parser.add_argument("--budget-minutes", type=float,
                             help="override max_session_minutes from the config")
    run_parser = commands.add_parser("run", help="run runnable experiments against a live worker")
    run_parser.add_argument("config")
    run_parser.add_argument("--url", default=os.environ.get("TENSORPILOT_URL", "http://127.0.0.1:8080"))
    run_parser.add_argument("--output-dir", required=True)
    run_parser.add_argument("--only", action="append", help="experiment id; repeatable")
    run_parser.add_argument("--session-minutes", type=float,
                            help="hard session budget; overrides max_session_minutes in the config")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        if args.command == "plan":
            budget = args.budget_minutes or config.get("max_session_minutes") or DEFAULT_BUDGET_MINUTES
            summary = plan(config, budget)
            print("ESTIMATES ONLY: derived from placeholder throughput assumptions, not measurements.")
            for row in summary["rows"]:
                print(f"{row['id']:<36} {row['kind']:<20} {row['status']:<9} {row['estimated_gpu_minutes']:>8} min")
            print(f"experiments {summary['experiment_minutes_estimate']} min + session overhead "
                  f"{summary['session_overhead_minutes']} min = {summary['session_minutes_estimate']} min; "
                  f"budget {summary['budget_minutes']} min; within_budget={str(summary['within_budget']).lower()}")
            print(summary["note"])
            sys.exit(0 if summary["within_budget"] else 1)
        manifest = run(config, url=args.url, output_dir=args.output_dir, only=args.only,
                       config_path=args.config, session_minutes=args.session_minutes)
        print(json.dumps({"output_dir": args.output_dir, "experiments": list(manifest["experiments"])}, indent=2))
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"gpu-bench failed: {exc}\n")


if __name__ == "__main__":
    main()
