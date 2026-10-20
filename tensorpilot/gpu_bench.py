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
import uuid
from datetime import datetime, timezone
from pathlib import Path

from . import load
from .reporting import write_json
from .workloads import build_stress_workload, validate_workload

SCHEMA = "tensorpilot.gpu-experiments/v1"
REVISION = re.compile(r"[0-9a-f]{40}")
RUNNABLE_KINDS = ("closed_loop", "open_loop", "saturation", "length_matrix")
PLANNED_KINDS = ("recovery_manual", "batch_width_planned", "two_gpu_planned")
ALL_KINDS = RUNNABLE_KINDS + PLANNED_KINDS
EXPERIMENT_KEYS = {"id", "kind", "status", "description", "requests", "concurrency", "prompt_repeats",
                   "max_tokens", "repeats", "warmup_requests", "timeout_s", "rates", "steps", "reason",
                   "estimated_minutes", "minimum_gpus"}
LIMITATIONS = (
    "The worker executes one sequence at a time; concurrency above one measures queueing, not batched throughput.",
    "Client timings include loopback HTTP and streaming parse time on the load-generating host.",
    "One model on one GPU; no multi-GPU or scaling claim.",
    "CUDA peak memory is process-lifetime from the PyTorch caching allocator, not total device pressure.",
    "Planning estimates are assumptions and are never reported as measurements.",
    "GPU clocks, temperature and co-tenancy are recorded by nvidia-smi samples, not controlled.",
)
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
    checks = {"requests": (1, 2000), "concurrency": (1, 64), "prompt_repeats": (1, 256),
              "max_tokens": (1, 512), "repeats": (1, 10), "warmup_requests": (0, 50),
              "timeout_s": (1, 120)}
    for field, (low, high) in checks.items():
        if field in experiment and not _bounded_int(experiment[field], low, high):
            raise ValueError(f"{identifier}: {field} must be an integer in {low}..{high}")
    for required in ("requests", "prompt_repeats", "max_tokens"):
        if required not in experiment:
            raise ValueError(f"{identifier}: {required} is required")
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
    return {"minutes": round(seconds / 60 + 1, 2), "note": "estimate: prefill and decode assumptions plus 1 minute"}


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
            "session_overhead_minutes": overhead, "session_minutes_estimate": round(session_minutes, 2),
            "budget_minutes": budget_minutes, "within_budget": session_minutes <= budget_minutes,
            "note": "Estimates depend on the declared assumptions and are not measurements."}


def config_fingerprint(config):
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_plan_for(experiment):
    return validate_workload(build_stress_workload(
        experiment["requests"], prompt_repeats=experiment["prompt_repeats"],
        output_tokens=experiment["max_tokens"]))


def _namespace(url, plan_path, journal, *, requests, concurrency, rate, max_tokens, timeout):
    return argparse.Namespace(
        url=url, requests=requests, concurrency=concurrency, rate=rate, max_tokens=max_tokens,
        timeout=timeout, workload="uniform", workload_file=str(plan_path), seed=42, duration_s=None,
        max_requests=requests, journal=str(journal), journal_fsync=False, compact=False,
        overwrite=True)


def _write_plan(directory, name, plan):
    path = directory / f"{name}.workload.json"
    path.write_text(json.dumps(plan, indent=2) + "\n")
    return path


async def run_experiment(config, experiment, *, url, output_dir, config_sha256):
    experiment_dir = Path(output_dir) / experiment["id"]
    experiment_dir.mkdir(parents=True, exist_ok=False)
    timeout = experiment.get("timeout_s", 60)
    concurrency = experiment.get("concurrency", 1)
    max_tokens = experiment["max_tokens"]
    artifacts = []
    plan = build_plan_for(experiment)
    warmup = experiment.get("warmup_requests", 0)
    if warmup:
        warm_plan = build_plan_for({**experiment, "requests": warmup})
        warm_path = _write_plan(experiment_dir, "warmup", warm_plan)
        journal = experiment_dir / "warmup.requests.jsonl"
        await load.run(_namespace(url, warm_path, journal, requests=warmup, concurrency=1,
                                  rate=0, max_tokens=max_tokens, timeout=timeout))
    labels = []
    if experiment["kind"] in ("open_loop", "saturation"):
        labels = [(f"rate{str(rate).replace('.', 'p')}", rate) for rate in experiment["rates"]]
    else:
        labels = [("closed", 0)]
    for repeat in range(1, experiment.get("repeats", 1) + 1):
        for label, rate in labels:
            name = f"repeat{repeat:02d}_{label}"
            plan_path = _write_plan(experiment_dir, name, plan)
            journal = experiment_dir / f"{name}.requests.jsonl"
            args = _namespace(url, plan_path, journal, requests=len(plan), concurrency=concurrency,
                              rate=rate, max_tokens=max_tokens, timeout=timeout)
            result = await load.run(args)
            result["experiment"] = {"id": experiment["id"], "kind": experiment["kind"], "repeat": repeat,
                                    "arrival": label, "rate_per_s": rate or None,
                                    "description": experiment.get("description", "")}
            result["config_sha256"] = config_sha256
            result["target"] = config["target"]
            result["limitations"] = list(LIMITATIONS)
            report = experiment_dir / f"{name}.json"
            write_json(report, result)
            artifacts.append(report.name)
    return artifacts


def run(config, *, url, output_dir, only=None, config_path=None):
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
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    digest = config_fingerprint(config)
    manifest = {"schema": SCHEMA, "run_id": uuid.uuid4().hex,
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "config_path": str(config_path) if config_path else None, "config_sha256": digest,
                "target": config["target"], "experiments": {}, "limitations": list(LIMITATIONS)}
    manifest_path = output / "manifest.json"
    for experiment in chosen:
        manifest["experiments"][experiment["id"]] = asyncio.run(run_experiment(
            config, experiment, url=url.rstrip("/"), output_dir=output, config_sha256=digest))
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan_parser = commands.add_parser("plan", help="estimate GPU time without touching a GPU")
    plan_parser.add_argument("config")
    plan_parser.add_argument("--budget-minutes", type=float, default=DEFAULT_BUDGET_MINUTES)
    run_parser = commands.add_parser("run", help="run runnable experiments against a live worker")
    run_parser.add_argument("config")
    run_parser.add_argument("--url", default=os.environ.get("TENSORPILOT_URL", "http://127.0.0.1:8080"))
    run_parser.add_argument("--output-dir", required=True)
    run_parser.add_argument("--only", action="append", help="experiment id; repeatable")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        if args.command == "plan":
            summary = plan(config, args.budget_minutes)
            for row in summary["rows"]:
                print(f"{row['id']:<36} {row['kind']:<20} {row['status']:<9} {row['estimated_gpu_minutes']:>8} min")
            print(f"experiments {summary['experiment_minutes_estimate']} min + session overhead "
                  f"{summary['session_overhead_minutes']} min = {summary['session_minutes_estimate']} min; "
                  f"budget {summary['budget_minutes']} min; within_budget={str(summary['within_budget']).lower()}")
            print(summary["note"])
            sys.exit(0 if summary["within_budget"] else 1)
        manifest = run(config, url=args.url, output_dir=args.output_dir, only=args.only, config_path=args.config)
        print(json.dumps({"output_dir": args.output_dir, "experiments": list(manifest["experiments"])}, indent=2))
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"gpu-bench failed: {exc}\n")


if __name__ == "__main__":
    main()
