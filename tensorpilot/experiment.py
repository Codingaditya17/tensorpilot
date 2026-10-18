"""Run matched scheduler experiments with one owned model process at a time."""
import argparse
import asyncio
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import os
import signal
import sys
from types import SimpleNamespace
import uuid
import zipfile
from importlib.metadata import distributions

import aiohttp

from .bench import DEFAULT_MODEL
from .compare import compare_reports
from .load import run as run_load
from .reporting import write_json
from .workloads import build_workload, fingerprint


def run_order(repetitions):
    # Counterbalance start order to reduce systematic thermal/time ordering bias.
    return [(repeat, policy) for repeat in range(1, repetitions + 1)
            for policy in (("fifo", "shortest-budget") if repeat % 2 else ("shortest-budget", "fifo"))]


async def ensure_port_free(port):
    try:
        _, writer = await asyncio.open_connection("127.0.0.1", port)
    except OSError:
        return
    writer.close()
    await writer.wait_closed()
    raise RuntimeError(f"port {port} is already in use; stop that service or choose another port")


async def stop_process(process):
    if process.returncode is None:
        try:
            process.send_signal(signal.SIGINT)
        except ProcessLookupError:
            pass
        try:
            await asyncio.wait_for(process.wait(), 15)
        except TimeoutError:
            # Only the child created by this experiment may be killed.
            process.kill()
            await process.wait()


async def wait_ready(process, url, instance_id, timeout=90):
    api_key = os.environ.get("TENSORPILOT_API_KEY")
    headers = {"Authorization": "Bearer " + api_key} if api_key else {}
    deadline = asyncio.get_running_loop().time() + timeout
    async with aiohttp.ClientSession(headers=headers, timeout=aiohttp.ClientTimeout(total=1)) as session:
        while asyncio.get_running_loop().time() < deadline:
            if process.returncode is not None:
                raise RuntimeError(f"model server exited during startup: {process.returncode}; inspect its log")
            try:
                async with session.get(url + "/health") as response:
                    health = await response.json()
                    if response.status == 200 and health.get("ready") and health.get("instance_id") == instance_id:
                        return health
            except (aiohttp.ClientError, TimeoutError, ValueError):
                pass
            await asyncio.sleep(.2)
    raise TimeoutError("model startup readiness timed out; inspect its log")


async def run_experiment(args):
    if not 1 <= args.concurrency <= args.queue_size:
        raise ValueError("comparison concurrency must be in 1..queue-size to avoid intentional overload")
    await ensure_port_free(args.port)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    plan = build_workload(args.requests, kind="mixed", seed=args.seed)
    plan_path = output / "workload.json"
    write_json(plan_path, plan)
    warmup_plan = []
    for item in plan:
        if item["class"] not in {request["class"] for request in warmup_plan}:
            warmup_plan.append(item)
    warmup_path = output / "warmup_workload.json"
    write_json(warmup_path, warmup_plan)
    # Source and installed versions travel with each experiment, independently
    # of Git availability and later edits in the developer's checkout.
    root = Path(__file__).resolve().parent.parent
    with zipfile.ZipFile(output / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        source_files = list((root / "tensorpilot").glob("*.py")) + list((root / "tests").glob("*.py"))
        source_files += [root / name for name in ("README.md", "pyproject.toml", "requirements.txt", "requirements-service.txt")]
        for path in sorted(source_files):
            if path.is_file():
                archive.write(path, path.relative_to(root))
    write_json(output / "installed_versions.json", {dist.metadata["Name"]: dist.version for dist in distributions()
                                                   if dist.metadata.get("Name")})
    manifest = {"schema_version": 1, "status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "config": vars(args), "workload_sha256": fingerprint(plan), "runs": [],
                "measurement_kind": "real_mlx_gpu_with_http_queueing"}
    manifest_path = output / "manifest.json"
    write_json(manifest_path, manifest)
    revision = args.revision
    reports = []
    try:
        for repeat, policy in run_order(args.repeat):
            await ensure_port_free(args.port)
            instance_id = uuid.uuid4().hex
            name = f"repeat{repeat:02d}_{policy}"
            command = [sys.executable, "-m", "tensorpilot.server", "--port", str(args.port),
                       "--model", args.model, "--queue-size", str(args.queue_size),
                       "--memory-gib", str(args.memory_gib), "--scheduler", policy,
                       "--max-queue-wait", str(args.max_queue_wait), "--instance-id", instance_id]
            if revision:
                command.extend(["--revision", revision])
            if args.offline:
                command.append("--offline")
            print(f"Starting {name}", flush=True)
            with (output / f"{name}.server.log").open("wb") as log:
                process = await asyncio.create_subprocess_exec(*command, stdout=log, stderr=log)
                try:
                    health = await wait_ready(process, f"http://127.0.0.1:{args.port}", instance_id)
                    if health.get("model_source") == "huggingface":
                        revision = health.get("model_revision")
                        if not revision:
                            raise ValueError("experiment requires an immutable model revision")
                    warmup_args = SimpleNamespace(url=f"http://127.0.0.1:{args.port}", requests=len(warmup_plan),
                                                  concurrency=1, rate=0, max_tokens=128, timeout=args.timeout,
                                                  workload_file=str(warmup_path), seed=args.seed, workload="mixed")
                    warmup_report = await run_load(warmup_args)
                    write_json(output / f"{name}.warmup.json", warmup_report)
                    if warmup_report["summary"]["failed_requests"]:
                        raise RuntimeError("workload warmup failed; inspect its report")
                    load_args = SimpleNamespace(url=f"http://127.0.0.1:{args.port}", requests=args.requests,
                                                concurrency=args.concurrency, rate=args.rate, max_tokens=128,
                                                timeout=args.timeout, output=str(output / f"{name}.json"),
                                                workload="mixed", workload_file=str(plan_path), seed=args.seed,
                                                journal=str(output / f"{name}.requests.jsonl"), journal_fsync=False,
                                                compact=False)
                    report = await run_load(load_args)
                    report["metadata"]["experiment_repeat"] = repeat
                    report["metadata"]["experiment_order_index"] = len(reports)
                    write_json(load_args.output, report)
                    reports.append(report)
                    manifest["runs"].append({"repeat": repeat, "policy": policy,
                                             "report": f"{name}.json", "instance_id": instance_id,
                                             "artifact_sha256": health["artifact_sha256"],
                                             "model_revision": health.get("model_revision")})
                    write_json(manifest_path, manifest)
                    summary = report["summary"]
                    print(f"{name}: {summary['successful_requests']} completed, {summary['failed_requests']} failed, "
                          f"{summary['completed_output_tokens_per_s']:.1f} output tokens/s", flush=True)
                finally:
                    await stop_process(process)
        comparison = compare_reports(reports)
        comparison["source_reports"] = [run["report"] for run in manifest["runs"]]
        write_json(output / "comparison.json", comparison)
        manifest["status"] = "completed"
        manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(manifest_path, manifest)
        print(json.dumps(comparison["relative_change"], indent=2))
        print(f"Experiment saved: {output}")
        return comparison
    except BaseException as exc:
        manifest["status"] = "interrupted" if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt)) else "failed"
        manifest["error_type"] = type(exc).__name__
        write_json(manifest_path, manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--requests", type=int, default=30)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rate", type=float, default=0)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--queue-size", type=int, default=8)
    parser.add_argument("--memory-gib", type=float, default=6)
    parser.add_argument("--max-queue-wait", type=float, default=1)
    parser.add_argument("--port", type=int, default=8080)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parser.add_argument("--output-dir", default=f"results/experiment_{stamp}")
    args = parser.parse_args()
    if min(args.requests, args.repeat, args.queue_size) < 1 or not 1 <= args.concurrency <= args.queue_size:
        parser.error("requests/repeat/queue-size must be positive; concurrency must be in 1..queue-size")
    if args.requests > 10000 or args.concurrency > 256:
        parser.error("comparison limits: requests <= 10000, concurrency <= 256")
    if not math.isfinite(args.rate) or args.rate < 0 or not 0 < args.timeout <= 120:
        parser.error("rate must be finite and nonnegative; timeout must be in (0, 120]")
    if not 0 < args.memory_gib <= 8 or not math.isfinite(args.max_queue_wait) or args.max_queue_wait <= 0:
        parser.error("memory-gib must be in (0, 8]; max-queue-wait must be finite and positive")
    try:
        asyncio.run(run_experiment(args))
    except (ValueError, RuntimeError, FileExistsError) as exc:
        parser.exit(1, f"Experiment failed: {exc}\n")


if __name__ == "__main__":
    main()
