"""Restart a failed TensorPilot process and gate it on readiness before service."""
import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import sys

import aiohttp


def event(stream, kind, **fields):
    record = {"at_utc": datetime.now(timezone.utc).isoformat(), "event": kind, **fields}
    stream.write(json.dumps(record, separators=(",", ":")) + "\n")
    stream.flush()


async def wait_ready(process, url, timeout):
    deadline = asyncio.get_running_loop().time() + timeout
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1)) as session:
        while asyncio.get_running_loop().time() < deadline:
            if process.returncode is not None:
                raise RuntimeError("worker exited before readiness")
            try:
                async with session.get(url.rstrip("/") + "/readyz") as response:
                    if response.status == 200 and (await response.json()).get("ready"):
                        return
            except (aiohttp.ClientError, TimeoutError, ValueError):
                pass
            await asyncio.sleep(.2)
    raise TimeoutError("worker readiness timeout")


async def stop_process(process):
    if process.returncode is not None:
        return
    try:
        process.send_signal(signal.SIGINT)
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(process.wait(), 15)
    except TimeoutError:
        process.kill()
        await process.wait()


async def supervise(command, health_url, lifecycle_path, worker_log_path, *, max_restarts=3,
                    ready_timeout=90, backoff_s=1, start_factory=None, readiness=wait_ready):
    start_factory = start_factory or asyncio.create_subprocess_exec
    restarts = 0
    process = None
    with Path(lifecycle_path).open("a", encoding="utf-8") as lifecycle, \
            Path(worker_log_path).open("ab") as worker_log:
        try:
            while True:
                event(lifecycle, "worker_starting", restart_index=restarts)
                process = await start_factory(*command, stdout=worker_log, stderr=worker_log)
                try:
                    await readiness(process, health_url, ready_timeout)
                    event(lifecycle, "worker_ready", pid=getattr(process, "pid", None), restart_index=restarts)
                    code = await process.wait()
                    event(lifecycle, "worker_exited", exit_code=code, restart_index=restarts)
                except (RuntimeError, TimeoutError) as exc:
                    event(lifecycle, "worker_start_failed", error_type=type(exc).__name__, restart_index=restarts)
                    await stop_process(process)
                if restarts >= max_restarts:
                    event(lifecycle, "restart_budget_exhausted", max_restarts=max_restarts)
                    return 1
                restarts += 1
                event(lifecycle, "worker_restart_scheduled", restart_index=restarts, backoff_s=backoff_s)
                await asyncio.sleep(backoff_s)
        finally:
            if process is not None:
                await stop_process(process)


def build_command(args):
    """The worker command line for the supervised server, including any flags forwarded with --worker-arg."""
    command = [sys.executable, "-m", "tensorpilot.server", "--host", args.host,
               "--port", str(args.port), "--execution", args.execution,
               "--batch-size", str(args.batch_size), "--queue-size", str(args.queue_size),
               "--memory-gib", str(args.memory_gib),
               "--max-output-tokens", str(args.max_output_tokens)]
    if args.offline:
        command.append("--offline")
    return command + list(args.worker_arg)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--max-restarts", type=int, default=3)
    parser.add_argument("--ready-timeout", type=float, default=90)
    parser.add_argument("--backoff", type=float, default=1)
    parser.add_argument("--lifecycle-log", default="results/worker_lifecycle.jsonl")
    parser.add_argument("--worker-log", default="results/supervised_worker.log")
    parser.add_argument("--execution", choices=("serial", "batch", "adaptive"), default="serial")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--queue-size", type=int, default=8)
    parser.add_argument("--memory-gib", type=float, default=6)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--worker-arg", action="append", default=[],
                        help="a flag or value forwarded to the worker, repeatable; for example "
                             "--worker-arg=--quantization --worker-arg=nf4")
    args = parser.parse_args()
    if not 0 <= args.max_restarts <= 20 or args.ready_timeout <= 0 or not 0 <= args.backoff <= 60:
        parser.error("restarts must be 0..20, timeout positive, and backoff 0..60 seconds")
    if not 1 <= args.port <= 65535 or not 1 <= args.max_output_tokens <= 512:
        parser.error("port must be valid and max-output-tokens must be in 1..512")
    command = build_command(args)
    try:
        code = asyncio.run(supervise(command, args.url, args.lifecycle_log, args.worker_log,
                                     max_restarts=args.max_restarts, ready_timeout=args.ready_timeout,
                                     backoff_s=args.backoff))
    except KeyboardInterrupt:
        code = 130
    raise SystemExit(code)


if __name__ == "__main__":
    main()
