"""Own a local supervised-worker cluster and its TensorPilot control plane."""
import argparse
import asyncio
from pathlib import Path
import signal
import sys

import aiohttp


async def wait_ready(url, timeout=90):
    deadline = asyncio.get_running_loop().time() + timeout
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1)) as session:
        while asyncio.get_running_loop().time() < deadline:
            try:
                async with session.get(url + "/readyz") as response:
                    if response.status == 200 and (await response.json()).get("ready"):
                        return
            except (aiohttp.ClientError, TimeoutError, ValueError):
                pass
            await asyncio.sleep(.2)
    raise TimeoutError(f"readiness timeout: {url}")


async def stop(process):
    if process.returncode is not None:
        return
    try:
        process.send_signal(signal.SIGINT)
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(process.wait(), 20)
    except TimeoutError:
        process.kill()
        await process.wait()


async def run_cluster(args):
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    processes, logs = [], []
    try:
        workers = []
        for index in range(args.workers):
            worker_id = f"worker-{index + 1}"
            port = args.base_worker_port + index
            url = f"http://127.0.0.1:{port}"
            lifecycle = output / f"{worker_id}.lifecycle.jsonl"
            worker_log = output / f"{worker_id}.log"
            supervisor_log = (output / f"{worker_id}.supervisor.log").open("ab")
            logs.append(supervisor_log)
            command = [sys.executable, "-m", "tensorpilot.supervisor", "--url", url,
                       "--host", "127.0.0.1", "--port", str(port),
                       "--execution", args.execution, "--batch-size", str(args.batch_size),
                       "--queue-size", str(args.queue_size), "--memory-gib", str(args.memory_gib),
                       "--max-restarts", str(args.max_restarts),
                       "--lifecycle-log", str(lifecycle), "--worker-log", str(worker_log)]
            if args.offline:
                command.append("--offline")
            processes.append(await asyncio.create_subprocess_exec(
                *command, stdout=supervisor_log, stderr=supervisor_log))
            workers.append((worker_id, url))
        await asyncio.gather(*(wait_ready(url) for _, url in workers))
        control_log = (output / "control-plane.log").open("ab")
        logs.append(control_log)
        control_command = [sys.executable, "-m", "tensorpilot.control_plane",
                           "--port", str(args.control_port), "--policy", args.policy,
                           "--stale-after", str(args.stale_after),
                           "--probe-interval", str(args.probe_interval)]
        for worker_id, url in workers:
            control_command.extend(["--worker", f"{worker_id}={url}"])
        control = await asyncio.create_subprocess_exec(
            *control_command, stdout=control_log, stderr=control_log)
        processes.append(control)
        await wait_ready(f"http://127.0.0.1:{args.control_port}")
        print(f"TensorPilot cluster ready: http://127.0.0.1:{args.control_port}", flush=True)
        code = await control.wait()
        return code
    finally:
        await asyncio.gather(*(stop(process) for process in reversed(processes)))
        for stream in logs:
            stream.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--base-worker-port", type=int, default=8101)
    parser.add_argument("--control-port", type=int, default=8090)
    parser.add_argument("--execution", choices=("serial", "batch", "adaptive"), default="adaptive")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--queue-size", type=int, default=8)
    parser.add_argument("--memory-gib", type=float, default=6)
    parser.add_argument("--max-restarts", type=int, default=3)
    parser.add_argument("--policy", choices=("capacity-aware", "round-robin"), default="capacity-aware")
    parser.add_argument("--stale-after", type=float, default=3)
    parser.add_argument("--probe-interval", type=float, default=.5)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output-dir", default="results/v07_cluster_runtime")
    args = parser.parse_args()
    if (not 1 <= args.workers <= 8 or args.base_worker_port < 1024 or
            args.base_worker_port + args.workers > 65535 or not 1 <= args.control_port <= 65535 or
            args.control_port in range(args.base_worker_port, args.base_worker_port + args.workers)):
        parser.error("workers must be 1..8 with valid, non-overlapping ports")
    try:
        code = asyncio.run(run_cluster(args))
    except KeyboardInterrupt:
        code = 130
    except (TimeoutError, FileExistsError, OSError) as exc:
        parser.exit(1, f"Cluster failed: {exc}\n")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
