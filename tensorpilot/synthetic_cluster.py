"""Run a bounded GPU-free cluster through real loopback HTTP and injected faults."""
import argparse
import asyncio
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import resource
import statistics
import threading
import time
import tracemalloc
import uuid

import aiohttp
from aiohttp import web

from .control_plane import create_app
from .reporting import write_json
from .scale_bench import percentile


@dataclass
class SyntheticWorker:
    worker_id: str
    seed: int
    delay_s: float
    jitter_s: float
    capacity: int
    online: bool = True
    hang_health: bool = False
    generation: int = 1
    active: int = 0
    requests: int = 0
    rng: random.Random = field(init=False, repr=False)

    def __post_init__(self):
        self.rng = random.Random(self.seed)

    async def delay(self):
        await asyncio.sleep(max(0, self.delay_s + self.rng.uniform(-self.jitter_s, self.jitter_s)))

    def app(self):
        app = web.Application()

        async def health(request):
            if self.hang_health:
                # Longer than the control-plane timeout, but bounded so shutdown
                # cannot wait on a deliberately stuck synthetic handler.
                await asyncio.sleep(2)
            await self.delay()
            if not self.online:
                return web.json_response({"ready": False}, status=503)
            return web.json_response({
                "ready": True, "active_sequences": self.active, "queue_depth": 0,
                "execution_concurrency": self.capacity, "queue_capacity": self.capacity,
                "admission": {"estimated_kv_reserved_bytes": 0,
                              "estimated_kv_budget_bytes": 8 * 1024**3},
                "model": "synthetic-model", "artifact_sha256": "0" * 64,
                "service_version": "0.8.synthetic",
                "instance_id": f"{self.worker_id}-generation-{self.generation}",
                "execution_mode": "synthetic",
            })

        async def generate(request):
            if not self.online:
                return web.json_response({"error": {"code": "synthetic_offline"}}, status=503)
            body = await request.json()
            self.requests += 1
            self.active += 1
            request_id = uuid.uuid4().hex
            try:
                response = web.StreamResponse(headers={"Content-Type": "application/x-ndjson",
                                                       "X-Request-ID": request_id})
                await response.prepare(request)
                await self.delay()
                event = {"type": "token", "text": "x", "token_id": 1,
                         "output": True, "request_id": request_id}
                done = {"type": "done", "generated_tokens": 1, "queue_wait_s": 0,
                        "server_elapsed_s": self.delay_s, "ttft_s": self.delay_s,
                        "request_id": request_id, "requested": body.get("max_tokens", 1)}
                await response.write((json.dumps(event) + "\n" + json.dumps(done) + "\n").encode())
                await response.write_eof()
                return response
            finally:
                self.active -= 1

        app.router.add_get("/health", health)
        app.router.add_post("/v1/generate", generate)
        return app


async def start_app(app):
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, f"http://127.0.0.1:{port}"


class EventLoopThread:
    """Own an asyncio loop so load, control, and workers are independently scheduled."""

    def __init__(self, name):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run, name=name, daemon=True)
        self.thread.start()

    def _run(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    async def submit(self, coroutine):
        future = asyncio.run_coroutine_threadsafe(coroutine, self.loop)
        return await asyncio.wrap_future(future)

    def stop(self):
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=5)
        self.loop.close()


async def worker_record(session, control_url, worker_id):
    cursor = None
    while True:
        params = {"limit": 100}
        if cursor:
            params["cursor"] = cursor
        async with session.get(control_url + "/v1/workers", params=params) as response:
            response.raise_for_status()
            page = await response.json()
        for worker in page["workers"]:
            if worker["worker_id"] == worker_id:
                return worker
        cursor = page["next_cursor"]
        if cursor is None:
            return None


async def wait_worker(session, control_url, worker_id, predicate, timeout):
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        worker = await worker_record(session, control_url, worker_id)
        if worker is not None and predicate(worker):
            return time.perf_counter(), worker
        await asyncio.sleep(.01)
    raise TimeoutError(f"worker transition timeout: {worker_id}")


async def run_scenario(*, workers=16, requests=2000, concurrency=64, delay_ms=20,
                       jitter_ms=5, fault_workers=2, fault_after_s=.1,
                       fault_duration_s=.3, heartbeat_timeout_s=.5, seed=42):
    if not 2 <= workers <= 64:
        raise ValueError("workers must be in 2..64")
    if not 1 <= requests <= 100_000 or not 1 <= concurrency <= min(requests, 2000):
        raise ValueError("requests must be 1..100000 and concurrency 1..min(requests, 2000)")
    if (not 1 <= fault_workers < workers or delay_ms < 0 or not 0 <= jitter_ms <= delay_ms or
            not .05 <= heartbeat_timeout_s <= 5):
        raise ValueError("fault_workers and delay/jitter configuration are invalid")

    synthetic = [SyntheticWorker(f"worker-{index:04d}", seed + index,
                                 delay_ms / 1000, jitter_ms / 1000,
                                 max(2, math.ceil(concurrency / workers)))
                 for index in range(workers)]
    runners = []
    worker_hosts = []
    urls = []
    control_runner = None
    control_host = EventLoopThread("tensorpilot-synthetic-control")
    tracemalloc.start()
    started_at = time.perf_counter()
    try:
        for worker in synthetic:
            host = EventLoopThread(f"tensorpilot-{worker.worker_id}")
            worker_hosts.append(host)
            runner, url = await host.submit(start_app(worker.app()))
            runners.append(runner)
            urls.append(url)
        async def start_control():
            control = create_app(workers=tuple((worker.worker_id, url)
                                               for worker, url in zip(synthetic, urls)),
                                 policy="power-of-two", stale_after_s=heartbeat_timeout_s * 3,
                                 probe_interval_s=.05, heartbeat_concurrency=min(16, workers),
                                 heartbeat_connect_timeout_s=min(.1, heartbeat_timeout_s),
                                 heartbeat_total_timeout_s=heartbeat_timeout_s)
            return await start_app(control)

        control_runner, control_url = await control_host.submit(start_control())
        timeout = aiohttp.ClientTimeout(total=5)
        connector = aiohttp.TCPConnector(limit=max(100, concurrency * 2))
        records = []
        next_request = 0
        event_loop_lag = []
        stop_monitor = asyncio.Event()

        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            async def monitor():
                expected = time.perf_counter() + .01
                while not stop_monitor.is_set():
                    await asyncio.sleep(max(0, expected - time.perf_counter()))
                    now = time.perf_counter()
                    event_loop_lag.append(max(0, now - expected))
                    expected += .01

            async def client_worker():
                nonlocal next_request
                while next_request < requests:
                    index = next_request
                    next_request += 1
                    began = time.perf_counter()
                    record = {"index": index, "started_offset_s": began - started_at}
                    try:
                        async with session.post(control_url + "/v1/generate",
                                                json={"prompt": f"synthetic-{index}",
                                                      "max_tokens": 1, "timeout_s": 2}) as response:
                            record["http_status"] = response.status
                            record["routing_latency_s"] = time.perf_counter() - began
                            record["worker_id"] = response.headers.get("X-TensorPilot-Worker")
                            events = []
                            async for line in response.content:
                                events.append(json.loads(line))
                            terminal = events[-1] if events else None
                            record["terminal_type"] = terminal.get("type") if terminal else None
                            record["terminal_code"] = (terminal.get("code") or
                                                       (terminal.get("error") or {}).get("code")) if terminal else None
                            record["ok"] = response.status == 200 and record["terminal_type"] == "done"
                    except (aiohttp.ClientError, TimeoutError, json.JSONDecodeError) as exc:
                        record.update(ok=False, error_type=type(exc).__name__)
                    record["latency_s"] = time.perf_counter() - began
                    records.append(record)

            async def faults():
                await asyncio.sleep(fault_after_s)
                targets = synthetic[:fault_workers]
                fault_started = time.perf_counter()
                for worker in targets:
                    worker.online = False
                    worker.hang_health = True
                detected_at, _ = await wait_worker(
                    session, control_url, targets[0].worker_id,
                    lambda item: not item["ready"], 10)
                await asyncio.sleep(max(0, fault_duration_s - (time.perf_counter() - fault_started)))
                recovery_started = time.perf_counter()
                for worker in targets:
                    worker.generation += 1
                    worker.hang_health = False
                    worker.online = True
                recovered_at, recovered = await wait_worker(
                    session, control_url, targets[0].worker_id,
                    lambda item: item["ready"] and item["instance_id"].endswith("generation-2"), 10)

                churn_worker = SyntheticWorker("worker-churn", seed + 9999,
                                               delay_ms / 1000, jitter_ms / 1000, 2)
                churn_host = EventLoopThread("tensorpilot-worker-churn")
                worker_hosts.append(churn_host)
                churn_runner, churn_url = await churn_host.submit(start_app(churn_worker.app()))
                runners.append(churn_runner)
                register_started = time.perf_counter()
                async with session.post(control_url + "/v1/workers",
                                        json={"worker_id": churn_worker.worker_id,
                                              "url": churn_url}) as response:
                    response.raise_for_status()
                registration_s = time.perf_counter() - register_started
                remove_started = time.perf_counter()
                async with session.post(control_url + "/v1/workers/worker-churn/drain",
                                        json={"draining": True}) as response:
                    response.raise_for_status()
                await wait_worker(session, control_url, "worker-churn",
                                  lambda item: item["in_flight_routes"] == 0, 10)
                async with session.delete(control_url + "/v1/workers/worker-churn") as response:
                    response.raise_for_status()
                removal_s = time.perf_counter() - remove_started
                return {"fault_started_offset_s": fault_started - started_at,
                        "detection_s": detected_at - fault_started,
                        "recovery_s": recovered_at - recovery_started,
                        "recovered_instance_id": recovered["instance_id"],
                        "registration_s": registration_s, "removal_s": removal_s}

            monitor_task = asyncio.create_task(monitor())
            fault_task = asyncio.create_task(faults())
            load_started = time.perf_counter()
            await asyncio.gather(*(client_worker() for _ in range(concurrency)))
            load_wall_s = time.perf_counter() - load_started
            fault_result = await fault_task
            stop_monitor.set()
            await monitor_task
            async with session.get(control_url + "/health") as response:
                final_health = await response.json()
            async with session.get(control_url + "/metrics") as response:
                metrics_text = await response.text()

        current_bytes, peak_bytes = tracemalloc.get_traced_memory()
        successes = [record for record in records if record["ok"]]
        latencies = [record["latency_s"] for record in successes]
        routing_latencies = [record["routing_latency_s"] for record in records
                             if "routing_latency_s" in record]
        placements = Counter(record.get("worker_id") for record in records if record.get("worker_id"))
        assignment_values = list(placements.values())
        stable_values = [count for worker_id, count in placements.items()
                         if worker_id != "worker-churn" and
                         int(worker_id.rsplit("-", 1)[-1]) >= fault_workers]
        return {
            "schema_version": 1, "benchmark": "synthetic-network-fault-cluster",
            "run_at_utc": datetime.now(timezone.utc).isoformat(),
            "configuration": {"workers": workers, "requests": requests,
                              "concurrency": concurrency, "delay_ms": delay_ms,
                              "jitter_ms": jitter_ms, "fault_workers": fault_workers,
                              "fault_after_s": fault_after_s,
                              "fault_duration_s": fault_duration_s, "seed": seed,
                              "heartbeat_timeout_s": heartbeat_timeout_s,
                              "gpu_allocated": False, "transport": "real-loopback-tcp"},
            "summary": {"successful_requests": len(successes),
                        "failed_requests": len(records) - len(successes),
                        "error_rate": (len(records) - len(successes)) / len(records),
                        "requests_per_s": len(records) / load_wall_s,
                        "successful_requests_per_s": len(successes) / load_wall_s,
                        "p50_latency_s": percentile(latencies, .50),
                        "p95_latency_s": percentile(latencies, .95),
                        "p99_latency_s": percentile(latencies, .99),
                        "p50_routing_latency_s": percentile(routing_latencies, .50),
                        "p95_routing_latency_s": percentile(routing_latencies, .95),
                        "p99_routing_latency_s": percentile(routing_latencies, .99),
                        "event_loop_lag_p95_s": percentile(event_loop_lag, .95),
                        "event_loop_lag_max_s": max(event_loop_lag, default=0),
                        "assignment_min": min(assignment_values, default=0),
                        "assignment_max": max(assignment_values, default=0),
                        "assignment_stddev": statistics.pstdev(assignment_values)
                        if assignment_values else 0,
                        "stable_assignment_min": min(stable_values, default=0),
                        "stable_assignment_max": max(stable_values, default=0),
                        "stable_assignment_stddev": statistics.pstdev(stable_values)
                        if stable_values else 0,
                        "traced_current_bytes": current_bytes,
                        "traced_peak_bytes": peak_bytes,
                        "process_max_rss_raw": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},
            "faults": fault_result, "final_health": final_health,
            "metrics_excerpt": [line for line in metrics_text.splitlines()
                                if line.startswith("tensorpilot_cluster_")],
            "worker_assignments": dict(sorted(placements.items())),
            "requests": sorted(records, key=lambda item: item["index"]),
            "limitations": ("All endpoints and the control plane share one Python event loop and one Mac. "
                            "This validates real HTTP routing and failure semantics, not independent nodes or GPUs."),
        }
    finally:
        tracemalloc.stop()
        if control_runner is not None:
            await control_host.submit(control_runner.cleanup())
        for runner, host in zip(runners, worker_hosts):
            try:
                await host.submit(runner.cleanup())
            except Exception:
                pass
        control_host.stop()
        for host in worker_hosts:
            host.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--requests", type=int, default=2000)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--delay-ms", type=float, default=20)
    parser.add_argument("--jitter-ms", type=float, default=5)
    parser.add_argument("--fault-workers", type=int, default=2)
    parser.add_argument("--fault-after", type=float, default=.1)
    parser.add_argument("--fault-duration", type=float, default=.3)
    parser.add_argument("--heartbeat-timeout", type=float, default=.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/v083_synthetic_cluster.json")
    args = parser.parse_args()
    path = Path(args.output)
    if path.exists():
        parser.error(f"output already exists: {path}")
    try:
        result = asyncio.run(run_scenario(workers=args.workers, requests=args.requests,
                                         concurrency=args.concurrency,
                                         delay_ms=args.delay_ms, jitter_ms=args.jitter_ms,
                                         fault_workers=args.fault_workers,
                                         fault_after_s=args.fault_after,
                                         fault_duration_s=args.fault_duration,
                                         heartbeat_timeout_s=args.heartbeat_timeout,
                                         seed=args.seed))
    except (ValueError, TimeoutError) as exc:
        parser.error(str(exc))
    write_json(path, result)
    print(json.dumps({"summary": result["summary"], "faults": result["faults"]}, indent=2))
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
