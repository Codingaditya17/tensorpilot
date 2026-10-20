"""Single-process TensorPilot control plane for local multi-worker experiments."""
import argparse
import asyncio
import hmac
import json
import math
import os
import time
import uuid

import aiohttp
from aiohttp import web
from prometheus_client import (CollectorRegistry, Counter, Gauge, Histogram,
                               generate_latest, CONTENT_TYPE_LATEST)

from . import __version__
from .admission import ResourceAdmission
from .routing import WorkerRegistry

REGISTRY = web.AppKey("registry", WorkerRegistry)
SESSION = web.AppKey("session", aiohttp.ClientSession)
PROBE_SESSION = web.AppKey("probe_session", aiohttp.ClientSession)
LOCK = web.AppKey("lock", asyncio.Lock)
CLUSTER_METRICS = web.AppKey("cluster_metrics", object)
REQUEST_ID = web.RequestKey("request_id", str) if hasattr(web, "RequestKey") else "request_id"


class ClusterMetrics:
    def __init__(self):
        self.registry = CollectorRegistry()
        options = {"registry": self.registry}
        self.placements = Counter("tensorpilot_cluster_placements", "Placement decisions", ["reason"], **options)
        self.placement_duration = Histogram("tensorpilot_cluster_placement_duration_seconds",
                                            "Lock wait and worker selection duration", **options)
        self.rejected = Counter("tensorpilot_cluster_rejections", "Requests rejected before worker dispatch", ["reason"], **options)
        self.outcomes = Counter("tensorpilot_cluster_proxy_outcomes", "Finished proxy streams", ["outcome"], **options)
        self.registered = Gauge("tensorpilot_cluster_registered_workers", "Registered workers", **options)
        self.ready = Gauge("tensorpilot_cluster_ready_workers", "Ready nonstale nondraining workers", **options)
        self.in_flight = Gauge("tensorpilot_cluster_in_flight_routes", "Currently routed requests", **options)
        self.heartbeat_queue = Gauge("tensorpilot_cluster_heartbeat_queue", "Queued worker probes", **options)
        self.heartbeat_duration = Histogram("tensorpilot_cluster_heartbeat_duration_seconds",
                                            "Worker heartbeat request duration", **options)
        self.heartbeat_outcomes = Counter("tensorpilot_cluster_heartbeat_total",
                                          "Worker heartbeat outcomes", ["outcome"], **options)
        self.worker_transitions = Counter("tensorpilot_cluster_worker_state_transitions_total",
                                          "Observed ready-state transitions", ["previous", "current"], **options)
        self.health_duration = Histogram("tensorpilot_cluster_health_endpoint_seconds",
                                         "Control-plane health handler duration", **options)
        self.heartbeat_sweep = Histogram("tensorpilot_cluster_heartbeat_sweep_seconds",
                                         "Time to probe one complete worker snapshot", **options)

    def render(self, registry):
        summary = registry.summary()
        self.registered.set(summary["worker_counts"]["registered"])
        self.ready.set(summary["worker_counts"]["available"])
        self.in_flight.set(summary["worker_counts"]["in_flight_routes"])
        return generate_latest(self.registry)


def create_app(*, workers=(), policy="capacity-aware", stale_after_s=5,
               probe_interval_s=1, allowed_hosts=("127.0.0.1", "localhost", "::1"),
               api_key=None, worker_api_key=None, heartbeat_concurrency=16,
               heartbeat_connect_timeout_s=.5, heartbeat_total_timeout_s=2,
               session_factory=aiohttp.ClientSession):
    if not 1 <= heartbeat_concurrency <= 1024:
        raise ValueError("heartbeat_concurrency must be in 1..1024")
    registry = WorkerRegistry(stale_after_s=stale_after_s, policy=policy)
    metrics = ClusterMetrics()
    admission = ResourceAdmission()

    @web.middleware
    async def operations(request, handler):
        started = time.perf_counter()
        request[REQUEST_ID] = uuid.uuid4().hex
        if api_key and request.path not in ("/livez", "/readyz"):
            supplied = request.headers.get("Authorization", "")
            if not hmac.compare_digest(supplied.encode(), ("Bearer " + api_key).encode()):
                return _error(request, 401, "unauthorized", "A valid control-plane token is required")
        try:
            response = await handler(request)
        except web.HTTPException as exc:
            code = {404: "not_found", 405: "method_not_allowed"}.get(exc.status, "http_error")
            response = _error(request, exc.status, code, exc.reason)
        if not response.prepared:
            response.headers.setdefault("X-Request-ID", request[REQUEST_ID])
        if request.path == "/health":
            metrics.health_duration.observe(time.perf_counter() - started)
        return response

    app = web.Application(client_max_size=32 * 1024, middlewares=[operations])
    app[REGISTRY], app[LOCK], app[CLUSTER_METRICS] = registry, asyncio.Lock(), metrics

    async def probe(worker_id):
        worker = registry.workers.get(worker_id)
        if worker is None:
            return False
        started = time.perf_counter()
        previous_ready = worker.ready
        try:
            timeout = aiohttp.ClientTimeout(total=heartbeat_total_timeout_s,
                                            connect=heartbeat_connect_timeout_s)
            async with app[PROBE_SESSION].get(worker.url + "/health", timeout=timeout) as response:
                if response.status != 200:
                    raise RuntimeError(f"health_http_{response.status}")
                health = await response.json()
            async with app[LOCK]:
                if worker_id in registry.workers:
                    registry.update(worker_id, health)
                    current_ready = registry.workers[worker_id].ready
                    if previous_ready != current_ready:
                        metrics.worker_transitions.labels(previous=str(previous_ready).lower(),
                                                          current=str(current_ready).lower()).inc()
            metrics.heartbeat_outcomes.labels(outcome="success").inc()
            return True
        except (aiohttp.ClientError, TimeoutError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
            async with app[LOCK]:
                if worker_id in registry.workers:
                    registry.failure(worker_id, type(exc).__name__)
                    if previous_ready:
                        metrics.worker_transitions.labels(previous="true", current="false").inc()
            metrics.heartbeat_outcomes.labels(outcome="failure").inc()
            return False
        finally:
            metrics.heartbeat_duration.observe(time.perf_counter() - started)

    queue = asyncio.Queue()

    async def heartbeat_worker():
        while True:
            worker_id = await queue.get()
            try:
                await probe(worker_id)
            finally:
                queue.task_done()
                metrics.heartbeat_queue.set(queue.qsize())

    async def sweep():
        started = time.perf_counter()
        async with app[LOCK]:
            worker_ids = list(registry.workers)
        for worker_id in worker_ids:
            await queue.put(worker_id)
            metrics.heartbeat_queue.set(queue.qsize())
        await queue.join()
        metrics.heartbeat_sweep.observe(time.perf_counter() - started)

    async def poll():
        while True:
            await sweep()
            await asyncio.sleep(probe_interval_s)

    async def lifecycle(app):
        headers = {"Authorization": "Bearer " + worker_api_key} if worker_api_key else {}
        app[SESSION] = session_factory(headers=headers)
        app[PROBE_SESSION] = session_factory(headers=headers)
        for worker_id, url in workers:
            clean = registry.validate_identity(worker_id, url, set(allowed_hosts))
            registry.register(worker_id, clean)
        consumers = [asyncio.create_task(heartbeat_worker(), name=f"tensorpilot-heartbeat-{index}")
                     for index in range(heartbeat_concurrency)]
        await sweep()
        task = asyncio.create_task(poll(), name="tensorpilot-cluster-heartbeats")
        try:
            yield
        finally:
            task.cancel()
            for consumer in consumers:
                consumer.cancel()
            await asyncio.gather(task, *consumers, return_exceptions=True)
            await app[PROBE_SESSION].close()
            await app[SESSION].close()

    async def livez(request):
        return web.json_response({"alive": True, "service": "tensorpilot-control-plane"})

    async def readyz(request):
        summary = registry.summary()
        ready = summary["worker_counts"]["available"] > 0
        return web.json_response({"ready": ready}, status=200 if ready else 503)

    async def health(request):
        summary = registry.summary()
        return web.json_response({"ready": summary["worker_counts"]["available"] > 0,
                                  "service_version": __version__,
                                  "service": "tensorpilot-control-plane", **summary})

    async def list_workers(request):
        try:
            raw_limit = request.query.get("limit", "100")
            if not raw_limit.isdigit():
                raise ValueError("limit must be an integer in 1..500")
            return web.json_response(registry.page(limit=int(raw_limit),
                                                   cursor=request.query.get("cursor")))
        except ValueError as exc:
            return _error(request, 400, "invalid_page", str(exc))
        except RuntimeError as exc:
            return _error(request, 409, "stale_cursor", str(exc))

    async def register(request):
        try:
            body = await request.json()
            if not isinstance(body, dict) or set(body) != {"worker_id", "url"}:
                raise ValueError("worker_id and url are required")
            url = registry.validate_identity(body["worker_id"], body["url"], set(allowed_hosts))
            async with app[LOCK]:
                worker = registry.register(body["worker_id"], url)
            await probe(worker.worker_id)
            return web.json_response(worker.snapshot(registry.clock(), registry.stale_after_s), status=201)
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            return _error(request, 400, "invalid_worker", str(exc))

    async def heartbeat(request):
        worker_id = request.match_info["worker_id"]
        if worker_id not in registry.workers:
            return _error(request, 404, "worker_not_found", "Worker is not registered")
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise ValueError("heartbeat must be a health object")
            async with app[LOCK]:
                worker = registry.update(worker_id, body)
            return web.json_response(worker.snapshot(registry.clock(), registry.stale_after_s))
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            return _error(request, 400, "invalid_heartbeat", str(exc))

    async def drain(request):
        worker_id = request.match_info["worker_id"]
        if worker_id not in registry.workers:
            return _error(request, 404, "worker_not_found", "Worker is not registered")
        try:
            body = await request.json()
            if not isinstance(body, dict) or set(body) != {"draining"}:
                raise ValueError("draining boolean is required")
            async with app[LOCK]:
                worker = registry.set_draining(worker_id, body["draining"])
            return web.json_response(worker.snapshot(registry.clock(), registry.stale_after_s))
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            return _error(request, 400, "invalid_drain", str(exc))

    async def remove(request):
        worker_id = request.match_info["worker_id"]
        if worker_id not in registry.workers:
            return _error(request, 404, "worker_not_found", "Worker is not registered")
        try:
            async with app[LOCK]:
                registry.remove(worker_id)
            return web.Response(status=204)
        except ValueError as exc:
            return _error(request, 409, "worker_busy", str(exc))

    async def prometheus(request):
        return web.Response(body=metrics.render(registry), headers={"Content-Type": CONTENT_TYPE_LATEST})

    async def generate(request):
        try:
            body = await request.json()
            if not isinstance(body, dict) or set(body) - {"prompt", "max_tokens", "timeout_s", "model",
                                                         "artifact_sha256"}:
                raise ValueError("invalid or unsupported request fields")
            prompt = body["prompt"]
            count = body.get("max_tokens", 64)
            timeout = body.get("timeout_s", 60)
            model = body.get("model")
            artifact_sha256 = body.get("artifact_sha256")
            if not isinstance(prompt, str) or not prompt.strip() or len(prompt.encode()) > 16384:
                raise ValueError("prompt must contain 1..16384 UTF-8 bytes")
            if type(count) is not int or not 1 <= count <= 512:
                raise ValueError("max_tokens must be an integer in 1..512")
            if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 120:
                raise ValueError("timeout_s must be in (0, 120]")
            if model is not None and (not isinstance(model, str) or not model):
                raise ValueError("model must be a nonempty string")
            if artifact_sha256 is not None and (not isinstance(artifact_sha256, str) or
                                                len(artifact_sha256) != 64 or
                                                any(character not in "0123456789abcdef" for character in artifact_sha256)):
                raise ValueError("artifact_sha256 must be a lowercase SHA-256 digest")
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            return _error(request, 400, "invalid_request", str(exc))
        estimate = admission.estimate(prompt, count)
        placement_started = time.perf_counter()
        async with app[LOCK]:
            placement = registry.select(estimate.kv_bytes, model=model, artifact_sha256=artifact_sha256,
                                        request_id=request[REQUEST_ID])
        metrics.placement_duration.observe(time.perf_counter() - placement_started)
        if placement is None:
            metrics.rejected.labels(reason="no_capacity").inc()
            return _error(request, 503, "no_worker_capacity", "No healthy compatible worker has capacity",
                          {"Retry-After": "1"})
        metrics.placements.labels(reason=placement.reason).inc()
        forwarded = {key: value for key, value in body.items()
                     if key not in ("model", "artifact_sha256")}
        outcome = "transport_error"
        downstream = None
        worker_request_id = None
        try:
            async with app[SESSION].post(placement.url + "/v1/generate", json=forwarded,
                                         timeout=aiohttp.ClientTimeout(total=timeout + 5)) as upstream:
                if upstream.status != 200:
                    data = await upstream.read()
                    outcome = f"worker_http_{upstream.status}"
                    return web.Response(status=upstream.status, body=data,
                                        content_type=upstream.content_type,
                                        headers={"Retry-After": upstream.headers.get("Retry-After", "1"),
                                                 "X-TensorPilot-Worker": placement.worker_id})
                headers = {"Content-Type": "application/x-ndjson", "Cache-Control": "no-store",
                           "X-TensorPilot-Worker": placement.worker_id,
                           "X-TensorPilot-Placement-Reason": placement.reason}
                if upstream.headers.get("X-Request-ID"):
                    headers["X-Request-ID"] = upstream.headers["X-Request-ID"]
                worker_request_id = upstream.headers.get("X-Request-ID")
                downstream = web.StreamResponse(headers=headers)
                await downstream.prepare(request)
                terminal = False
                async for line in upstream.content:
                    try:
                        event = json.loads(line)
                        if event.get("type") in ("done", "error"):
                            event["placement"] = placement.as_dict()
                            line = (json.dumps(event) + "\n").encode()
                            terminal = True
                    except (ValueError, UnicodeDecodeError):
                        pass
                    await downstream.write(line)
                await downstream.write_eof()
                outcome = "completed" if terminal else "incomplete_stream"
                return downstream
        except (aiohttp.ClientError, TimeoutError, ConnectionError) as exc:
            async with app[LOCK]:
                if placement.worker_id in registry.workers:
                    registry.failure(placement.worker_id, type(exc).__name__)
            if downstream is not None and downstream.prepared:
                event = {"type": "error", "code": "worker_transport_error",
                         "error": "Selected worker connection failed",
                         "request_id": worker_request_id, "placement": placement.as_dict()}
                try:
                    await downstream.write((json.dumps(event) + "\n").encode())
                    await downstream.write_eof()
                except ConnectionError:
                    downstream.force_close()
                return downstream
            return _error(request, 503, "worker_transport_error", "Selected worker connection failed",
                          {"Retry-After": "1", "X-TensorPilot-Worker": placement.worker_id})
        finally:
            async with app[LOCK]:
                registry.release(placement.worker_id)
            metrics.outcomes.labels(outcome=outcome).inc()

    app.cleanup_ctx.append(lifecycle)
    app.router.add_get("/livez", livez)
    app.router.add_get("/readyz", readyz)
    app.router.add_get("/health", health)
    app.router.add_get("/metrics", prometheus)
    app.router.add_get("/v1/workers", list_workers)
    app.router.add_post("/v1/workers", register)
    app.router.add_post("/v1/workers/{worker_id}/heartbeat", heartbeat)
    app.router.add_post("/v1/workers/{worker_id}/drain", drain)
    app.router.add_delete("/v1/workers/{worker_id}", remove)
    app.router.add_post("/v1/generate", generate)
    app.router.add_post("/generate", generate)
    return app


def _error(request, status, code, message, headers=None):
    return web.json_response({"error": {"code": code, "message": message},
                              "request_id": request[REQUEST_ID]}, status=status, headers=headers)


def _worker(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("worker must be ID=http://host:port")
    worker_id, url = value.split("=", 1)
    return worker_id, url


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--worker", action="append", type=_worker, default=[])
    parser.add_argument("--allow-worker-host", action="append", default=[])
    parser.add_argument("--policy", choices=WorkerRegistry.POLICIES, default="capacity-aware")
    parser.add_argument("--stale-after", type=float, default=5)
    parser.add_argument("--probe-interval", type=float, default=1)
    parser.add_argument("--heartbeat-concurrency", type=int, default=16)
    parser.add_argument("--heartbeat-connect-timeout", type=float, default=.5)
    parser.add_argument("--heartbeat-total-timeout", type=float, default=2)
    args = parser.parse_args()
    if (not 1 <= args.port <= 65535 or args.stale_after <= 0 or args.probe_interval <= 0 or
            not 1 <= args.heartbeat_concurrency <= 1024 or
            not 0 < args.heartbeat_connect_timeout <= args.heartbeat_total_timeout):
        parser.error("port must be valid and heartbeat intervals positive")
    api_key = os.environ.get("TENSORPILOT_CONTROL_API_KEY")
    if args.host not in ("127.0.0.1", "localhost", "::1") and not api_key:
        parser.error("set TENSORPILOT_CONTROL_API_KEY before binding a non-loopback address")
    allowed = tuple(dict.fromkeys(("127.0.0.1", "localhost", "::1", *args.allow_worker_host)))
    app = create_app(workers=args.worker, policy=args.policy, stale_after_s=args.stale_after,
                     probe_interval_s=args.probe_interval, allowed_hosts=allowed, api_key=api_key,
                     heartbeat_concurrency=args.heartbeat_concurrency,
                     heartbeat_connect_timeout_s=args.heartbeat_connect_timeout,
                     heartbeat_total_timeout_s=args.heartbeat_total_timeout,
                     worker_api_key=os.environ.get("TENSORPILOT_WORKER_API_KEY"))
    web.run_app(app, host=args.host, port=args.port, handler_cancellation=True, access_log=None)


if __name__ == "__main__":
    main()
