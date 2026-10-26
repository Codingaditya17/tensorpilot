"""TensorPilot streaming data plane: obtain a route lease, then proxy it."""
import argparse
import asyncio
import hmac
import json
import os
import time
import uuid
from urllib.parse import urlparse

import aiohttp
from aiohttp import web
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST

from . import __version__
from .control_plane import _validate_generate_body

CONTROL_SESSION = web.AppKey("control_session", aiohttp.ClientSession)
WORKER_SESSION = web.AppKey("worker_session", aiohttp.ClientSession)
METRICS = web.AppKey("metrics", object)
REQUEST_ID = web.RequestKey("request_id", str) if hasattr(web, "RequestKey") else "request_id"


class DataPlaneMetrics:
    def __init__(self):
        self.registry = CollectorRegistry()
        options = {"registry": self.registry}
        self.active = Gauge("tensorpilot_data_plane_active_streams", "Active gateway streams", **options)
        self.duration = Histogram("tensorpilot_data_plane_stream_duration_seconds", "Gateway stream duration", **options)
        self.placement = Histogram("tensorpilot_data_plane_placement_seconds", "Control-plane placement latency", **options)
        self.chunks = Counter("tensorpilot_data_plane_chunks_total", "Proxied stream chunks", **options)
        self.outcomes = Counter("tensorpilot_data_plane_outcomes_total", "Gateway outcomes", ["outcome"], **options)


LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")


def create_app(*, control_url, api_key=None, control_api_key=None, max_connections=2048,
               session_factory=aiohttp.ClientSession, worker_api_key=None, allowed_worker_hosts=LOOPBACK_HOSTS):
    control_url = control_url.rstrip("/")
    metrics = DataPlaneMetrics()

    @web.middleware
    async def operations(request, handler):
        request[REQUEST_ID] = uuid.uuid4().hex
        if api_key and request.path not in ("/livez", "/readyz"):
            supplied = request.headers.get("Authorization", "")
            if not hmac.compare_digest(supplied.encode(), ("Bearer " + api_key).encode()):
                return _error(request, 401, "unauthorized", "A valid data-plane token is required")
        try:
            response = await handler(request)
        except web.HTTPException as exc:
            response = _error(request, exc.status, "http_error", exc.reason)
        if not response.prepared:
            response.headers.setdefault("X-Request-ID", request[REQUEST_ID])
        return response

    app = web.Application(client_max_size=32 * 1024, middlewares=[operations])
    app[METRICS] = metrics

    async def lifecycle(app):
        headers = {"Authorization": "Bearer " + control_api_key} if control_api_key else {}
        app[CONTROL_SESSION] = session_factory(headers=headers,
                                               connector=aiohttp.TCPConnector(limit=max_connections))
        app[WORKER_SESSION] = session_factory(connector=aiohttp.TCPConnector(limit=max_connections))
        try:
            yield
        finally:
            await app[CONTROL_SESSION].close()
            await app[WORKER_SESSION].close()

    async def livez(request):
        return web.json_response({"alive": True, "service": "tensorpilot-data-plane"})

    async def readyz(request):
        try:
            async with app[CONTROL_SESSION].get(control_url + "/readyz", timeout=aiohttp.ClientTimeout(total=1)) as response:
                ready = response.status == 200
        except (aiohttp.ClientError, TimeoutError):
            ready = False
        return web.json_response({"ready": ready}, status=200 if ready else 503)

    async def health(request):
        return web.json_response({"ready": True, "service": "tensorpilot-data-plane",
                                  "service_version": __version__})

    async def prometheus(request):
        return web.Response(body=generate_latest(metrics.registry),
                            headers={"Content-Type": CONTENT_TYPE_LATEST})

    async def complete_route(route_id, outcome):
        try:
            async with app[CONTROL_SESSION].post(control_url + f"/v1/routes/{route_id}/complete",
                                                 json={"outcome": outcome},
                                                 timeout=aiohttp.ClientTimeout(total=2)):
                pass
        except (aiohttp.ClientError, TimeoutError):
            pass  # The bounded control-plane lease reaper releases abandoned capacity.

    async def generate(request):
        try:
            body = await request.json()
            _, _, timeout, _, _ = _validate_generate_body(body)
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            return _error(request, 400, "invalid_request", str(exc))

        placement_started = time.perf_counter()
        try:
            async with app[CONTROL_SESSION].post(control_url + "/v1/routes", json=body,
                                         timeout=aiohttp.ClientTimeout(total=5)) as response:
                route_body = await response.json()
                if response.status != 201:
                    return web.json_response(route_body, status=response.status,
                                             headers={"Retry-After": response.headers.get("Retry-After", "1")})
        except (aiohttp.ClientError, TimeoutError, json.JSONDecodeError):
            return _error(request, 503, "control_plane_unavailable", "Control plane placement failed",
                          {"Retry-After": "1"})
        finally:
            metrics.placement.observe(time.perf_counter() - placement_started)

        route_id = route_body["route_id"]
        placement = route_body["placement"]
        forwarded = {key: value for key, value in body.items() if key not in ("model", "artifact_sha256")}
        downstream = None
        worker_request_id = None
        # A caller that disconnects or cancels is not evidence that the worker failed.
        outcome = "client_disconnected"
        started = time.perf_counter()
        metrics.active.inc()
        try:
            # The worker credential is attached per request and only for an allow-listed worker host, so
            # a routed URL cannot steer it to an arbitrary destination.
            worker_headers = {}
            if worker_api_key:
                if urlparse(route_body["worker_url"]).hostname not in allowed_worker_hosts:
                    outcome = "worker_host_rejected"
                    return _error(request, 502, "worker_host_not_allowed",
                                  "Routed worker host may not receive the worker credential",
                                  {"X-TensorPilot-Worker": route_body["worker_id"]})
                worker_headers = {"Authorization": "Bearer " + worker_api_key}
            async with app[WORKER_SESSION].post(route_body["worker_url"] + "/v1/generate", json=forwarded,
                                         headers=worker_headers, allow_redirects=False,
                                         timeout=aiohttp.ClientTimeout(total=timeout + 5)) as upstream:
                if 300 <= upstream.status < 400:
                    # Never followed: a redirect would reach a host that was not checked against the allow-list.
                    outcome = "worker_redirect_refused"
                    return _error(request, 502, "worker_redirect_refused",
                                  "Worker answered with a redirect, which is never followed",
                                  {"X-TensorPilot-Worker": route_body["worker_id"]})
                if upstream.status in (401, 403):
                    # The caller authenticated to the data plane; the failure is internal. Relaying the
                    # worker's 401 would blame the caller's token, so it is reported as a gateway fault.
                    await upstream.read()
                    outcome = "worker_unauthorized"
                    return _error(request, 502, "worker_unauthorized",
                                  "Worker rejected the data plane's credential; check TENSORPILOT_WORKER_API_KEY "
                                  "on the data plane", {"X-TensorPilot-Worker": route_body["worker_id"]})
                if upstream.status != 200:
                    data = await upstream.read()
                    outcome = f"worker_http_{upstream.status}"
                    return web.Response(status=upstream.status, body=data, content_type=upstream.content_type,
                                        headers={"Retry-After": upstream.headers.get("Retry-After", "1"),
                                                 "X-TensorPilot-Worker": route_body["worker_id"]})
                worker_request_id = upstream.headers.get("X-Request-ID")
                headers = {"Content-Type": "application/x-ndjson", "Cache-Control": "no-store",
                           "X-TensorPilot-Worker": route_body["worker_id"],
                           "X-TensorPilot-Placement-Reason": placement["reason"]}
                if worker_request_id:
                    headers["X-Request-ID"] = worker_request_id
                downstream = web.StreamResponse(headers=headers)
                terminal = False
                try:
                    await downstream.prepare(request)
                    async for line in upstream.content:
                        try:
                            event = json.loads(line)
                            if event.get("type") in ("done", "error"):
                                event["placement"] = placement
                                line = (json.dumps(event) + "\n").encode()
                                terminal = True
                        except (ValueError, UnicodeDecodeError):
                            pass
                        await downstream.write(line)
                        metrics.chunks.inc()
                    await downstream.write_eof()
                except ConnectionError:
                    # Only the caller's socket raises builtin ConnectionError here. Closing
                    # upstream lets the worker cancel decode; the worker is not penalized.
                    downstream.force_close()
                    return downstream
                outcome = "completed" if terminal else "incomplete_stream"
                return downstream
        except (aiohttp.ClientError, TimeoutError):
            outcome = "worker_transport_error"
            if downstream is not None and downstream.prepared:
                event = {"type": "error", "code": "worker_transport_error",
                         "error": "Selected worker connection failed", "request_id": worker_request_id,
                         "placement": placement}
                try:
                    await downstream.write((json.dumps(event) + "\n").encode())
                    await downstream.write_eof()
                except ConnectionError:
                    downstream.force_close()
                return downstream
            return _error(request, 503, "worker_transport_error", "Selected worker connection failed",
                          {"Retry-After": "1", "X-TensorPilot-Worker": route_body["worker_id"]})
        finally:
            try:
                await asyncio.shield(complete_route(route_id, outcome))
            except asyncio.CancelledError:
                raise
            metrics.active.dec()
            metrics.duration.observe(time.perf_counter() - started)
            metrics.outcomes.labels(outcome=outcome).inc()

    app.cleanup_ctx.append(lifecycle)
    app.router.add_get("/livez", livez)
    app.router.add_get("/readyz", readyz)
    app.router.add_get("/health", health)
    app.router.add_get("/metrics", prometheus)
    app.router.add_post("/v1/generate", generate)
    app.router.add_post("/generate", generate)
    return app


def _error(request, status, code, message, headers=None):
    return web.json_response({"error": {"code": code, "message": message},
                              "request_id": request[REQUEST_ID]}, status=status, headers=headers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--control-url", default="http://127.0.0.1:8090")
    parser.add_argument("--max-connections", type=int, default=2048)
    parser.add_argument("--allow-worker-host", action="append", default=[],
                        help="extra worker host that may receive TENSORPILOT_WORKER_API_KEY (loopback is always allowed)")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or not 1 <= args.max_connections <= 100_000:
        parser.error("port and max-connections must be positive")
    api_key = os.environ.get("TENSORPILOT_DATA_API_KEY")
    if args.host not in ("127.0.0.1", "localhost", "::1") and not api_key:
        parser.error("set TENSORPILOT_DATA_API_KEY before binding a non-loopback address")
    web.run_app(create_app(control_url=args.control_url, api_key=api_key,
                           control_api_key=os.environ.get("TENSORPILOT_CONTROL_API_KEY"),
                           max_connections=args.max_connections,
                           worker_api_key=os.environ.get("TENSORPILOT_WORKER_API_KEY"),
                           allowed_worker_hosts=tuple(dict.fromkeys((*LOOPBACK_HOSTS, *args.allow_worker_host)))), host=args.host, port=args.port,
                handler_cancellation=True, access_log=None)


if __name__ == "__main__":
    main()
