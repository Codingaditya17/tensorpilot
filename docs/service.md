# Integrating and operating TensorPilot

TensorPilot currently serves one local MLX model with concurrent HTTP clients,
bounded FIFO admission and serial model execution. This service milestone adds
an application-facing API and operational visibility. It is not a production
availability claim: authentication is a shared token, inference is single-node,
and MLX execution cannot be interrupted inside an individual generation step.

## Start

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m tensorpilot.server
```

The listener starts after model initialization and warmup. Default bind is
127.0.0.1:8080. To require authentication, set `TENSORPILOT_API_KEY` in the
environment before starting the server. Generation, model metadata, detailed
health and metrics then require `Authorization: Bearer <key>`. Only `/livez` and
`/readyz` remain unauthenticated. Non-loopback binding requires a configured key.
Use a TLS reverse proxy for remote access; the built-in listener serves HTTP.
The workload runner reads the same environment variable without putting it in
its saved configuration. The Python client takes an explicit `api_key` argument.

## Versioned API

`POST /v1/generate` accepts only:

```json
{"prompt": "Explain bounded queues.", "max_tokens": 64, "timeout_s": 60}
```

The response is NDJSON, not SSE. `token` events contain `text`, `token_id`,
`output` (false for EOS), and `request_id`. A `done` event carries generated
token count and timing measurements. A stream `error` contains `code`, `error`
(a safe public message), `error_type` and `request_id`. HTTP status remains 200
after headers are sent, so callers must check the terminal event. A stream
without `done` or `error` is incomplete and must be treated as failed.

Before streaming, errors use this JSON contract:

```json
{"error": {"code": "overloaded", "message": "request queue full"}, "request_id": "..."}
```

HTTP errors include invalid requests (400), unauthorized callers (401), missing
routes (404), invalid method (405), bodies larger than 32 KiB (413), and
overload/unavailable worker (503). Full-queue rejection includes Retry-After: 1.
Every response has a service-generated X-Request-ID. Every streaming event uses
the same ID. Caller-supplied IDs are ignored. Do not automatically retry after
receiving output; generation has no idempotency guarantee. Retries before output
should be bounded and delayed, with jitter, to avoid an overload retry storm.

`GET /v1/model` describes the model and request limits. `/generate` and `/health`
remain available for v0.2 compatibility. Unknown generation fields are rejected
so misspelled options do not silently change experiments.

## Python integration

Run from the repository root in the project environment:

```python
import asyncio
import os
from contextlib import aclosing
from tensorpilot.client import Client, TensorPilotError

async def main():
    async with Client(api_key=os.environ.get("TENSORPILOT_API_KEY")) as client:
        try:
            async with aclosing(client.stream("Explain bounded queues.", max_tokens=32)) as stream:
                async for event in stream:
                    if event["type"] == "token":
                        print(event["text"], end="", flush=True)
                    elif event["type"] == "done":
                        print("\nRequest:", event["request_id"])
        except TensorPilotError as exc:
            print(exc.code, exc.request_id, str(exc))

asyncio.run(main())
```

`aclosing` closes the HTTP stream if application code breaks out early. The client
raises `TensorPilotError` for HTTP errors, stream errors, connection failures,
and incomplete streams. It never retries generated work automatically.

## Operations

- `/livez`: process can handle an HTTP request. This is not a GPU progress check.
- `/readyz`: model startup completed, worker task is alive and service is accepting
  work. Returns 503 while shutting down or when the worker has stopped. A full
  queue remains ready; individual excess requests receive 503. A backend call
  longer than `--stall-timeout` (default 30 seconds) marks readiness unhealthy,
  fails active/waiting clients and rejects new work. The watchdog measures host
  call duration, not a GPU hardware fault. Tune it for the deployed model/context.
- `/metrics`: Prometheus exposition with HTTP handler statuses, admitted/rejected
  jobs, worker outcomes, generated tokens, queue depth, active execution, queue
  wait histogram, and server TTFT histogram. Labels have bounded values and never
  contain prompt text or request IDs. A scrape reflects the state before its own
  handler completes.
- Logs: structured HTTP completion records with request ID, status, method, and
  handler duration. Worker failure diagnostics contain only request ID, error
  type and public code. Prompts, response text, headers and keys are not logged.

Worker outcome `completed` means generation finished and its terminal event was
placed in the bounded output buffer. It does not prove delivery to the client.
Use client reports for delivered-request success. Output-token counters include
partial failed/cancelled jobs; workload throughput includes only successful jobs.
HTTP counters include probes and metrics requests, and HTTP 200 can contain a
stream failure. Do not use the HTTP 200 count as an inference-success SLI.

On shutdown, admission closes, waiting jobs receive a shutdown error, and active
execution is cancelled at a generation boundary. The shutdown hook runs before
HTTP handlers are drained. Cleanup is idempotent. An in-flight GPU operation can
delay shutdown. The progress watchdog changes health and fails clients but cannot
interrupt the thread; a process supervisor must restart a persistently stuck model.

Suggested first alerts to evaluate under measured load: sustained queue depth,
rising rejected/admitted ratio, worker failures, and tail queue wait. No alert
threshold or latency SLO has been validated yet.

## Validation and deployment limits

```bash
python -m pytest -q
python -m tensorpilot.load --requests 10 --concurrency 2 --max-tokens 32 --output results/service_smoke.json
```

CPU CI installs only `requirements-service.txt` and tests API behavior against a
fake backend on Python 3.11 and 3.14. It does not validate GPU speed, model quality,
or Metal compatibility. The GitHub Actions workflow runs once this directory is
published as its own repository; it has not been run remotely by this change.

No Kubernetes manifests or Linux inference container are provided: local MLX
execution needs macOS/Metal. Shared-token auth provides no per-tenant quotas,
roles, rotation protocol or audit store. There is no durable request queue,
multi-node failover, pinned model snapshot, or system memory-pressure controller.
Those require separate implementation and validation before external deployment.
