"""Pass/fail gate and sanitized diagnosis for the gateway smoke run.

An aggregate error rate says a run failed but not why. On the first real GPU Pod, five requests failed
in 15 ms and the cause (a worker 401 relayed by the gateway) was only visible after matching request ids
across three files. This prints the failure grouped by HTTP status and error code, the first failing
request, a plain-language hint, and the tail of each service log, with credentials redacted.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from .gpu_preflight import redact

MESSAGE_CHARS = 240
LOG_LINES = 5

HINTS = (
    (lambda code, status, message: code == "worker_unauthorized",
     "the data plane could not authenticate to the worker: set TENSORPILOT_WORKER_API_KEY on the data plane "
     "to the worker's TENSORPILOT_API_KEY"),
    (lambda code, status, message: code == "worker_host_not_allowed",
     "the routed worker host is not on the data plane's allow-list (--allow-worker-host)"),
    (lambda code, status, message: status == 401 and "data-plane token" in message,
     "the gateway rejected the client token: the load client must use TENSORPILOT_DATA_API_KEY"),
    (lambda code, status, message: status == 401,
     "a service behind the gateway rejected the request with 401; an older data plane relays the worker's 401, "
     "so check the data plane's worker credential"),
    (lambda code, status, message: code in ("transport_error", "control_plane_unavailable", "worker_transport_error"),
     "a service was unreachable; check that it is running and listening on the expected loopback port"),
    (lambda code, status, message: status == 503, "no ready worker was available; check the worker readiness log"),
)


def _message(record):
    raw = record.get("error")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)["error"]["message"]
        except (ValueError, KeyError, TypeError):
            pass
    return redact(str(raw or ""))[:MESSAGE_CHARS]


def diagnose(report, log_paths=()):
    requests = report.get("requests", [])
    failed = [record for record in requests if not record.get("ok")]
    lines = [f"gateway smoke: {len(requests) - len(failed)} of {len(requests)} streams validated"]
    groups = Counter((record.get("http_status"), record.get("error_code")) for record in failed)
    for (status, code), count in groups.most_common():
        lines.append(f"  {count} x http_status={status} error_code={code}")
    if failed:
        first = failed[0]
        message = _message(first)
        lines.append(f"  first failure: index={first.get('index')} request_id={first.get('request_id')} "
                     f"after {first.get('client_elapsed_s')} s: {message}")
        for matches, hint in HINTS:
            if matches(first.get("error_code"), first.get("http_status"), message):
                lines.append(f"  likely cause: {hint}")
                break
    for path in log_paths:
        try:
            tail = Path(path).read_text(errors="replace").splitlines()[-LOG_LINES:]
        except OSError as error:
            tail = [f"(unreadable: {type(error).__name__})"]
        lines.append(f"  --- last {len(tail)} line(s) of {Path(path).name}")
        lines.extend("  " + redact(line)[:300] for line in tail)
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", help="gateway-smoke.json written by tensorpilot.load")
    parser.add_argument("--expect", type=int, required=True, help="number of requests that must succeed")
    parser.add_argument("--log", action="append", default=[], help="service log to tail on failure")
    args = parser.parse_args(argv)
    try:
        report = json.loads(Path(args.report).read_text())
        successful = report["summary"]["successful_requests"]
        failed = report["summary"]["failed_requests"]
    except (OSError, ValueError, KeyError) as error:
        print(f"gateway smoke produced no readable report ({type(error).__name__}); the load client likely "
              "crashed before sending requests", file=sys.stderr)
        print(diagnose({}, args.log), file=sys.stderr)
        return 1
    if failed == 0 and successful == args.expect:
        print(f"gateway smoke: {successful} of {args.expect} streams validated through the data plane")
        return 0
    print(diagnose(report, args.log), file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
