"""Bounded worker-crash experiment: kill the worker under load, let the supervisor restart it, then audit the result.

What it does, in order: start the worker under the supervisor and wait for readiness; start an open-loop load
(a steady arrival rate, so the offered load does not collapse when requests start failing); SIGKILL the worker
process mid-run; watch /readyz to timestamp the outage and the recovery; let the load finish; confirm requests
succeed again, authentication still holds, and no process is left behind. Then audit the request journal.

The audit (``verify_accounting``) is the point of the experiment. After a crash it must still be true that
every request has exactly one terminal outcome, no request completed across the crash, no token was delivered
twice, nothing was silently retried, and requests dispatched after recovery complete. Each check reports
pass or fail with the evidence it used; a run is only ``passed`` if every check passes.

Everything is bounded: a restart budget in the supervisor, a hard wall-clock limit on the whole experiment, and a
``finally`` that stops the supervisor (which stops the worker) however the run ends. Run it only against a worker
you started for the purpose; it kills a process.
"""
import argparse
import asyncio
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import aiohttp

from . import load, supervisor
from .reporting import write_json
from .workloads import build_spec_workload, validate_workload

SCHEMA = "tensorpilot.fault-run/v1"
TERMINAL_FAILURES = frozenset(("failed", "rejected", "timed_out"))


def check(name, passed, detail, **evidence):
    return {"name": name, "passed": bool(passed), "detail": detail, **evidence}


def verify_accounting(records, *, kill_offset_s, restored_offset_s, expected_requests, tolerance_s=0.05,
                      recovery_margin_s=0.5):
    """Audit a request journal around a crash. All times are seconds on the load run's own clock."""
    checks = []
    indices = [record["index"] for record in records]
    duplicated = sorted({i for i in indices if indices.count(i) > 1})
    checks.append(check("one_record_per_request", not duplicated and len(records) == expected_requests,
                        f"{len(records)} records for {expected_requests} planned requests"
                        + (f"; duplicated indices {duplicated[:5]}" if duplicated else ""),
                        records=len(records), expected=expected_requests, duplicated=duplicated))
    outcomes = {"completed", "client_cancelled", *TERMINAL_FAILURES}
    unknown = [r["index"] for r in records if r.get("outcome") not in outcomes]
    checks.append(check("every_request_has_exactly_one_terminal_outcome", not unknown,
                        "every record carries one of completed, rejected, timed_out, failed, client_cancelled"
                        if not unknown else f"records without a terminal outcome: {unknown[:5]}"))

    def span(record):
        start = record["dispatch_offset_s"]
        return start, start + record["client_elapsed_s"]

    completed = [r for r in records if r["outcome"] == "completed"]
    mismatched = [r["index"] for r in completed if r["received_output_tokens"] != r["server"]["generated_tokens"]]
    checks.append(check("completed_requests_received_exactly_the_tokens_generated", not mismatched,
                        "client and server token counts agree on every completed request"
                        if not mismatched else f"count mismatch on {mismatched[:5]}"))
    straddling = [r["index"] for r in completed
                  if span(r)[0] < kill_offset_s - tolerance_s and span(r)[1] > kill_offset_s + tolerance_s]
    checks.append(check("no_request_completed_across_the_crash", not straddling,
                        "no completed request was in flight when the worker was killed"
                        if not straddling else f"completed although in flight at the kill: {straddling[:5]}"))
    in_flight = [r for r in records if span(r)[0] <= kill_offset_s <= span(r)[1] and r["outcome"] != "client_cancelled"]
    survivors = [r["index"] for r in in_flight if r["outcome"] == "completed"
                 and span(r)[1] > kill_offset_s + tolerance_s]
    checks.append(check("in_flight_requests_ended_in_a_terminal_failure", not survivors,
                        f"{len(in_flight)} request(s) were in flight at the kill; "
                        f"{sum(r['outcome'] in TERMINAL_FAILURES for r in in_flight)} ended in a terminal failure",
                        in_flight=len(in_flight)))
    outage = [r for r in records if kill_offset_s + tolerance_s <= span(r)[0] < restored_offset_s - tolerance_s]
    leaked = [r["index"] for r in outage if r["outcome"] == "completed"]
    slowest = max((r["client_elapsed_s"] for r in outage), default=0.0)
    checks.append(check("requests_during_the_outage_failed_fast_and_none_completed", not leaked,
                        f"{len(outage)} request(s) were dispatched during the outage; none completed"
                        if not leaked else f"completed during the outage: {leaked[:5]}",
                        dispatched_during_outage=len(outage), slowest_outage_request_s=slowest))
    after = [r for r in records if span(r)[0] >= restored_offset_s + recovery_margin_s]
    failed_after = [r["index"] for r in after if r["outcome"] not in ("completed", "client_cancelled")]
    checks.append(check("requests_after_recovery_complete", bool(after) and not failed_after,
                        f"{len(after)} request(s) were dispatched after recovery; "
                        + ("all completed" if not failed_after else f"failures: {failed_after[:5]}")
                        if after else "no request was dispatched after recovery: lengthen the load window",
                        dispatched_after_recovery=len(after)))
    return checks


def read_lifecycle(path):
    try:
        return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    except FileNotFoundError:
        return []


async def wait_for_ready_events(path, count, timeout_s):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        ready = [event for event in read_lifecycle(path) if event["event"] == "worker_ready"]
        if len(ready) >= count:
            return ready
        await asyncio.sleep(0.05)
    return None


async def watch_readiness(url, session, origin, timeline, stop):
    """Record every change in what /readyz says, with offsets from ``origin`` (the load run's clock)."""
    last = object()
    while not stop.is_set():
        try:
            async with session.get(url + "/readyz", timeout=aiohttp.ClientTimeout(total=1)) as response:
                state = "ready" if response.status == 200 else f"http_{response.status}"
        except (aiohttp.ClientError, TimeoutError):
            state = "unreachable"
        if state != last:
            timeline.append({"offset_s": time.perf_counter() - origin, "state": state})
            last = state
        await asyncio.sleep(0.05)


async def auth_checks(url, api_key):
    """After the restart the worker must still refuse unauthenticated and wrongly authenticated requests."""
    body = {"prompt": " the of", "max_tokens": 1}
    results = []
    async with aiohttp.ClientSession() as session:
        for label, headers, expected in (("no_token", {}, 401), ("wrong_token", {"Authorization": "Bearer wrong"}, 401),
                                         ("valid_token", {"Authorization": f"Bearer {api_key}"}, 200)):
            if label == "valid_token" and not api_key:
                continue
            async with session.post(url + "/v1/generate", json=body, headers=headers) as response:
                await response.read()
                results.append((label, response.status, expected))
    return check("authentication_survives_the_restart", all(status == expected for _, status, expected in results),
                 ", ".join(f"{label}={status}" for label, status, _ in results),
                 observed={label: status for label, status, _ in results})


def pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


async def run_fault_experiment(*, worker_command, url, api_key, output_dir, plan, rate, concurrency, kill_after_s,
                               timeout_s=60, post_requests=5, ready_timeout_s=120, recovery_timeout_s=120,
                               max_seconds=900, max_restarts=2, backoff_s=0.5):
    url = url.rstrip("/")
    if not 0 < kill_after_s < len(plan) / rate:
        raise ValueError("kill_after_s must fall inside the load window (0 < kill_after_s < requests / rate)")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    lifecycle, worker_log = output / "lifecycle.jsonl", output / "worker.log"
    plan_path = output / "workload.json"
    plan_path.write_text(json.dumps(validate_workload(plan)))
    checks, timeline = [], []
    started_utc = datetime.now(timezone.utc).isoformat()
    supervising = asyncio.create_task(supervisor.supervise(
        worker_command, url, str(lifecycle), str(worker_log), max_restarts=max_restarts,
        ready_timeout=ready_timeout_s, backoff_s=backoff_s))
    stop_watching = asyncio.Event()
    watcher = load_task = None
    kill = {}
    try:
        async with asyncio.timeout(max_seconds):
            first = await wait_for_ready_events(lifecycle, 1, ready_timeout_s)
            checks.append(check("worker_became_ready", first is not None,
                                "the supervised worker reported ready" if first else "it never became ready"))
            if first is None:
                return _finish(output, started_utc, checks, timeline, kill, records=[], lifecycle=lifecycle)
            first_pid = first[-1]["pid"]
            args = SimpleNamespace(url=url, requests=len(plan), concurrency=concurrency, rate=rate, max_tokens=512,
                                   timeout=timeout_s, workload="uniform", workload_file=str(plan_path), seed=42,
                                   duration_s=None, max_requests=len(plan), journal=str(output / "load.requests.jsonl"),
                                   journal_fsync=False, compact=False, overwrite=True, arrival="uniform",
                                   burst_size=8, slo=None)
            # The kill is scheduled against this approximate origin; offsets are re-expressed on the load run's own
            # clock (metadata.run_started_perf_s) once it has finished.
            origin = time.perf_counter()
            load_task = asyncio.create_task(load.run(args))

            async def inject():
                await asyncio.sleep(max(0, origin + kill_after_s - time.perf_counter()))
                pid = [e for e in read_lifecycle(lifecycle) if e["event"] == "worker_ready"][-1]["pid"]
                kill.update(pid=pid, at_perf_s=time.perf_counter(), at_utc=datetime.now(timezone.utc).isoformat())
                os.kill(pid, signal.SIGKILL)

            async with aiohttp.ClientSession() as session:
                watcher = asyncio.create_task(watch_readiness(url, session, origin, timeline, stop_watching))
                killer = asyncio.create_task(inject())
                result = await load_task
                await killer
                recovered = await wait_for_ready_events(lifecycle, 2, recovery_timeout_s)
                stop_watching.set()
                await watcher
            t0 = result["metadata"]["run_started_perf_s"]
            drift = t0 - origin
            for entry in timeline:
                entry["offset_s"] -= drift  # re-express on the load run's clock
            kill_offset = kill["at_perf_s"] - t0
            restored = next((e["offset_s"] for e in timeline if e["state"] == "ready" and e["offset_s"] > kill_offset),
                            None)
            checks.append(check("worker_was_killed", bool(kill), f"SIGKILL sent to pid {kill.get('pid')}",
                                pid=kill.get("pid"), first_pid=first_pid))
            checks.append(check("supervisor_restarted_the_worker_with_a_new_process",
                                recovered is not None and recovered[-1]["pid"] != kill.get("pid"),
                                f"restart took {restored - kill_offset:.2f} s from the kill to /readyz"
                                if recovered and restored is not None else "no second worker_ready event"))
            records = result["requests"]
            if restored is not None:
                checks += verify_accounting(records, kill_offset_s=kill_offset, restored_offset_s=restored,
                                            expected_requests=len(plan))
            post = await load.run(SimpleNamespace(
                url=url, requests=post_requests, concurrency=1, rate=0, max_tokens=8, timeout=timeout_s,
                workload="uniform", workload_file=None, seed=1, duration_s=None, max_requests=post_requests,
                journal=None, journal_fsync=False, compact=False, overwrite=True, arrival="uniform", burst_size=8,
                slo=None))
            checks.append(check("fresh_requests_succeed_after_recovery",
                                post["summary"]["successful_requests"] == post_requests,
                                f"{post['summary']['successful_requests']} of {post_requests} succeeded"))
            checks.append(await auth_checks(url, api_key))
    except TimeoutError:
        checks.append(check("finished_within_the_time_limit", False, f"the experiment exceeded {max_seconds} s"))
        records = []
    finally:
        stop_watching.set()
        for task in (watcher, load_task, supervising):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(*(t for t in (watcher, load_task, supervising) if t is not None), return_exceptions=True)
    pids = {e["pid"] for e in read_lifecycle(lifecycle) if e.get("pid")}
    survivors = sorted(pid for pid in pids if pid_alive(pid))
    checks.append(check("no_worker_process_left_running", not survivors,
                        "every worker process the supervisor started is gone" if not survivors
                        else f"still running: {survivors}", survivors=survivors))
    return _finish(output, started_utc, checks, timeline, kill, records=records, lifecycle=lifecycle)


def _finish(output, started_utc, checks, timeline, kill, *, records, lifecycle):
    events = read_lifecycle(lifecycle)
    report = {"schema": SCHEMA, "started_at_utc": started_utc, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
              "passed": all(item["passed"] for item in checks), "checks": checks, "kill": kill,
              "readiness_timeline": timeline, "lifecycle": events,
              "outcomes": {name: sum(r.get("outcome") == name for r in records)
                           for name in sorted({r.get("outcome") for r in records})},
              "safety": ["kills one process: the worker this run started under its own supervisor",
                         "bounded by a restart budget, a wall-clock limit and a finally that stops the supervisor"]}
    write_json(output / "fault-report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:8180")
    parser.add_argument("--port", type=int, default=8180)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workload-spec", required=True, help="workload spec (JSON) for the load during the crash")
    parser.add_argument("--requests", type=int, default=300)
    parser.add_argument("--rate", type=float, default=10.0, help="arrivals per second; keep it below capacity")
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--kill-after", type=float, default=10.0, help="seconds into the load window")
    parser.add_argument("--execution", choices=("serial", "batch"), default="batch")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--queue-size", type=int, default=64)
    parser.add_argument("--ready-timeout", type=float, default=600)
    parser.add_argument("--max-seconds", type=float, default=1800)
    parser.add_argument("--worker-arg", action="append", default=[], help="forwarded to tensorpilot.server")
    args = parser.parse_args()
    plan = build_spec_workload(json.loads(Path(args.workload_spec).read_text()), args.requests)
    command = supervisor.build_command(SimpleNamespace(
        host="127.0.0.1", port=args.port, execution=args.execution, batch_size=args.batch_size,
        queue_size=args.queue_size, memory_gib=6, max_output_tokens=512, offline=False, worker_arg=args.worker_arg))
    report = asyncio.run(run_fault_experiment(
        worker_command=command, url=args.url, api_key=os.environ.get("TENSORPILOT_API_KEY"),
        output_dir=args.output_dir, plan=plan, rate=args.rate, concurrency=args.concurrency,
        kill_after_s=args.kill_after, ready_timeout_s=args.ready_timeout, max_seconds=args.max_seconds))
    for item in report["checks"]:
        print(f"[{'PASS' if item['passed'] else 'FAIL'}] {item['name']}: {item['detail']}")
    print(f"passed: {str(report['passed']).lower()}  report: {Path(args.output_dir) / 'fault-report.json'}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
