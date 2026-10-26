"""Check the real memory margin on the GPU before a stage spends time benchmarking.

The planner guarantees a margin by construction, but that is arithmetic on estimates. This reads the worker's own measured
allocator numbers right after it has loaded the model and allocated its KV pool, and refuses to continue if the real margin is
thinner than required or the pool is smaller than the stage asked for. It uses the process-lifetime PEAK reserved bytes, so a
transient during loading counts. The margin is measured against the allocator cap (the fraction of the card PyTorch may use);
the rest of the card holds the CUDA context and is not counted as margin.

    python -m tensorpilot.memory_gate --url http://127.0.0.1:8080 --expect-rows 48 --min-margin-gib 1.5 --output gate.json
"""
import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

GIB = 1024**3


def evaluate(health, *, min_margin_gib, expect_rows=None):
    memory = health.get("last_measured_memory") or {}
    gpu = health.get("gpu") or {}
    total, fraction = gpu.get("total_memory_bytes"), health.get("cuda_memory_fraction")
    peak = memory.get("cuda_peak_reserved_bytes") or memory.get("cuda_reserved_bytes")
    checks = []
    if not (total and fraction and peak):
        checks.append({"name": "memory_information_present", "passed": False,
                       "detail": "the worker did not report the card size, the allocator fraction and reserved bytes; the "
                                 "margin cannot be verified, so the stage must not continue"})
        return {"ok": False, "checks": checks}
    cap = total * fraction
    margin = cap - peak
    checks.append({"name": "measured_margin_inside_the_allocator_cap", "passed": margin >= min_margin_gib * GIB,
                   "detail": f"{margin / GIB:.2f} GiB free of a {cap / GIB:.2f} GiB cap at peak reserved "
                             f"{peak / GIB:.2f} GiB (need {min_margin_gib:g} GiB)",
                   "margin_gib": round(margin / GIB, 3), "cap_gib": round(cap / GIB, 3), "peak_reserved_gib": round(peak / GIB, 3)})
    if memory.get("cuda_oom_events"):
        checks.append({"name": "no_out_of_memory_events", "passed": False,
                       "detail": f"{memory['cuda_oom_events']} OOM event(s) already recorded"})
    pool = health.get("kv_pool")
    if expect_rows is not None:
        granted = (pool or {}).get("capacity_rows")
        checks.append({"name": "pool_holds_the_requested_slots", "passed": bool(granted) and granted >= expect_rows,
                       "detail": f"{granted} slot(s) granted, {expect_rows} requested"
                       if pool else "no KV pool reported although a batching stage requested one",
                       "granted": granted, "requested": expect_rows})
    return {"ok": all(c["passed"] for c in checks), "checks": checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--min-margin-gib", type=float, default=1.5)
    parser.add_argument("--expect-rows", type=int)
    parser.add_argument("--output")
    args = parser.parse_args()
    if not args.url.startswith(("http://", "https://")):
        parser.error("--url must be an http or https URL")
    key = os.environ.get("TENSORPILOT_API_KEY")
    request = urllib.request.Request(args.url.rstrip("/") + "/health",
                                     headers={"Authorization": "Bearer " + key} if key else {})
    with urllib.request.urlopen(request, timeout=15) as response:  # nosec B310 - scheme validated above
        health = json.loads(response.read())
    result = evaluate(health, min_margin_gib=args.min_margin_gib, expect_rows=args.expect_rows)
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    for check in result["checks"]:
        print(f"[{'PASS' if check['passed'] else 'FAIL'}] {check['name']}: {check['detail']}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
