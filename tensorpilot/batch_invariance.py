"""Does a request's output depend on what else is in the batch? Measure it on a live worker.

With greedy decoding a request should produce the same tokens however it is scheduled. In exact arithmetic that holds.
On a GPU in bf16 or fp16, batching changes tensor shapes and therefore reduction order, so the logits move slightly and a
near-tie can flip: some divergence is expected and is not a bug. What matters is how much, and where it starts. This tool
runs the same prompts three ways against ONE worker and compares every request with its solo run:

  solo      each prompt alone, one at a time (the reference: batch width one)
  together  all prompts at once, so they share decode steps and join together
  staggered the first half starts, the second half joins mid-flight, so sequences enter a batch already decoding

It reports exact-match rate, how far the matching prefix extends, and the first divergence. A request that is wrong
(missing, duplicated or reordered tokens) shows up as a mismatch near position zero and an error in its record; numeric
near-tie flips show up late and rarely. Compare against the serial engine separately (``compare`` on two outputs).

    python -m tensorpilot.batch_invariance --url http://127.0.0.1:8080 --prompts 32 --output invariance.json
"""
import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from . import load
from .reporting import write_json
from .workloads import build_spec_workload, validate_workload

SCHEMA = "tensorpilot.batch-invariance/v1"
SPEC = {"seed": 21, "classes": [
    {"name": "short", "weight": 2, "prompt_tokens": {"dist": "uniform", "min": 12, "max": 64},
     "output_tokens": {"dist": "uniform", "min": 24, "max": 64}},
    {"name": "medium", "weight": 1, "prompt_tokens": {"dist": "uniform", "min": 128, "max": 384},
     "output_tokens": {"dist": "uniform", "min": 24, "max": 64}}]}


def _args(url, path, count, concurrency, *, timeout):
    return SimpleNamespace(url=url, requests=count, concurrency=concurrency, rate=0, max_tokens=64, timeout=timeout,
                           workload="uniform", workload_file=str(path), seed=1, duration_s=None, max_requests=count,
                           journal=None, journal_fsync=False, compact=False, overwrite=True, arrival="uniform",
                           burst_size=8, slo=None, capture_tokens=True)


async def _pass(url, plan, path, concurrency, timeout):
    result = await load.run(_args(url, path, len(plan), concurrency, timeout=timeout))
    return {record["index"]: record for record in result["requests"]}, result["summary"]


async def _staggered(url, plan, path, timeout, head_start_s):
    """Start the first half, then the second half after ``head_start_s`` so its sequences join a running batch."""
    half = len(plan) // 2
    first_path, second_path = Path(str(path) + ".a"), Path(str(path) + ".b")
    first_path.write_text(json.dumps(plan[:half]))
    second_path.write_text(json.dumps(plan[half:]))

    async def second():
        await asyncio.sleep(head_start_s)
        return await load.run(_args(url, second_path, len(plan) - half, len(plan) - half, timeout=timeout))

    first_result, second_result = await asyncio.gather(
        load.run(_args(url, first_path, half, half, timeout=timeout)), second())
    records = {r["index"]: r for r in first_result["requests"]}
    records.update({r["index"] + half: r for r in second_result["requests"]})
    return records


def compare_runs(reference, other):
    """How each request in ``other`` differs from its solo run: exact, prefix agreement, first divergence."""
    rows, exact = [], 0
    for index in sorted(reference):
        solo, test = reference[index], other.get(index)
        if not (solo.get("ok") and test and test.get("ok")):
            rows.append({"index": index, "comparable": False, "solo_ok": bool(solo.get("ok")),
                         "test_ok": bool(test and test.get("ok"))})
            continue
        a, b = solo["token_ids"], test["token_ids"]
        shared = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        same = a == b
        exact += same
        rows.append({"index": index, "comparable": True, "exact": same, "solo_tokens": len(a), "test_tokens": len(b),
                     "solo_token_ids": None if same else a, "test_token_ids": None if same else b,
                     "first_divergence": None if same else shared,
                     "prefix_fraction": shared / max(len(a), len(b), 1)})
    comparable = [r for r in rows if r["comparable"]]
    diverged = [r for r in comparable if not r["exact"]]
    return {"requests": len(rows), "comparable": len(comparable), "exact_matches": exact,
            "exact_match_fraction": exact / len(comparable) if comparable else None,
            "mean_prefix_fraction": (sum(r["prefix_fraction"] for r in comparable) / len(comparable)
                                     if comparable else None),
            "diverged": len(diverged),
            "earliest_divergence": min((r["first_divergence"] for r in diverged), default=None),
            "median_divergence": (sorted(r["first_divergence"] for r in diverged)[len(diverged) // 2]
                                  if diverged else None),
            "not_comparable": len(rows) - len(comparable),
            # The token ids of a diverged request are kept so it can be examined later (divergence_probe) without a rerun.
            "diverged_requests": [{k: r[k] for k in ("index", "first_divergence", "solo_tokens", "test_tokens",
                                                     "solo_token_ids", "test_token_ids")}
                                  for r in diverged][:20]}


async def measure(url, *, prompts=32, timeout=120, head_start_s=0.3, workdir=None, spec=None):
    plan = validate_workload(build_spec_workload(spec or SPEC, prompts))
    directory = Path(workdir or ".")
    path = directory / "invariance-plan.json"
    path.write_text(json.dumps(plan))
    solo, solo_summary = await _pass(url, plan, path, 1, timeout)
    together, together_summary = await _pass(url, plan, path, prompts, timeout)
    staggered = await _staggered(url, plan, path, timeout, head_start_s)
    classes = {i: item["class"] for i, item in enumerate(plan)}
    report = {"schema": SCHEMA, "at_utc": datetime.now(timezone.utc).isoformat(), "url": "(loopback worker)",
            "prompts": prompts,
            "plan_classes": {item["class"]: sum(other["class"] == item["class"] for other in plan) for item in plan},
            "solo_summary": {k: solo_summary[k] for k in ("successful_requests", "failed_requests")},
            "together_summary": {k: together_summary[k] for k in ("successful_requests", "failed_requests")},
            "together_vs_solo": compare_runs(solo, together), "staggered_vs_solo": compare_runs(solo, staggered),
            "reading_guide": ["exact_match_fraction near 1 means a request's output does not depend on its co-batched neighbours",
                              "divergence late in a sequence is the signature of a numeric near-tie; divergence at the start, "
                              "or token-count differences, indicate a real defect",
                              "bf16/fp16 batching changes reduction order, so some divergence on a GPU is expected and is "
                              "measured here rather than assumed away"]}
    for name in ("together_vs_solo", "staggered_vs_solo"):
        for row in report[name]["diverged_requests"]:
            row["class"], row["prompt_tokens"] = classes[row["index"]], plan[row["index"]].get("prompt_tokens")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--prompts", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not 4 <= args.prompts <= 256:
        parser.error("--prompts must be in 4..256")
    output = Path(args.output)
    report = asyncio.run(measure(args.url, prompts=args.prompts, timeout=args.timeout, workdir=output.parent))
    write_json(output, report)
    for name in ("together_vs_solo", "staggered_vs_solo"):
        r = report[name]
        print(f"{name}: {r['exact_matches']}/{r['comparable']} identical to solo "
              f"(mean prefix {r['mean_prefix_fraction']:.3f}, earliest divergence {r['earliest_divergence']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
