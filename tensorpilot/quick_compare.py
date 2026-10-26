"""Read the two quick-session stage directories and say, from their evidence alone, whether CUDA batching earned the next step.

    python -m tensorpilot.quick_compare SERIAL_STAGE_DIR BATCH_STAGE_DIR [--output quick-verdict.json]

The criteria are a gate for spending more GPU time, chosen before any measurement and printed with the result. They are not a
performance claim: passing them means "worth running 7B", nothing more. Anything missing is reported as missing, never inferred.
"""
import argparse
import json
import sys
from pathlib import Path

CRITERIA = {
    "batched_c8_throughput_over_serial_c8": 1.5,   # eight clients decoded together should clearly beat eight queued behind one
    "batched_c1_throughput_over_serial_c1": 0.7,   # batching must not badly slow a lone stream
    "invariance_exact_fraction_min": 0.9,          # most requests must match their solo output; divergence is reported either way
}


def load(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def experiment_summary(stage_dir, name):
    """Pooled over the experiment's repeats; None if it produced no result."""
    directory = Path(stage_dir) / "bench" / name
    reports = [load(p) for p in sorted(directory.glob("repeat*.json")) if not p.name.endswith(".workload.json")]
    summaries = [r["summary"] for r in reports if r and "summary" in r]
    if not summaries:
        return None
    tokens, wall = sum(s["completed_output_tokens"] for s in summaries), sum(s["wall_s"] for s in summaries)
    ttfts = [s["p95_client_ttft_s"] for s in summaries if s.get("p95_client_ttft_s") is not None]
    return {"repeats": len(summaries), "requests": sum(s["requests_total"] for s in summaries),
            "completed": sum(s["successful_requests"] for s in summaries),
            "failed": sum(s["failed_requests"] for s in summaries),
            "tok_per_s": tokens / wall if wall else None, "p95_ttft_s": max(ttfts) if ttfts else None}


def stage_facts(stage_dir):
    manifest = load(Path(stage_dir) / "bench" / "manifest.json", {})
    return {"manifest_present": bool(manifest), "interrupted": bool(manifest.get("interrupted")),
            "experiments": {name: experiment_summary(stage_dir, name) for name in manifest.get("experiments", {})},
            "statuses": {name: e.get("status") for name, e in manifest.get("experiments", {}).items()},
            "invariance": load(Path(stage_dir) / "invariance.json"),
            "memory_gate": load(Path(stage_dir) / "memory-gate.json")}


def judge(serial, batch):
    checks = []

    def check(name, passed, detail):
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    s1, s8 = serial["experiments"].get("quick-c1"), serial["experiments"].get("quick-c8")
    b1, b8 = batch["experiments"].get("quick-c1"), batch["experiments"].get("quick-c8")
    if not all(e and e["tok_per_s"] for e in (s1, s8, b1, b8)):
        check("evidence_present", False, "quick-c1 and quick-c8 results are missing from one or both stages")
    else:
        need8, need1 = CRITERIA["batched_c8_throughput_over_serial_c8"], CRITERIA["batched_c1_throughput_over_serial_c1"]
        ratio8, ratio1 = b8["tok_per_s"] / s8["tok_per_s"], b1["tok_per_s"] / s1["tok_per_s"]
        check("batched_c8_beats_serial_c8", ratio8 >= need8,
              f"{b8['tok_per_s']:.0f} vs {s8['tok_per_s']:.0f} tok/s = {ratio8:.2f}x (need {need8}x)")
        check("batched_c1_not_much_slower", ratio1 >= need1,
              f"{b1['tok_per_s']:.0f} vs {s1['tok_per_s']:.0f} tok/s = {ratio1:.2f}x (need {need1}x)")
    stages = (("serial", serial), ("batch", batch))
    # The cancellation experiment is expected to end requests early, so its failures are not failures of the stage.
    failing = {f"{stage}:{name}": e["failed"] for stage, facts in stages for name, e in facts["experiments"].items()
               if e and name != "quick-cancel" and e["failed"]}
    unfinished = [f"{stage}:{name}" for stage, facts in stages for name, status in facts["statuses"].items()
                  if status != "completed"]
    check("no_failed_requests_and_every_experiment_completed", not failing and not unfinished,
          f"failed requests: {failing or 'none'}; not completed: {unfinished or 'none'}")
    invariance = batch["invariance"]
    if not invariance:
        check("batch_invariance_present", False, "invariance.json is missing from the batched stage")
    else:
        fractions = {k: invariance[k].get("exact_match_fraction") for k in ("together_vs_solo", "staggered_vs_solo")}
        early = {k: invariance[k].get("earliest_divergence") for k in fractions}
        ok = all(v is not None and v >= CRITERIA["invariance_exact_fraction_min"] for v in fractions.values())
        check("batch_invariance", ok, f"exact-match fraction {fractions}; earliest divergence (token index) {early}. "
              "Divergence at the very start of a sequence would indicate a defect rather than a numeric near-tie.")
    gate = batch["memory_gate"]
    check("memory_gate", gate and gate.get("ok"),
          "; ".join(c["detail"] for c in (gate or {}).get("checks", [])) or "memory-gate.json is missing")
    check("neither_stage_was_interrupted", not serial["interrupted"] and not batch["interrupted"], "")
    return {"passed": all(c["passed"] for c in checks), "checks": checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("serial_dir")
    parser.add_argument("batch_dir")
    parser.add_argument("--output")
    args = parser.parse_args()
    serial, batch = stage_facts(args.serial_dir), stage_facts(args.batch_dir)
    verdict = judge(serial, batch)
    report = {"schema": "tensorpilot.quick-verdict/v1", "criteria": CRITERIA, "serial": serial, "batch": batch, **verdict,
              "meaning": "PASS only means the 7B session is worth its cost. It is not a performance result."}
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    for check in verdict["checks"]:
        print(f"[{'PASS' if check['passed'] else 'FAIL'}] {check['name']}: {check['detail']}")
    print("QUICK GATE:", "PASS - the next stage is justified" if verdict["passed"] else "FAIL - do not spend more GPU time yet")
    return 0 if verdict["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
