"""Decide, from two 0.5B quick-batch stage directories, whether packed prefill earned a place in the 7B session.

    python -m tensorpilot.packing_gate UNPACKED_STAGE_DIR PACKED_STAGE_DIR [--output packing-verdict.json]

Both stages run the same experiments on the same Pod (``0.5b-quick-batch`` and ``0.5b-quick-batch-packed``). The rule is fixed here,
before any measurement, and printed with the result:

  * a REGRESSION anywhere (throughput below 0.95x the unpacked run at any shared point, P95 TTFT above 1.10x at c=32 or c=128, a
    failed request, an unfinished experiment, a failed memory gate, an invariance result worse than one request below the unpacked
    run or under 0.90) means: run the 7B stages unpacked (``PREFILL_PACK_TOKENS=0``) and investigate;
  * with no regression, packing is USED only if it also helped: at least 1.05x the throughput at c=128, or a P95 TTFT at c=128 of at
    most 0.9x the unpacked run. No measurable gain is not a reason to carry an unmeasured code path into a larger session.

Passing is a go/no-go for one option in one session. It is not a performance result for 7B, where the step is dominated by weights.
"""
import argparse
import json
import sys
from pathlib import Path

from .quick_compare import stage_facts

CRITERIA = {"min_throughput_ratio_everywhere": 0.95, "max_ttft_ratio_at_c32_c128": 1.10,
            "min_throughput_gain_at_c128_to_use": 1.05, "max_ttft_ratio_at_c128_to_use": 0.90,
            "invariance_exact_fraction_min": 0.90, "invariance_tolerance_requests": 1, "invariance_prompts": 32}
SHARED = ("quick-c1", "quick-c8", "quick-c32", "quick-c128")


def judge(unpacked, packed):
    checks, ratios = [], {}

    def check(name, passed, detail):
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    for name in SHARED:
        a, b = unpacked["experiments"].get(name), packed["experiments"].get(name)
        if not (a and b and a["tok_per_s"] and b["tok_per_s"]):
            check(f"{name}_present", False, "missing from one or both stages")
            continue
        ratios[name] = {"throughput": b["tok_per_s"] / a["tok_per_s"],
                        "p95_ttft": (b["p95_ttft_s"] / a["p95_ttft_s"]) if a["p95_ttft_s"] and b["p95_ttft_s"] else None}
        need = CRITERIA["min_throughput_ratio_everywhere"]
        check(f"{name}_throughput_not_worse", ratios[name]["throughput"] >= need,
              f"{b['tok_per_s']:.0f} vs {a['tok_per_s']:.0f} tok/s = {ratios[name]['throughput']:.2f}x (need {need}x)")
    for name in ("quick-c32", "quick-c128"):
        ratio = (ratios.get(name) or {}).get("p95_ttft")
        if ratio is not None:
            limit = CRITERIA["max_ttft_ratio_at_c32_c128"]
            check(f"{name}_p95_ttft_not_worse", ratio <= limit, f"{ratio:.2f}x the unpacked P95 TTFT (limit {limit}x)")
    failing = {f"{label}:{n}": e["failed"] for label, facts in (("unpacked", unpacked), ("packed", packed))
               for n, e in facts["experiments"].items() if e and n != "quick-cancel" and e["failed"]}
    unfinished = [f"{label}:{n}" for label, facts in (("unpacked", unpacked), ("packed", packed))
                  for n, status in facts["statuses"].items() if status != "completed"]
    check("no_failed_requests_and_every_experiment_completed", not failing and not unfinished,
          f"failed: {failing or 'none'}; not completed: {unfinished or 'none'}")
    fractions = {}
    for label, facts in (("unpacked", unpacked), ("packed", packed)):
        inv = facts["invariance"]
        fractions[label] = min((inv[k]["exact_match_fraction"] for k in ("together_vs_solo", "staggered_vs_solo")),
                               default=None) if inv else None
    if None in fractions.values():
        check("invariance_present", False, "invariance.json is missing from a stage")
    else:
        slack = CRITERIA["invariance_tolerance_requests"] / CRITERIA["invariance_prompts"]
        check("invariance_not_worse", fractions["packed"] >= CRITERIA["invariance_exact_fraction_min"]
              and fractions["packed"] >= fractions["unpacked"] - slack - 1e-9,
              f"exact-match fraction {fractions['packed']:.4f} packed vs {fractions['unpacked']:.4f} unpacked")
    gate = packed["memory_gate"]
    check("memory_gate", gate and gate.get("ok"), "; ".join(c["detail"] for c in (gate or {}).get("checks", []))
          or "memory-gate.json is missing")
    check("neither_stage_was_interrupted", not unpacked["interrupted"] and not packed["interrupted"], "")
    no_regression = all(c["passed"] for c in checks)
    c128 = ratios.get("quick-c128") or {}
    helped = bool(c128 and ((c128["throughput"] >= CRITERIA["min_throughput_gain_at_c128_to_use"]) or
                            (c128["p95_ttft"] is not None and c128["p95_ttft"] <= CRITERIA["max_ttft_ratio_at_c128_to_use"])))
    if not no_regression:
        decision, env = "REGRESSION", "PREFILL_PACK_TOKENS=0"
    elif helped:
        decision, env = "USE_PACKING", "(leave the stage default)"
    else:
        decision, env = "NO_GAIN", "PREFILL_PACK_TOKENS=0"
    return {"decision": decision, "run_7b_stages_with": env, "no_regression": no_regression, "helped_at_c128": helped,
            "ratios": ratios, "checks": checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("unpacked_dir")
    parser.add_argument("packed_dir")
    parser.add_argument("--output")
    args = parser.parse_args()
    verdict = judge(stage_facts(args.unpacked_dir), stage_facts(args.packed_dir))
    report = {"schema": "tensorpilot.packing-verdict/v1", "criteria": CRITERIA, **verdict,
              "meaning": "A go/no-go for packed prefill in the 7B session; not a 7B performance result."}
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    for item in verdict["checks"]:
        print(f"[{'PASS' if item['passed'] else 'FAIL'}] {item['name']}: {item['detail']}")
    print(f"PACKING GATE: {verdict['decision']} - run the 7B stages with {verdict['run_7b_stages_with']}")
    return 0 if verdict["decision"] == "USE_PACKING" else 1


if __name__ == "__main__":
    sys.exit(main())
