"""Explain a batched-vs-solo greedy divergence with the model's own logits, not with an assumption.

``batch_invariance`` says THAT a request's greedy output changed when it was batched (the v1.0 quick session: 31 of 32
identical, one request diverging at output token 7 of 56). It cannot say WHY. Two very different causes look alike from the
outside:

  * a numeric near-tie: the two best tokens are separated by less than the logits' own rounding, so a different reduction
    order (batch width, padded attention width) legitimately flips the argmax. Expected in fp16/bf16, not a defect.
  * a real defect (KV from the wrong slot, a mask or position error): the runs disagree where the model is NOT near a tie.

This tool loads the model once, runs every prompt alone and then all together through the same ``CUDABatchAdapter`` the worker
uses, records the top-2 tokens and logits at every step (``margin_probe``), and for each diverged request reports the logit gaps
in both runs at the first differing token. A divergence is ``near_tie`` only if ALL hold:

  1. the runs agree on the two candidates: solo's runner-up is the batched choice and the batched runner-up is solo's choice;
  2. the gap in each run is at most ``ULP_TOLERANCE`` units in the last place of the logit dtype at that magnitude
     (fp16 spacing is 2**-10 of the magnitude's power of two, bf16 2**-7).

Anything else is ``unexplained`` and the verdict is FAIL for that request. Prefill is exercised the same way in both runs
(prompts are packed when ``--prefill-pack-tokens`` is set), so the probe also covers the packed path.

    python -m tensorpilot.divergence_probe --model M --revision SHA --dtype bfloat16 --prompts 32 --output probe.json
"""
import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "tensorpilot.divergence-probe/v1"
MANTISSA_BITS = {"float16": 10, "bfloat16": 7, "float32": 23}
ULP_TOLERANCE = 4
MAX_PROMPTS = 256


def ulp(dtype, magnitude):
    """Spacing of representable values of ``dtype`` near ``magnitude``."""
    if dtype not in MANTISSA_BITS:
        raise ValueError(f"unknown dtype {dtype!r}")
    magnitude = abs(magnitude)
    exponent = math.floor(math.log2(magnitude)) if magnitude > 0 else 0
    return 2.0 ** (exponent - MANTISSA_BITS[dtype])


def classify(solo_step, batched_step, dtype):
    """Judge one first-divergence step. Each step is (top1_id, top2_id, top1_logit, top2_logit)."""
    a1, a2, av1, av2 = solo_step
    b1, b2, bv1, bv2 = batched_step
    gap_solo, gap_batched = av1 - av2, bv1 - bv2
    spacing = max(ulp(dtype, av1), ulp(dtype, bv1))
    consistent = a2 == b1 and b2 == a1
    tolerance = ULP_TOLERANCE * spacing
    verdict = "near_tie" if consistent and gap_solo <= tolerance and gap_batched <= tolerance else "unexplained"
    return {"verdict": verdict, "candidates_consistent": consistent, "gap_solo": gap_solo, "gap_batched": gap_batched,
            "logit_ulp": spacing, "gap_solo_ulps": gap_solo / spacing, "gap_batched_ulps": gap_batched / spacing,
            "tolerance_ulps": ULP_TOLERANCE}


def run_adapter(adapter, prompts, max_tokens):
    """Drive one adapter to completion. Returns per prompt {'ids': [...], 'steps': [probe tuples]}."""
    adapter.margin_probe = {}
    uid_of = [adapter.insert(prompt, budget) for prompt, budget in zip(prompts, max_tokens)]
    ids = {uid: [] for uid in uid_of}
    guard = 0
    while adapter.sequences:
        for uid, token in adapter.step().events:
            if token.output:
                ids[uid].append(token.token_id)
        guard += 1
        if guard > 100000:
            raise RuntimeError("adapter did not drain")
    return [{"ids": ids[uid], "steps": adapter.margin_probe.get(uid, [])} for uid in uid_of]


def compare(solo, batched, dtype):
    rows = []
    for index, (a, b) in enumerate(zip(solo, batched)):
        if a["ids"] == b["ids"]:
            continue
        at = next((i for i, (x, y) in enumerate(zip(a["ids"], b["ids"])) if x != y), min(len(a["ids"]), len(b["ids"])))
        row = {"index": index, "first_divergence": at, "solo_tokens": len(a["ids"]), "batched_tokens": len(b["ids"])}
        if at < len(a["steps"]) and at < len(b["steps"]) and at < min(len(a["ids"]), len(b["ids"])):
            row.update(classify(a["steps"][at], b["steps"][at], dtype))
        else:   # one run simply ended earlier: that is a length mismatch, never a near-tie
            row.update({"verdict": "unexplained", "reason": "output lengths differ without a differing token"})
        rows.append(row)
    return rows


def build_report(prompts, solo, batched, dtype, settings):
    rows = compare(solo, batched, dtype)
    unexplained = [r for r in rows if r["verdict"] != "near_tie"]
    return {"schema": SCHEMA, "at_utc": datetime.now(timezone.utc).isoformat(), "dtype": dtype, "settings": settings,
            "prompts": len(prompts), "exact_matches": len(prompts) - len(rows), "diverged": len(rows),
            "near_tie": len(rows) - len(unexplained), "unexplained": len(unexplained), "diverged_requests": rows,
            "passed": not unexplained,
            "meaning": ("every divergence is a near-tie within the logit dtype's rounding" if rows and not unexplained
                        else "no divergence" if not rows else "at least one divergence is NOT explained by rounding: "
                        "treat it as a possible defect")}


def make_prompts(count, spec=None):
    from .batch_invariance import SPEC
    from .workloads import build_spec_workload, validate_workload
    plan = validate_workload(build_spec_workload(spec or SPEC, count))
    return [item["prompt"] for item in plan], [item["max_tokens"] for item in plan]


def probe(backend, *, prompts=32, dtype, capacity=None, pack_tokens=0, prefill_step_size=256, kv_budget_bytes=None,
          spec=None):
    from .cuda_batch import CUDABatchAdapter
    texts, budgets = make_prompts(prompts, spec)
    capacity = capacity or prompts
    solo_adapter = CUDABatchAdapter(backend, capacity=1, prefill_step_size=prefill_step_size,
                                    prefill_pack_tokens=pack_tokens, kv_budget_bytes=kv_budget_bytes)
    solo = [run_adapter(solo_adapter, [t], [b])[0] for t, b in zip(texts, budgets)]
    solo_adapter.close()
    together = CUDABatchAdapter(backend, capacity=capacity, prefill_step_size=prefill_step_size,
                                prefill_pack_tokens=pack_tokens, kv_budget_bytes=kv_budget_bytes)
    # The pool planner grants only what fits inside the memory cap and does not raise when that is fewer than asked. A probe that
    # silently ran fewer sequences, or crashed on the 32nd insert, would not be the experiment it claims to be: refuse up front.
    granted, row_tokens = together.capacity, together.row_tokens
    if granted < capacity:
        together.close()
        raise ValueError(f"the KV pool granted {granted} of the {capacity} slots this probe needs (rows of {row_tokens} tokens, "
                         f"limited by {together.plan.limited_by}); lower --max-prompt-tokens or --prompts. The memory cap is not "
                         "relaxed to make the probe fit")
    batched = run_adapter(together, texts, budgets)
    together.close()
    return build_report(texts, solo, batched, dtype, {"capacity": capacity, "granted_capacity": granted,
                                                      "row_tokens": row_tokens, "prefill_pack_tokens": pack_tokens,
                                                      "prefill_step_size": prefill_step_size})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=("float16", "bfloat16", "float32"))
    parser.add_argument("--quantization", default="none", choices=("none", "nf4"))
    parser.add_argument("--cache-dir")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true", help="for local testing only")
    parser.add_argument("--prompts", type=int, default=32)
    parser.add_argument("--capacity", type=int)
    parser.add_argument("--max-prompt-tokens", type=int, default=1024,
                        help="sizes the KV rows (this plus 512 output tokens); the benchmark stages use 1024, and a probe with "
                             "a different value would not share their pool")
    parser.add_argument("--prefill-pack-tokens", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not 4 <= args.prompts <= MAX_PROMPTS:
        parser.error(f"--prompts must be in 4..{MAX_PROMPTS}")
    from .cuda_backend import CUDABackend
    backend = CUDABackend(args.model, revision=args.revision, device=args.device, dtype=args.dtype,
                          quantization=args.quantization, cache_dir=args.cache_dir, offline=args.offline,
                          allow_cpu=args.allow_cpu, max_prompt_tokens=args.max_prompt_tokens)
    report = probe(backend, prompts=args.prompts, dtype=args.dtype, capacity=args.capacity,
                   pack_tokens=args.prefill_pack_tokens)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(f"{report['exact_matches']}/{report['prompts']} identical; {report['near_tie']} near-tie, "
          f"{report['unexplained']} unexplained: {report['meaning']}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
