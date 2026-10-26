"""Explain a batched-vs-solo greedy divergence with the model's own logits, not with an assumption.

``batch_invariance`` says THAT a request's greedy output changed when it was batched. It cannot say WHY. Two very different causes look
alike from the outside:

  * a numeric near-tie: the two runs disagree about a choice the logits cannot settle (the candidates are within a few units in the
    last place of the logit dtype), so a different reduction order, which a different batch shape legitimately causes, flips the argmax.
    Expected in fp16/bf16, not a defect;
  * a real defect (KV from the wrong slot, a mask or position error): the runs disagree where the model is NOT near a tie, or the batched
    logits sit farther from an independent reference than the serial ones do.

This tool loads the model once, runs every prompt alone and then all together through the same ``CUDABatchAdapter`` the worker uses (the token
is chosen by ``argmax``, exactly as in production; the top-``PROBE_TOP_K`` candidates are only recorded), and for each diverged request:

  1. CROSS-EVALUATION. Each run's own logits are asked how far apart the two runs' choices are: solo's logit for its choice minus solo's logit
     for the batched run's choice, and the same in the batched run. A divergence is a ``near_tie`` only if BOTH are at most ``ULP_TOLERANCE``
     units in the last place of the logit dtype at that magnitude (fp16 spacing 2**-10 of the leading power of two, bf16 2**-7), and each
     run's choice appears in the other run's top-k at all. This replaces an earlier rule that required the two runs' runner-ups to mirror each
     other, which wrongly rejected three-way ties, where which token is "second" is arbitrary;
  2. AUDIT. The diverged requests are replayed (the batched pass is repeated exactly, so the same shapes recur), the full logit rows at the
     divergence step are kept from both runs, and a plain Transformers forward of the same context provides an independent reference. The batched
     logits must be no farther from that reference than the solo logits are, give or take ``AUDIT_SLACK_ULPS`` (the tie tolerance). A replay that does not reproduce
     the tokens is reported as non-deterministic and is never a pass.

Anything else is ``unexplained`` and the probe exits non-zero. It also reports how often the model was within one unit of a tie at all
(``tie_exposure``), which is what makes bf16 divergences common.

    python -m tensorpilot.divergence_probe --model M --revision SHA --dtype bfloat16 --prompts 32 --output probe.json
"""
import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "tensorpilot.divergence-probe/v2"
MANTISSA_BITS = {"float16": 10, "bfloat16": 7, "float32": 23, "float64": 52}
ULP_TOLERANCE = 4
AUDIT_SLACK_ULPS = ULP_TOLERANCE   # the same tolerance that defines a tie: a flip of one unit each way already moves the logits two apart
AUDIT_TOP = 64
MAX_PROMPTS = 256


def ulp(dtype, magnitude):
    """Spacing of representable values of ``dtype`` near ``magnitude``."""
    if dtype not in MANTISSA_BITS:
        raise ValueError(f"unknown dtype {dtype!r}")
    magnitude = abs(magnitude)
    exponent = math.floor(math.log2(magnitude)) if magnitude > 0 else 0
    return 2.0 ** (exponent - MANTISSA_BITS[dtype])


def classify(solo_step, batched_step, dtype):
    """Judge one first-divergence step. A step is (top-k ids, top-k logits), the run's own choice first."""
    solo_ids, solo_values = solo_step
    bat_ids, bat_values = batched_step
    a, c = solo_ids[0], bat_ids[0]
    solo, batched = dict(zip(solo_ids, solo_values)), dict(zip(bat_ids, bat_values))
    spacing = max(ulp(dtype, solo_values[0]), ulp(dtype, bat_values[0]))
    tolerance = ULP_TOLERANCE * spacing
    tied = lambda values: sum(abs(values[0] - v) <= spacing for v in values)   # noqa: E731 - candidates within one unit of the top
    row = {"solo_choice": a, "batched_choice": c, "logit_ulp": spacing, "tolerance_ulps": ULP_TOLERANCE,
           "tied_candidates_solo": tied(solo_values), "tied_candidates_batched": tied(bat_values)}
    if c not in solo or a not in batched:
        row.update(verdict="unexplained", reason="one run's choice is not among the other run's top candidates at all",
                   candidates_cross_checked=False)
        return row
    gap_solo, gap_batched = solo[a] - solo[c], batched[c] - batched[a]
    row.update(candidates_cross_checked=True, gap_solo=gap_solo, gap_batched=gap_batched,
               gap_solo_ulps=gap_solo / spacing, gap_batched_ulps=gap_batched / spacing,
               verdict="near_tie" if gap_solo <= tolerance and gap_batched <= tolerance else "unexplained")
    return row


def audit(solo_row, batched_row, reference_row, solo_choice, batched_choice, dtype):
    """Compare the full logit rows of both runs with an independent reference at one step (all three are float lists)."""
    top = sorted(range(len(reference_row)), key=lambda i: -reference_row[i])[:AUDIT_TOP]
    spacing = ulp(dtype, reference_row[top[0]])

    def worst(row):
        return max(abs(row[i] - reference_row[i]) for i in top) / spacing

    err_solo, err_batched = worst(solo_row), worst(batched_row)
    return {"reference_gap_ulps": (reference_row[solo_choice] - reference_row[batched_choice]) / spacing,
            "solo_error_ulps": err_solo, "batched_error_ulps": err_batched,
            "batched_not_worse": err_batched <= err_solo + AUDIT_SLACK_ULPS + 1e-9, "slack_ulps": AUDIT_SLACK_ULPS,
            "compared_candidates": len(top)}


def run_adapter(adapter, prompts, max_tokens):
    """Drive one adapter to completion. Returns per prompt {'ids': [...], 'steps': [(top-k ids, top-k logits)]}."""
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


def tie_exposure(runs, dtype):
    """How often the model was within one unit of a tie at all: the reason bf16 divergences are common."""
    steps = near = exact = 0
    for run in runs:
        for ids, values in run["steps"]:
            if len(values) < 2:
                continue
            steps += 1
            gap = values[0] - values[1]
            exact += gap == 0
            near += gap <= ulp(dtype, values[0])
    return {"steps": steps, "within_one_ulp": near, "exact_ties": exact,
            "fraction_within_one_ulp": near / steps if steps else None}


def build_report(prompts, solo, batched, dtype, settings, rows=None, replay=None):
    rows = compare(solo, batched, dtype) if rows is None else rows
    unexplained = [r for r in rows if r["verdict"] != "near_tie"]
    return {"schema": SCHEMA, "at_utc": datetime.now(timezone.utc).isoformat(), "dtype": dtype, "settings": settings,
            "prompts": len(prompts), "exact_matches": len(prompts) - len(rows), "diverged": len(rows),
            "near_tie": len(rows) - len(unexplained), "unexplained": len(unexplained), "diverged_requests": rows,
            "tie_exposure": {"solo": tie_exposure(solo, dtype), "batched": tie_exposure(batched, dtype)},
            "replay": replay, "passed": not unexplained,
            "meaning": ("every divergence is a near-tie within the logit dtype's rounding and the batched logits are as close to an "
                        "independent reference as the serial ones" if rows and not unexplained
                        else "no divergence" if not rows else "at least one divergence is NOT explained by rounding: "
                        "treat it as a possible defect")}


def make_prompts(count, spec=None):
    from .batch_invariance import SPEC
    from .workloads import build_spec_workload, validate_workload
    plan = validate_workload(build_spec_workload(spec or SPEC, count))
    return [item["prompt"] for item in plan], [item["max_tokens"] for item in plan]


def _adapter(backend, capacity, step, pack, budget, hook, role):
    from .cuda_batch import CUDABatchAdapter
    adapter = CUDABatchAdapter(backend, capacity=capacity, prefill_step_size=step, prefill_pack_tokens=pack,
                               kv_budget_bytes=budget)
    if hook is not None:
        hook(adapter, role)
    return adapter


def _replay_audit(backend, texts, budgets, solo, batched, rows, dtype, *, capacity, pack, step, budget, hook):
    """Replay the diverged requests with full-logit capture and audit each against a plain Transformers forward."""
    import torch
    wanted = {r["index"]: r["first_divergence"] for r in rows if "solo_choice" in r}
    if not wanted:
        return {"audited": 0, "deterministic": True, "details": {}}
    order = sorted(wanted)
    solo_adapter = _adapter(backend, 1, step, pack, budget, hook, "solo")
    solo_adapter.request_index = lambda uid: order[uid]    # the replayed solo adapter numbers only the diverged requests
    solo_adapter.logit_capture_filter = lambda uid, n: uid < len(order) and n == wanted[order[uid]]
    replay_solo = [run_adapter(solo_adapter, [texts[i]], [budgets[i]])[0] for i in order]
    solo_adapter.close()
    # uids in the solo adapter count 0, 1, ... across the sequential runs, so uid k is order[k]
    solo_rows = {order[uid]: row for (uid, n), row in solo_adapter.logit_capture.items()}
    together = _adapter(backend, capacity, step, pack, budget, hook, "batched")
    together.logit_capture_filter = lambda uid, n: uid in wanted and n == wanted[uid]
    replay_batched = run_adapter(together, texts, budgets)
    together.close()
    batched_rows = {uid: row for (uid, n), row in together.logit_capture.items()}
    # The replay is the same experiment only if EVERY request behaved as before: the other requests set the batch shapes.
    deterministic = all(replay_solo[k]["ids"] == solo[i]["ids"] for k, i in enumerate(order)) and \
        all(replay_batched[i]["ids"] == batched[i]["ids"] for i in range(len(batched)))
    details = {}
    for i in order:
        row = next(r for r in rows if r["index"] == i)
        context = backend.tokenizer.encode(texts[i]) + solo[i]["ids"][:wanted[i]]
        with torch.inference_mode():
            reference = backend.model(torch.tensor([context], device=backend.device)).logits[0, -1].float().cpu().tolist()
        details[i] = audit(solo_rows[i].tolist(), batched_rows[i].tolist(), reference, row["solo_choice"],
                           row["batched_choice"], dtype)
    return {"audited": len(order), "deterministic": deterministic, "details": details}


def probe(backend, *, prompts=32, dtype, capacity=None, pack_tokens=0, prefill_step_size=256, kv_budget_bytes=None,
          spec=None, hook=None, audit_replay=True):
    texts, budgets = make_prompts(prompts, spec)
    capacity = capacity or prompts
    solo_adapter = _adapter(backend, 1, prefill_step_size, pack_tokens, kv_budget_bytes, hook, "solo")
    solo = [run_adapter(solo_adapter, [t], [b])[0] for t, b in zip(texts, budgets)]
    solo_adapter.close()
    together = _adapter(backend, capacity, prefill_step_size, pack_tokens, kv_budget_bytes, hook, "batched")
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
    rows = compare(solo, batched, dtype)
    replay = None
    if rows and audit_replay:
        replay = _replay_audit(backend, texts, budgets, solo, batched, rows, dtype, capacity=capacity, pack=pack_tokens,
                               step=prefill_step_size, budget=kv_budget_bytes, hook=hook)
        for row in rows:
            detail = replay["details"].get(row["index"])
            if detail is not None:
                row["audit"] = detail
            if row["verdict"] != "near_tie":
                continue
            if not replay["deterministic"]:
                row.update(verdict="unexplained", reason="the replay did not reproduce the tokens, so the divergence is not a "
                                                          "function of the shapes alone")
            elif detail is not None and not detail["batched_not_worse"]:
                row.update(verdict="unexplained", reason="the batched logits are farther from the reference than the serial logits")
    return build_report(texts, solo, batched, dtype, {"capacity": capacity, "granted_capacity": granted, "row_tokens": row_tokens,
                                                      "prefill_pack_tokens": pack_tokens, "prefill_step_size": prefill_step_size,
                                                      "top_k": 8, "ulp_tolerance": ULP_TOLERANCE}, rows, replay)


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
    exposure = report["tie_exposure"]["solo"]
    print(f"{report['exact_matches']}/{report['prompts']} identical; {report['near_tie']} near-tie, "
          f"{report['unexplained']} unexplained; {exposure['within_one_ulp']}/{exposure['steps']} solo steps were within one ulp of a tie: "
          f"{report['meaning']}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
