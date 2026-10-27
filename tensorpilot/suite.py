"""The v1.0 GPU evaluation suite, defined once and generated into benchmarks/gpu/v10-*.json.

One definition produces every stage, so the TensorPilot and vLLM stages use identical workload specs, request
counts and arrival rates by construction; tests assert the committed files equal this generator's output.

Time budgets are planning numbers, not predictions. Each experiment's expected minutes come from a stated model
(a single-stream decode rate for the model, scaled by a batching-efficiency exponent), and its hard cap is a
multiple of that. The caps are what bound cost: if the assumptions are wrong the experiment is aborted at its cap
and the evidence collected so far is kept. Single-stream rates other than the measured 0.5B figure are assumptions.

    python -m tensorpilot.suite --write benchmarks/gpu     # regenerate the committed configs
    python -m tensorpilot.suite --plan                     # print expected and capped minutes per stage
"""
import argparse
import json
import math
import shlex
import sys
from pathlib import Path

from .workloads import build_spec_workload

SCHEMA = "tensorpilot.gpu-experiments/v1"
BATCHING_EFFICIENCY = 0.8     # aggregate tokens/s ~ single_stream * min(concurrency, slots) ** 0.8 (an assumption)
SETUP_MINUTES = 0.5           # per-experiment overhead in the plan
CAP_FACTOR = 2.5

MODELS = {
    "0.5b": {"repo": "Qwen/Qwen2.5-0.5B-Instruct", "revision": "7ae557604adf67be50417f59c2c2f167def9a775",
             "single_stream_tok_s": 58, "single_stream_source": "measured on an RTX 4090 with the serial worker (v0.9.1-4)",
             "slots": 128, "slo": {"ttft_s": 1.0, "e2e_s": 10.0, "tpot_s": 0.05}},
    "7b": {"repo": "Qwen/Qwen2.5-7B-Instruct", "revision": "a09a35458c702b33eeacc393d103063234e8bc28",
           "single_stream_tok_s": 40, "single_stream_source": "ASSUMPTION: bf16 weight reads bound decode near 45 tok/s",
           "slots": 48, "slo": {"ttft_s": 2.0, "e2e_s": 20.0, "tpot_s": 0.15}},
    "14b": {"repo": "Qwen/Qwen2.5-14B-Instruct", "revision": "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8",
            "single_stream_tok_s": 25, "single_stream_source": "ASSUMPTION: NF4 dequantization overhead, unmeasured",
            "slots": 28, "slo": {"ttft_s": 4.0, "e2e_s": 40.0, "tpot_s": 0.35}},
}

CLASS_SHORT = {"name": "short", "prompt_tokens": {"dist": "uniform", "min": 16, "max": 64},
               "output_tokens": {"dist": "uniform", "min": 16, "max": 64}}
CLASS_MEDIUM = {"name": "medium", "prompt_tokens": {"dist": "uniform", "min": 128, "max": 384},
                "output_tokens": {"dist": "uniform", "min": 64, "max": 192}}
CLASS_LONG = {"name": "long", "prompt_tokens": {"dist": "uniform", "min": 512, "max": 896},
              "output_tokens": {"dist": "uniform", "min": 128, "max": 320}}


def spec(seed, classes, **extra):
    return {"seed": seed, "classes": classes, **extra}


def weighted(entry, weight):
    return {**entry, "weight": weight}


SHORT = spec(11, [weighted(CLASS_SHORT, 1)])
MIXED = spec(12, [weighted(CLASS_SHORT, 6), weighted(CLASS_MEDIUM, 3), weighted(CLASS_LONG, 1)])
CANCEL = spec(13, [weighted(CLASS_SHORT, 6), weighted(CLASS_MEDIUM, 3), weighted(CLASS_LONG, 1)],
              cancel={"fraction": 0.3, "after_tokens": {"dist": "uniform", "min": 2, "max": 24},
                      "after_s": {"dist": "uniform", "min": 0.05, "max": 1.5}})
TIMEOUTS = spec(14, [weighted(CLASS_SHORT, 6), weighted(CLASS_MEDIUM, 3), weighted(CLASS_LONG, 1)],
                timeout_s={"dist": "uniform", "min": 0.5, "max": 6.0})


def long_context(seed, prompts, outputs=(128, 256)):
    return spec(seed, [{"name": "context", "prompt_tokens": {"dist": "choice", "values": prompts},
                        "output_tokens": {"dist": "uniform", "min": outputs[0], "max": outputs[1]}}])


def mean_output_tokens(workload):
    plan = build_spec_workload(workload, 500)
    return sum(item["max_tokens"] for item in plan) / len(plan)


def aggregate_tok_s(model, concurrency, execution):
    single = MODELS[model]["single_stream_tok_s"]
    if execution == "serial":
        return single
    return single * min(concurrency, MODELS[model]["slots"]) ** BATCHING_EFFICIENCY


def _minutes(expected):
    return round(expected + SETUP_MINUTES, 1)


def _cap(expected_minutes, factor=None):
    return int(min(240, max(5, math.ceil(expected_minutes * (factor or CAP_FACTOR)) + 2)))


def closed(model, execution, name, concurrency, requests, workload, *, repeats=1, warmup=4, timeout_s=120,
           abort_error_rate=0.5, slo=True, telemetry=True, description="", cap_factor=None):
    seconds = repeats * requests * mean_output_tokens(workload) / aggregate_tok_s(model, concurrency, execution)
    expected = _minutes(seconds / 60 + warmup * 0.05)
    experiment = {"id": name, "kind": "closed_loop", "description": description, "requests": requests,
                  "concurrency": concurrency, "repeats": repeats, "warmup_requests": warmup, "timeout_s": timeout_s,
                  "workload_spec": workload, "estimated_minutes": expected, "max_minutes": _cap(expected, cap_factor),
                  "abort_error_rate": abort_error_rate}
    if slo:
        experiment["slo"] = dict(MODELS[model]["slo"])
    if not telemetry:
        experiment["telemetry"] = False
    return experiment


def ladder(model, execution, name, rates, requests, workload, *, arrival="poisson", burst_size=None, timeout_s=120,
           abort_error_rate=0.2, warmup=4, description="", slo=True, concurrency=256, cap_factor=None):
    capacity = aggregate_tok_s(model, 10**6, execution)
    per_request_s = mean_output_tokens(workload) / capacity
    seconds = sum(max(requests / rate, requests * per_request_s) for rate in rates)
    expected = _minutes(seconds / 60 + warmup * 0.05)
    experiment = {"id": name, "kind": "open_loop", "description": description, "requests": requests,
                  "concurrency": concurrency, "rates": list(rates), "arrival": arrival, "repeats": 1,
                  "warmup_requests": warmup, "timeout_s": timeout_s, "workload_spec": workload,
                  "estimated_minutes": expected, "max_minutes": _cap(expected, cap_factor),
                  "abort_error_rate": abort_error_rate}
    if burst_size:
        experiment["burst_size"] = burst_size
    if slo:
        experiment["slo"] = dict(MODELS[model]["slo"])
    return experiment


SWEEP = (1, 8, 16, 32, 64, 128)
PREFILL_PACK_TOKENS = 2048    # padded tokens per packed prefill forward (8 prompts of 256, or 32 of 64)
PROD_QUEUE = 1100             # admits all 1,024 clients; the overload experiment is what exercises rejection
PROD_SWEEP = (1, 32, 64, 128, 256, 512, 1024)
PROD_REQUESTS = {   # requests per sweep point; the 7B sweep alone is 10,280 requests. 14B is scaled down but gives every client a request.
    "7b": {1: 40, 32: 1500, 64: 1500, 128: 1500, 256: 2000, 512: 2000, 1024: 2000},
    "14b": {1: 16, 32: 240, 64: 240, 128: 256, 256: 256, 512: 512, 1024: 1024}}
PROD_LADDER = {"7b": (1, 2, 3, 4, 6, 8, 12, 16, 24, 32), "14b": (0.5, 1, 2, 3, 4, 6, 8)}
PROD_LADDER_REQUESTS = {"7b": 240, "14b": 80}
PROD_CAP_FACTOR = 1.5         # production experiments are capped at 1.5x their estimate (the older stages use 2.5x)
CMP_REPEATS = 3               # the vLLM comparison needs at least three measured repetitions per key configuration
CMP_POINTS = ((1, 24), (8, 64), (32, 128))   # (clients, requests), SHORT workload: identical in serial, batched and vLLM stages
CMP_MIXED = (128, 384)        # (clients, requests), MIXED workload: batching engines only


def sweep_requests(concurrency):
    """Enough requests per point for stable percentiles where affordable; small points are labeled small-sample."""
    return {1: 60, 8: 240, 16: 320, 32: 480, 64: 640, 128: 768}[concurrency]


def sweep(model, execution, concurrencies=SWEEP, repeats=3):
    return [closed(model, execution, f"sweep-c{c}", c, sweep_requests(c), MIXED, repeats=repeats if c <= 32 else 2,
                   description=f"{c} concurrent clients, mixed prompt and output lengths, closed loop")
            for c in concurrencies]


def rate_ladder(model, execution):
    base = {"0.5b": (2, 5, 10, 20, 40, 80), "7b": (0.5, 1, 2, 4, 8, 16), "14b": (0.25, 0.5, 1, 2, 4, 8)}[model]
    return ladder(model, execution, "ladder-poisson", base, 120, MIXED,
                  description="Poisson arrivals at rising rates; aborts when more than 20% of requests fail")


STAGES = {
    "0.5b-serial": {"model": "0.5b", "engine": "tensorpilot", "execution": "serial", "quantization": "none",
                    "dtype": "float16", "max_prompt_tokens": 1024, "batch_size": 1, "queue_size": 64},
    "0.5b-batch": {"invariance": True, "model": "0.5b", "engine": "tensorpilot", "execution": "batch", "quantization": "none",
                   "dtype": "float16", "max_prompt_tokens": 1024, "batch_size": 128, "queue_size": 256},
    # The first, short GPU session: the same four workloads on the 0.5B model, serial and then batched, each experiment capped at
    # three minutes. It exists to answer "does CUDA batching run, stay correct and beat serial?" before any 7B or 14B spend.
    "0.5b-quick-serial": {"model": "0.5b", "engine": "tensorpilot", "execution": "serial", "quantization": "none",
                          "dtype": "float16", "max_prompt_tokens": 1024, "batch_size": 1, "queue_size": 64},
    "0.5b-quick-batch": {"invariance": True, "model": "0.5b", "engine": "tensorpilot", "execution": "batch",
                         "quantization": "none", "dtype": "float16", "max_prompt_tokens": 1024, "batch_size": 128,
                         "queue_size": 256},
    "7b-bf16-serial": {"model": "7b", "engine": "tensorpilot", "execution": "serial", "quantization": "none",
                       "dtype": "bfloat16", "max_prompt_tokens": 1024, "batch_size": 1, "queue_size": 64},
    "7b-bf16-batch": {"invariance": True, "model": "7b", "engine": "tensorpilot", "execution": "batch", "quantization": "none",
                      "dtype": "bfloat16", "max_prompt_tokens": 1024, "batch_size": 48, "queue_size": 192},
    "7b-bf16-long": {"model": "7b", "engine": "tensorpilot", "execution": "batch", "quantization": "none",
                     "dtype": "bfloat16", "max_prompt_tokens": 2048, "batch_size": 24, "queue_size": 64},
    "14b-nf4-batch": {"invariance": True, "model": "14b", "engine": "tensorpilot", "execution": "batch", "quantization": "nf4",
                      "dtype": "bfloat16", "max_prompt_tokens": 1024, "batch_size": 28, "queue_size": 192},
    "14b-nf4-long": {"model": "14b", "engine": "tensorpilot", "execution": "batch", "quantization": "nf4",
                     "dtype": "bfloat16", "max_prompt_tokens": 2048, "batch_size": 12, "queue_size": 64},
    "vllm-7b-bf16": {"model": "7b", "engine": "vllm", "execution": "batch", "quantization": "none",
                     "dtype": "bfloat16", "max_prompt_tokens": 1024, "batch_size": 48, "queue_size": 192},
    "vllm-14b-nf4": {"model": "14b", "engine": "vllm", "execution": "batch", "quantization": "nf4",
                     "dtype": "bfloat16", "max_prompt_tokens": 1024, "batch_size": 28, "queue_size": 192},
    # ----- production-scale evaluation (docs/v10-production-evaluation.md) -----
    # The A/B gate for packed prefill: the quick-batch workloads again with packing on, in the same Pod as 0.5b-quick-batch.
    "0.5b-quick-batch-packed": {"invariance": True, "model": "0.5b", "engine": "tensorpilot", "execution": "batch",
                                "quantization": "none", "dtype": "float16", "max_prompt_tokens": 1024, "batch_size": 128,
                                "queue_size": 256, "prefill_pack": PREFILL_PACK_TOKENS, "backlog": 2048},
    # Queue 1,100 admits every one of 1,024 clients, so the sweep measures latency and throughput at that concurrency rather
    # than how fast clients are turned away; the bounded-queue behaviour is the separate `overload` experiment.
    "7b-bf16-prod-serial": {"model": "7b", "engine": "tensorpilot", "execution": "serial", "quantization": "none",
                            "dtype": "bfloat16", "max_prompt_tokens": 1024, "batch_size": 1, "queue_size": 64},
    "7b-bf16-prod": {"invariance": True, "model": "7b", "engine": "tensorpilot", "execution": "batch",
                     "quantization": "none", "dtype": "bfloat16", "max_prompt_tokens": 1024, "batch_size": 48,
                     "queue_size": PROD_QUEUE, "prefill_pack": PREFILL_PACK_TOKENS, "backlog": 2048},
    "14b-nf4-prod": {"invariance": True, "model": "14b", "engine": "tensorpilot", "execution": "batch",
                     "quantization": "nf4", "dtype": "bfloat16", "max_prompt_tokens": 1024, "batch_size": 28,
                     "queue_size": PROD_QUEUE, "prefill_pack": PREFILL_PACK_TOKENS, "backlog": 2048},
    "vllm-7b-bf16-prod": {"model": "7b", "engine": "vllm", "execution": "batch", "quantization": "none",
                          "dtype": "bfloat16", "max_prompt_tokens": 1024, "batch_size": 48, "queue_size": PROD_QUEUE},
    "vllm-14b-nf4-prod": {"model": "14b", "engine": "vllm", "execution": "batch", "quantization": "nf4",
                          "dtype": "bfloat16", "max_prompt_tokens": 1024, "batch_size": 28, "queue_size": PROD_QUEUE},
    # The lean vLLM comparison, for re-running only what a failed vLLM stage lost (docs/v10-7b-results.md). Same model, settings and
    # experiment definitions as the -prod stages, restricted to the comparison points and the saturation plateau.
    "7b-bf16-cmp": {"model": "7b", "engine": "tensorpilot", "execution": "batch", "quantization": "none",
                    "dtype": "bfloat16", "max_prompt_tokens": 1024, "batch_size": 48, "queue_size": PROD_QUEUE,
                    "prefill_pack": PREFILL_PACK_TOKENS, "backlog": 2048},
    "vllm-7b-bf16-cmp": {"model": "7b", "engine": "vllm", "execution": "batch", "quantization": "none",
                         "dtype": "bfloat16", "max_prompt_tokens": 1024, "batch_size": 48, "queue_size": PROD_QUEUE},
}
# Lean stages: the -prod stage they are cut from, and the experiment ids they keep (definitions are copied unchanged, so the
# workload hashes match the archived -prod runs). TensorPilot's control repeats only the comparison points, on the vLLM Pod's card.
LEAN_STAGES = {
    "7b-bf16-cmp": ("7b-bf16-prod", ("cold-start", "cmp-c1", "cmp-c8", "cmp-c32", "cmp-c128")),
    "vllm-7b-bf16-cmp": ("vllm-7b-bf16-prod", ("cold-start", "cmp-c1", "cmp-c8", "cmp-c32", "cmp-c128",
                                               "scale-c64", "scale-c256", "scale-c1024")),
}


# Stages that are not gpu_bench experiment files but run under the same cap, ledger and evidence rules.
EXTRA_STAGES = {
    "quality-7b": {"model": "7b", "engine": "quality", "quantization": "none", "dtype": "bfloat16",
                   "expected_minutes": 14, "cap_minutes": 30,
                   "what": "7B bf16 reference capture, then NF4 capture against it, then the comparison"},
    "quality-14b": {"model": "14b", "engine": "quality", "quantization": "nf4", "dtype": "bfloat16",
                    "expected_minutes": 12, "cap_minutes": 30,
                    "what": "14B NF4 capture: absolute metrics only, there is no bf16 reference on 24 GB"},
    "faults-0.5b": {"model": "0.5b", "engine": "faults", "quantization": "none", "dtype": "float16",
                    "expected_minutes": 5, "cap_minutes": 20, "rate": 10, "requests": 300, "kill_after": 12,
                    "what": "SIGKILL the worker under load and audit request accounting (fast restart)"},
    "faults-7b": {"model": "7b", "engine": "faults", "quantization": "none", "dtype": "bfloat16",
                  "expected_minutes": 12, "cap_minutes": 30, "rate": 5, "requests": 900, "kill_after": 25,
                  "what": "the same crash experiment with a 7B reload in the recovery path"},
    "probe-0.5b": {"model": "0.5b", "engine": "probe", "quantization": "none", "dtype": "float16", "prefill_pack": 2048,
                   "expected_minutes": 3, "cap_minutes": 12,
                   "what": "explain solo-vs-batched divergence with the model's own top-2 logits (the 31/32 quick-session case)"},
    "probe-7b": {"model": "7b", "engine": "probe", "quantization": "none", "dtype": "bfloat16", "prefill_pack": 2048,
                 "expected_minutes": 6, "cap_minutes": 20,
                 "what": "the same logit-margin explanation for bf16 7B: is every divergence a rounding-level near-tie?"},
}
# The batch-invariance check (tensorpilot.batch_invariance) runs before the benchmark in the batching stages.
INVARIANCE_CAP_MINUTES = 10
INVARIANCE_EXPECTED_MINUTES = 3
READY_TIMEOUT_S = {"0.5b": 900, "7b": 1800, "14b": 3600}
DOWNLOAD_GIB = {"0.5b": 0.92, "7b": 14.19, "14b": 27.51}
SPECS = {"short": SHORT, "mixed": MIXED, "cancel": CANCEL, "timeouts": TIMEOUTS}


# ----- the budget model ------------------------------------------------------------------------------------------
# The production evaluation runs in three Pods. Each session's SESSION_BUDGET_MINUTES is a ledger bound on stage time
# (a stage is refused if the time already used plus its hard cap exceeds it); boot, downloads, model loads and export are
# outside the ledger, so they are added as an assumed overhead plus slack. Price is an ASSUMPTION until you rent the card.
BUDGET_USD = 9.0
ASSUMED_PRICE_PER_HOUR = 0.90
SLACK_MINUTES = 15
# A stage's ledger entry is its WHOLE wall time, not just its experiments: preflight (which loads the model once), the worker's own
# load, the smoke test, the memory gate and the evidence export all count, and so does the model download in a session's first stage.
# These per-stage start-up figures and the download rate are ASSUMPTIONS (7B loads from a warm cache in roughly two minutes each).
STAGE_STARTUP_MINUTES = {"tensorpilot": 6, "vllm": 7, "quality": 2, "faults": 3, "probe": 2}
DOWNLOAD_MB_PER_S = 30
PATH_MARGIN = 1.15   # time used before a stage may run 15% over its expected total before the ledger would refuse the next stage
SESSIONS = {
    "gate": {"stages": ["0.5b-quick-batch", "0.5b-quick-batch-packed", "probe-0.5b"], "overhead_minutes": 15,
             "purpose": "does packed prefill help or hurt, and is every solo-vs-batched divergence a rounding-level near-tie?"},
    # probe-0.5b runs first with PREFILL_PACK_TOKENS=0: the unpacked HTTP path is deterministic and equals the probe's insert-all-then-step
    # order, so it reproduces (and explains with logits) the quick-session divergence that the packed gate-session probe did not.
    "7b": {"stages": ["probe-7b", "7b-bf16-prod-serial", "7b-bf16-prod", "faults-7b", "vllm-7b-bf16-prod", "quality-7b"],
           "overhead_minutes": 40, "purpose": "7B bf16: serial baseline, scale sweep, reliability, crash recovery, vLLM"},
    "14b": {"stages": ["14b-nf4-prod", "quality-14b"], "overhead_minutes": 25,
            "purpose": "14B NF4: load, generate, scale sweep, fidelity"},
}
# Re-running only what the 7B session lost: the vLLM comparison (its stage failed to start) and the NF4 quality evaluation (skipped).
# The TensorPilot control repeats the comparison points on the same card as vLLM, so the two engines share a physical GPU and a Pod.
RERUN_SESSIONS = {
    "7b-vllm-rerun": {"stages": ["vllm-7b-bf16-cmp", "7b-bf16-cmp", "quality-7b"], "overhead_minutes": 35,
                      "purpose": "vLLM against TensorPilot on one card at the comparison points, and the 7B NF4 quality evaluation"},
}
# Not counted against the $9: only if credits remain after the core sessions. A fresh Pod re-downloads 28 GB and reinstalls vLLM.
OPTIONAL_SESSIONS = {
    "14b-vllm": {"stages": ["vllm-14b-nf4-prod"], "overhead_minutes": 20,
                 "purpose": "vLLM on the same 14B NF4 workloads (a different Pod from the TensorPilot 14B run)"},
}


def _stage_minutes(stage):
    env = stage_env(stage)
    return env["V10_EXPECTED_MINUTES"], env["V10_CAP_MINUTES"]


def _ledger_minutes(stage, first_in_session):
    """Expected ledger time for a stage: its experiments plus assumed start-up, plus the download if it is the first stage."""
    env = stage_env(stage)
    minutes = env["V10_EXPECTED_MINUTES"] + STAGE_STARTUP_MINUTES[env["V10_ENGINE"] if env["V10_ENGINE"] in STAGE_STARTUP_MINUTES
                                                                   else "tensorpilot"]
    if first_in_session:
        minutes += DOWNLOAD_GIB[env["V10_MODEL_KEY"]] * 1024 / DOWNLOAD_MB_PER_S / 60
    return minutes


def budget_plan(price_per_hour=ASSUMED_PRICE_PER_HOUR, budget_usd=BUDGET_USD, sessions=None):
    """Expected and worst-case cost per session, the SESSION_BUDGET_MINUTES that enforces it, and whether it fits.

    Worst case for a session = its ledger budget + the assumed overhead outside the ledger (boot, vLLM install, idle) + slack. The
    ledger counts each stage's whole wall time, and refuses a stage whose cap would take the used time past the budget. The wanted
    budget is therefore the largest, over the stages, of (expected time used before it, with a 15% margin) + its cap; budgets are
    scaled down together if the sum of worst cases would exceed the money.
    """
    sessions = SESSIONS if sessions is None else sessions
    per_minute = price_per_hour / 60
    allowed_minutes = budget_usd / per_minute
    rows = {}
    for name, session in sessions.items():
        mins = [_stage_minutes(stage) for stage in session["stages"]]
        expected = sum(m[0] for m in mins)
        caps = [m[1] for m in mins]
        ledger = [_ledger_minutes(stage, i == 0) for i, stage in enumerate(session["stages"])]
        rows[name] = {"stages": list(session["stages"]), "expected_stage_minutes": round(expected, 1),
                      "stage_caps": caps, "overhead_minutes": session["overhead_minutes"], "slack_minutes": SLACK_MINUTES,
                      "expected_ledger_minutes": round(sum(ledger), 1),
                      "wanted_budget_minutes": math.ceil(max(PATH_MARGIN * sum(ledger[:i]) + caps[i] for i in range(len(caps)))),
                      "purpose": session["purpose"]}
    wanted_worst = sum(r["wanted_budget_minutes"] + r["overhead_minutes"] + r["slack_minutes"] for r in rows.values())
    # Scale the stage-time budgets so that the sum of worst cases is the money, if the wanted budgets would overspend.
    free = allowed_minutes - sum(r["overhead_minutes"] + r["slack_minutes"] for r in rows.values())
    wanted_stage = sum(r["wanted_budget_minutes"] for r in rows.values())
    scale = min(1.0, free / wanted_stage) if free > 0 else 0.0
    total_expected = total_worst = 0.0
    for r in rows.values():
        r["session_budget_minutes"] = math.floor(r["wanted_budget_minutes"] * scale)
        used, refused = 0.0, None
        for i, (stage, cap) in enumerate(zip(r["stages"], r["stage_caps"])):
            minutes = _ledger_minutes(stage, i == 0)
            if PATH_MARGIN * used + cap > r["session_budget_minutes"]:
                refused = stage
                break
            used += minutes
        r["ledger_refuses_on_the_expected_path"] = refused
        r["expected_total_minutes"] = round(r["expected_ledger_minutes"] + r["overhead_minutes"], 1)
        r["worst_total_minutes"] = r["session_budget_minutes"] + r["overhead_minutes"] + r["slack_minutes"]
        r["expected_cost_usd"] = round(r["expected_total_minutes"] * per_minute, 2)
        r["worst_cost_usd"] = round(r["worst_total_minutes"] * per_minute, 2)
        total_expected += r["expected_cost_usd"]
        total_worst += r["worst_cost_usd"]
    return {"price_per_hour_usd": price_per_hour, "price_is_assumption": True, "budget_usd": budget_usd,
            "wanted_worst_case_usd": round(wanted_worst * per_minute, 2), "scaled_to_fit": scale < 1.0,
            "sessions": rows, "optional_sessions": {name: {"stages": o["stages"], "purpose": o["purpose"],
                                                            "expected_stage_minutes": round(sum(_stage_minutes(x)[0] for x in o["stages"]), 1),
                                                            "worst_cost_usd": round((sum(_stage_minutes(x)[1] for x in o["stages"])
                                                                                     + o["overhead_minutes"] + SLACK_MINUTES) * per_minute, 2)}
                                                    for name, o in OPTIONAL_SESSIONS.items()},
            "expected_total_usd": round(total_expected, 2), "worst_total_usd": round(total_worst, 2),
            "fits_budget_worst_case": total_worst <= budget_usd + 1e-6,
            "every_expected_stage_can_start": all(r["ledger_refuses_on_the_expected_path"] is None for r in rows.values())}


def all_stage_names():
    return list(STAGES) + list(EXTRA_STAGES)


def stage_env(stage):
    """Everything the stage runner needs, from this one definition, as {NAME: value}."""
    if stage in STAGES:
        s, config = STAGES[stage], config_for(stage)
        env = {"V10_ENGINE": s["engine"], "V10_EXECUTION": s["execution"], "V10_QUANT": s["quantization"],
               "V10_DTYPE": s["dtype"], "V10_BATCH_SIZE": s["batch_size"], "V10_QUEUE_SIZE": s["queue_size"],
               "V10_MAX_PROMPT_TOKENS": s["max_prompt_tokens"], "V10_CONFIG": f"benchmarks/gpu/v10-{stage}.json",
               "V10_CAP_MINUTES": config["max_session_minutes"] - 5 + (INVARIANCE_CAP_MINUTES if s.get("invariance") else 0),
               "V10_EXPECTED_MINUTES": round(config["expected_experiment_minutes"]
                                             + (INVARIANCE_EXPECTED_MINUTES if s.get("invariance") else 0), 1),
               "V10_INVARIANCE": 1 if s.get("invariance") else 0,
               "V10_PREFILL_PACK": s.get("prefill_pack", 0), "V10_BACKLOG": s.get("backlog", 128)}
    elif stage in EXTRA_STAGES:
        s = EXTRA_STAGES[stage]
        env = {"V10_ENGINE": s["engine"], "V10_EXECUTION": "batch", "V10_QUANT": s["quantization"],
               "V10_DTYPE": s["dtype"], "V10_BATCH_SIZE": 32, "V10_QUEUE_SIZE": 128, "V10_MAX_PROMPT_TOKENS": 1024,
               "V10_CONFIG": "", "V10_CAP_MINUTES": s["cap_minutes"], "V10_EXPECTED_MINUTES": s["expected_minutes"],
               "V10_WHAT": s["what"]}
        env["V10_PREFILL_PACK"] = s.get("prefill_pack", 0)
        env["V10_BACKLOG"] = 128
        for key in ("rate", "requests", "kill_after"):
            if key in s:
                env[f"V10_FAULT_{key.upper()}"] = s[key]
    else:
        raise KeyError(stage)
    model = MODELS[s["model"]]
    env.update(V10_STAGE=stage, V10_MODEL_KEY=s["model"], V10_MODEL_REPO=model["repo"],
               V10_REVISION=model["revision"], V10_READY_TIMEOUT_S=READY_TIMEOUT_S[s["model"]],
               V10_DOWNLOAD_GIB=DOWNLOAD_GIB[s["model"]])
    return env


QUICK_EXPERIMENT_CAP_MINUTES = 3


def quick_experiments(stage, cold):
    """quick-c1 and quick-c8 are identical in the serial and batched stages, so the two can be compared directly."""
    model, execution = STAGES[stage]["model"], STAGES[stage]["execution"]
    items = [cold, closed(model, execution, "quick-c1", 1, 24, SHORT, warmup=2,
                          description="one client: batching must not slow a single stream"),
             closed(model, execution, "quick-c8", 8, 64, SHORT, warmup=2,
                    description="eight clients: queued behind one sequence (serial) or decoded together (batch)")]
    if execution == "batch":
        items += [closed(model, execution, "quick-c32", 32, 128, SHORT, warmup=2,
                         description="thirty-two clients decoding together"),
                  closed(model, execution, "quick-c128", 128, 256, SHORT, warmup=2,
                         description="as many clients as the pool has slots: every slot busy at once"),
                  closed(model, execution, "quick-cancel", 32, 96, CANCEL, warmup=2, abort_error_rate=0.5,
                         description="30% of clients abandon their request: cleanup of a shared batch")]
    for item in items:
        item["max_minutes"] = min(item["max_minutes"], QUICK_EXPERIMENT_CAP_MINUTES)
    return items


def prod_experiments(stage, cold):
    """The production-scale experiments. Ids are shared across engines so the report and the acceptance checks can pair them.

    The 7B stages carry the full set (sweep to 1,024 clients, ladder, sustained run, cancellation, timeouts, overload). The 14B
    stage is a leaner set of the same measurements: the safety experiments exercise engine code that does not depend on the
    model, so repeating them at 14B would spend the budget on the same code path.
    """
    s = STAGES[stage]
    model, execution, engine = s["model"], s["execution"], s["engine"]
    lean = model == "14b"
    cap = PROD_CAP_FACTOR
    items = [cold]
    for clients, requests in CMP_POINTS:
        items.append(closed(model, execution, f"cmp-c{clients}", clients, requests, SHORT, repeats=CMP_REPEATS, warmup=2,
                            cap_factor=cap,
                            description=f"comparison point: {clients} clients, short prompts, identical in every engine's stage"))
    if execution == "serial":
        return items
    clients, requests = CMP_MIXED
    items.append(closed(model, execution, f"cmp-c{clients}", clients, requests, MIXED, repeats=CMP_REPEATS, warmup=2,
                        cap_factor=cap,
                        description=f"comparison point: {clients} clients, mixed lengths, identical in every batching stage"))
    counts = PROD_REQUESTS[model]
    for clients in PROD_SWEEP:
        items.append(closed(model, execution, f"scale-c{clients}", clients, counts[clients], MIXED,
                            repeats=2 if clients == 1 else 1, abort_error_rate=0.9, cap_factor=cap,
                            description=f"{clients} concurrent clients, mixed lengths, {counts[clients]} requests: scale sweep"))
    items.append(ladder(model, execution, "ladder-poisson", PROD_LADDER[model], PROD_LADDER_REQUESTS[model], MIXED,
                        concurrency=1024, abort_error_rate=0.5, cap_factor=cap,
                        description="Poisson arrivals at rising rates until saturation: the maximum sustainable rate"))
    items.append(closed(model, execution, "sustained-c32", 32, 800 if lean else 3000, MIXED, abort_error_rate=0.2,
                        cap_factor=cap,
                        description="a long steady run at a concurrency the slots can serve: sustained-load reliability"))
    if engine == "tensorpilot" and not lean:
        items.append(closed(model, execution, "cancellation", 32, 320, CANCEL, abort_error_rate=0.5, cap_factor=cap,
                            description="30% of clients abandon their request, while queued or while decoding"))
        items.append(closed(model, execution, "timeouts", 32, 240, TIMEOUTS, abort_error_rate=1.0, cap_factor=cap,
                            description="tight per-request deadlines: expiry, cleanup and accounting under pressure"))
        items.append(ladder(model, execution, "overload", (400,), 1500, SHORT, abort_error_rate=1.0, slo=False,
                            timeout_s=30, concurrency=1024, cap_factor=cap,
                            description="arrivals far above capacity: a bounded queue, rejections, and recovery"))
        items.append(closed(model, execution, "recovery-after-overload", 8, 64, SHORT, slo=False, warmup=0, cap_factor=cap,
                            description="the same worker right after overload: it must serve normally again"))
    return items


def experiments_for(stage):
    s = STAGES[stage]
    model, execution = s["model"], s["execution"]
    cold = closed(model, execution, "cold-start", 1, 8, SHORT, warmup=0, slo=False, abort_error_rate=0.5,
                  description="the first requests after readiness, before any warmup request: cold-start latency")
    if "-quick-" in stage and not stage.endswith("-packed"):
        return quick_experiments(stage, cold)
    if stage == "0.5b-quick-batch-packed":
        return quick_experiments("0.5b-quick-batch", cold)
    if stage.endswith("-prod") or stage.endswith("-prod-serial"):
        return prod_experiments(stage, cold)
    if stage in LEAN_STAGES:
        parent, keep = LEAN_STAGES[stage]
        return [e for e in prod_experiments(parent, cold) if e["id"] in keep]
    if stage == "0.5b-serial":
        legacy = {"prompt_tokens": 17, "output_tokens": 32}
        regression = spec(10, [{"name": "regression", "prompt_tokens": legacy["prompt_tokens"],
                                "output_tokens": legacy["output_tokens"]}])
        return [cold, closed(model, execution, "regression-c1", 1, 40, regression, repeats=3,
                             description="the v0.9.1-4 baseline workload shape, to confirm no regression"),
                closed(model, execution, "serial-c8", 8, 80, SHORT, repeats=2,
                       description="eight clients against the serial worker: queueing, not batching")]
    if stage == "7b-bf16-serial":
        return [cold, closed(model, execution, "serial-c1", 1, 40, SHORT, repeats=2),
                closed(model, execution, "serial-c8", 8, 64, SHORT, repeats=2,
                       description="eight clients against the serial worker: the no-batching reference")]
    if stage.endswith("-long"):
        # 2048 tokens is the longest prompt the service accepts through the word-built prompt generator
        # (16384-byte prompt limit); longer contexts would need a different prompt source and server limit.
        prompts = [1024, 2048]
        workload = long_context(15, prompts)
        points = (1, 4, 8) if "14b" in stage else (1, 4, 8, 16)
        return [cold] + [closed(model, execution, f"long-c{c}", c, max(24, 8 * c), workload, repeats=2, warmup=2,
                                description=f"{c} clients with {prompts[0]}-{prompts[-1]} token prompts")
                         for c in points]
    items = [cold] + sweep(model, execution)
    items.append(rate_ladder(model, execution))
    items.append(ladder(model, execution, "bursty", (MODELS[model]["slots"] // 8 or 1,), 160, MIXED, arrival="burst",
                        burst_size=8, abort_error_rate=0.5, description="bursts of eight arrivals at a fixed mean rate"))
    if s["engine"] == "tensorpilot":
        items.append(closed(model, execution, "cancellation", 32, 320, CANCEL, repeats=1, abort_error_rate=0.5,
                            description="30% of clients abandon their request, while queued or while decoding"))
        items.append(closed(model, execution, "timeouts", 32, 240, TIMEOUTS, repeats=1, abort_error_rate=1.0,
                            description="tight per-request deadlines: expiry, cleanup and accounting under pressure"))
        items.append(ladder(model, execution, "overload", (400,), 600, SHORT, abort_error_rate=1.0, slo=False,
                            timeout_s=30, description="arrivals far above capacity: admission rejection and recovery"))
        items.append(closed(model, execution, "recovery-after-overload", 8, 64, SHORT, slo=False, warmup=0,
                            description="the same worker right after overload: it must serve normally again"))
        if stage in ("7b-bf16-batch", "0.5b-batch"):
            for flag in (True, False):
                items.append(closed(model, execution, f"overhead-telemetry-{'on' if flag else 'off'}", 32, 320, SHORT,
                                    repeats=3, telemetry=flag, slo=False, warmup=4,
                                    description="identical workload with the telemetry sampler "
                                                f"{'on' if flag else 'off'}, to measure its effect on throughput"))
    return items


def config_for(stage):
    s = STAGES[stage]
    model = MODELS[s["model"]]
    experiments = experiments_for(stage)
    expected = round(sum(e["estimated_minutes"] for e in experiments), 1)
    cap = int(sum(e["max_minutes"] for e in experiments))
    config = {"schema": SCHEMA, "suite": f"v1.0-{stage}",
              "target": {"model": model["repo"], "revision": model["revision"], "license": "Apache-2.0",
                         "prompt_format": "raw text, tokenizer default special tokens (no chat template)",
                         "engine": s["engine"], "execution": s["execution"], "quantization": s["quantization"],
                         "dtype": s["dtype"], "max_prompt_tokens": s["max_prompt_tokens"]},
              "assumptions": {"status": "placeholder_unmeasured",
                              "decode_tokens_per_s": model["single_stream_tok_s"], "prefill_tokens_per_s": 2000,
                              "single_stream_source": model["single_stream_source"],
                              "batching_efficiency_exponent": BATCHING_EFFICIENCY,
                              "note": "planning numbers only; the experiment caps, not these estimates, bound cost"},
              "session_overhead_minutes": 5, "max_session_minutes": cap + 5,
              "expected_experiment_minutes": expected,
              "telemetry": {"interval_s": 1.0}, "experiments": experiments}
    if s["engine"] == "vllm":
        config["api"], config["model_name"] = "openai", model["repo"]
        config["target"]["quantization_note"] = ("bitsandbytes NF4 loaded by vLLM: the same quantization format as "
                                                 "TensorPilot's, through a different implementation"
                                                 if s["quantization"] == "nf4" else None)
    return config


def stage_names():
    return list(STAGES)


def render(config):
    return json.dumps(config, indent=2) + "\n"


def plan_text():
    lines = [f"{'stage':<16}{'experiments':>12}{'expected min':>14}{'hard cap min':>14}"]
    total_expected = total_cap = 0
    for stage in all_stage_names():
        env = stage_env(stage)
        count = len(config_for(stage)["experiments"]) if stage in STAGES else "-"
        total_expected += env["V10_EXPECTED_MINUTES"]
        total_cap += env["V10_CAP_MINUTES"]
        lines.append(f"{stage:<16}{count:>12}{env['V10_EXPECTED_MINUTES']:>14}{env['V10_CAP_MINUTES']:>14}")
    lines.append(f"{'TOTAL':<16}{'':>12}{round(total_expected, 1):>14}{total_cap:>14}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", metavar="DIR", help="write benchmarks/gpu/v10-<stage>.json for every stage")
    parser.add_argument("--plan", action="store_true", help="print expected and capped experiment minutes")
    parser.add_argument("--budget", action="store_true", help="print the session budgets and costs (JSON)")
    parser.add_argument("--rerun-budget", action="store_true", help="print the budget of the vLLM/quality re-run session (JSON)")
    parser.add_argument("--price-per-hour", type=float, default=ASSUMED_PRICE_PER_HOUR, help="your Pod's real hourly price")
    parser.add_argument("--budget-usd", type=float, default=BUDGET_USD)
    parser.add_argument("--list-stages", action="store_true", help="print stage names, one per line")
    parser.add_argument("--stage-env", metavar="STAGE", help="print the stage's settings as shell exports")
    parser.add_argument("--dump-spec", metavar="NAME", choices=sorted(SPECS), help="print a named workload spec as JSON")
    args = parser.parse_args()
    if args.budget:
        print(json.dumps(budget_plan(args.price_per_hour, args.budget_usd), indent=2))
        return 0
    if args.rerun_budget:
        print(json.dumps(budget_plan(args.price_per_hour, args.budget_usd, RERUN_SESSIONS), indent=2))
        return 0
    if args.list_stages:
        print("\n".join(all_stage_names()))
        return 0
    if args.stage_env:
        try:
            env = stage_env(args.stage_env)
        except KeyError:
            print(f"unknown stage {args.stage_env!r}; choose from: {', '.join(all_stage_names())}", file=sys.stderr)
            return 2
        for key, value in env.items():
            print(f"export {key}={shlex.quote(str(value))}")
        return 0
    if args.dump_spec:
        print(json.dumps(SPECS[args.dump_spec], indent=2))
        return 0
    if args.write:
        directory = Path(args.write)
        directory.mkdir(parents=True, exist_ok=True)
        for stage in STAGES:
            (directory / f"v10-{stage}.json").write_text(render(config_for(stage)))
        print(f"wrote {len(STAGES)} configs to {directory}")
    if args.plan or not args.write:
        print(plan_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
