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
           "slots": 64, "slo": {"ttft_s": 2.0, "e2e_s": 20.0, "tpot_s": 0.15}},
    "14b": {"repo": "Qwen/Qwen2.5-14B-Instruct", "revision": "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8",
            "single_stream_tok_s": 25, "single_stream_source": "ASSUMPTION: NF4 dequantization overhead, unmeasured",
            "slots": 34, "slo": {"ttft_s": 4.0, "e2e_s": 40.0, "tpot_s": 0.35}},
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


def _cap(expected_minutes):
    return int(min(240, max(5, math.ceil(expected_minutes * CAP_FACTOR) + 2)))


def closed(model, execution, name, concurrency, requests, workload, *, repeats=1, warmup=4, timeout_s=120,
           abort_error_rate=0.5, slo=True, telemetry=True, description=""):
    seconds = repeats * requests * mean_output_tokens(workload) / aggregate_tok_s(model, concurrency, execution)
    expected = _minutes(seconds / 60 + warmup * 0.05)
    experiment = {"id": name, "kind": "closed_loop", "description": description, "requests": requests,
                  "concurrency": concurrency, "repeats": repeats, "warmup_requests": warmup, "timeout_s": timeout_s,
                  "workload_spec": workload, "estimated_minutes": expected, "max_minutes": _cap(expected),
                  "abort_error_rate": abort_error_rate}
    if slo:
        experiment["slo"] = dict(MODELS[model]["slo"])
    if not telemetry:
        experiment["telemetry"] = False
    return experiment


def ladder(model, execution, name, rates, requests, workload, *, arrival="poisson", burst_size=None, timeout_s=120,
           abort_error_rate=0.2, warmup=4, description="", slo=True):
    capacity = aggregate_tok_s(model, 10**6, execution)
    per_request_s = mean_output_tokens(workload) / capacity
    seconds = sum(max(requests / rate, requests * per_request_s) for rate in rates)
    expected = _minutes(seconds / 60 + warmup * 0.05)
    experiment = {"id": name, "kind": "open_loop", "description": description, "requests": requests,
                  "concurrency": 256, "rates": list(rates), "arrival": arrival, "repeats": 1,
                  "warmup_requests": warmup, "timeout_s": timeout_s, "workload_spec": workload,
                  "estimated_minutes": expected, "max_minutes": _cap(expected), "abort_error_rate": abort_error_rate}
    if burst_size:
        experiment["burst_size"] = burst_size
    if slo:
        experiment["slo"] = dict(MODELS[model]["slo"])
    return experiment


SWEEP = (1, 8, 16, 32, 64, 128)


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
    "7b-bf16-serial": {"model": "7b", "engine": "tensorpilot", "execution": "serial", "quantization": "none",
                       "dtype": "bfloat16", "max_prompt_tokens": 1024, "batch_size": 1, "queue_size": 64},
    "7b-bf16-batch": {"invariance": True, "model": "7b", "engine": "tensorpilot", "execution": "batch", "quantization": "none",
                      "dtype": "bfloat16", "max_prompt_tokens": 1024, "batch_size": 64, "queue_size": 192},
    "7b-bf16-long": {"model": "7b", "engine": "tensorpilot", "execution": "batch", "quantization": "none",
                     "dtype": "bfloat16", "max_prompt_tokens": 2048, "batch_size": 32, "queue_size": 64},
    "14b-nf4-batch": {"invariance": True, "model": "14b", "engine": "tensorpilot", "execution": "batch", "quantization": "nf4",
                      "dtype": "bfloat16", "max_prompt_tokens": 1024, "batch_size": 64, "queue_size": 192},
    "14b-nf4-long": {"model": "14b", "engine": "tensorpilot", "execution": "batch", "quantization": "nf4",
                     "dtype": "bfloat16", "max_prompt_tokens": 2048, "batch_size": 16, "queue_size": 64},
    "vllm-7b-bf16": {"model": "7b", "engine": "vllm", "execution": "batch", "quantization": "none",
                     "dtype": "bfloat16", "max_prompt_tokens": 1024, "batch_size": 64, "queue_size": 192},
    "vllm-14b-nf4": {"model": "14b", "engine": "vllm", "execution": "batch", "quantization": "nf4",
                     "dtype": "bfloat16", "max_prompt_tokens": 1024, "batch_size": 64, "queue_size": 192},
}


# Stages that are not gpu_bench experiment files but run under the same cap, ledger and evidence rules.
EXTRA_STAGES = {
    "quality-7b": {"model": "7b", "engine": "quality", "quantization": "none", "dtype": "bfloat16",
                   "expected_minutes": 14, "cap_minutes": 40,
                   "what": "7B bf16 reference capture, then NF4 capture against it, then the comparison"},
    "quality-14b": {"model": "14b", "engine": "quality", "quantization": "nf4", "dtype": "bfloat16",
                    "expected_minutes": 12, "cap_minutes": 45,
                    "what": "14B NF4 capture: absolute metrics only, there is no bf16 reference on 24 GB"},
    "faults-0.5b": {"model": "0.5b", "engine": "faults", "quantization": "none", "dtype": "float16",
                    "expected_minutes": 5, "cap_minutes": 20, "rate": 10, "requests": 300, "kill_after": 12,
                    "what": "SIGKILL the worker under load and audit request accounting (fast restart)"},
    "faults-7b": {"model": "7b", "engine": "faults", "quantization": "none", "dtype": "bfloat16",
                  "expected_minutes": 12, "cap_minutes": 50, "rate": 5, "requests": 900, "kill_after": 25,
                  "what": "the same crash experiment with a 7B reload in the recovery path"},
}
# The batch-invariance check (tensorpilot.batch_invariance) runs before the benchmark in the batching stages.
INVARIANCE_CAP_MINUTES = 10
INVARIANCE_EXPECTED_MINUTES = 3
READY_TIMEOUT_S = {"0.5b": 900, "7b": 1800, "14b": 3600}
DOWNLOAD_GIB = {"0.5b": 0.92, "7b": 14.19, "14b": 27.51}
SPECS = {"short": SHORT, "mixed": MIXED, "cancel": CANCEL, "timeouts": TIMEOUTS}


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
               "V10_INVARIANCE": 1 if s.get("invariance") else 0}
    elif stage in EXTRA_STAGES:
        s = EXTRA_STAGES[stage]
        env = {"V10_ENGINE": s["engine"], "V10_EXECUTION": "batch", "V10_QUANT": s["quantization"],
               "V10_DTYPE": s["dtype"], "V10_BATCH_SIZE": 32, "V10_QUEUE_SIZE": 128, "V10_MAX_PROMPT_TOKENS": 1024,
               "V10_CONFIG": "", "V10_CAP_MINUTES": s["cap_minutes"], "V10_EXPECTED_MINUTES": s["expected_minutes"],
               "V10_WHAT": s["what"]}
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


def experiments_for(stage):
    s = STAGES[stage]
    model, execution = s["model"], s["execution"]
    cold = closed(model, execution, "cold-start", 1, 8, SHORT, warmup=0, slo=False, abort_error_rate=0.5,
                  description="the first requests after readiness, before any warmup request: cold-start latency")
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
    parser.add_argument("--list-stages", action="store_true", help="print stage names, one per line")
    parser.add_argument("--stage-env", metavar="STAGE", help="print the stage's settings as shell exports")
    parser.add_argument("--dump-spec", metavar="NAME", choices=sorted(SPECS), help="print a named workload spec as JSON")
    args = parser.parse_args()
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
