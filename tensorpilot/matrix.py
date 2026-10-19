"""Generate bounded, reproducible v0.6 GPU runtime experiment configurations."""
import argparse
from datetime import datetime, timezone
import itertools
import json

from .reporting import write_json


def estimated_kv_bytes(input_tokens, output_tokens, width, *, bytes_per_token=16 * 1024,
                       sequence_overhead_bytes=2 * 1024**2):
    return width * (sequence_overhead_bytes + (input_tokens + output_tokens) * bytes_per_token)


def build_matrix(*, input_tokens=(128, 512, 2048), output_tokens=(128, 256, 512),
                 widths=(1, 2, 4, 8), arrival_rates=(0, 4), budget_bytes=512 * 1024**2,
                 safety_fraction=.8):
    if not 0 < safety_fraction <= 1:
        raise ValueError("safety_fraction must be in (0, 1]")
    cases = []
    for input_target, output_target, width, rate in itertools.product(
            input_tokens, output_tokens, widths, arrival_rates):
        estimate = estimated_kv_bytes(input_target, output_target, width)
        safe = estimate <= budget_bytes * safety_fraction
        common = {"input_target_tokens": input_target, "output_tokens": output_target,
                  "max_batch_width": width, "arrival_rate_rps": rate,
                  "estimated_kv_bytes": estimate,
                  "estimate_kind": "conservative_preflight_not_measured_memory",
                  "preflight_safe": safe}
        cases.append({**common, "execution": "batch",
                      "id": f"fixed-w{width}-in{input_target}-out{output_target}-r{rate}"})
        cases.append({**common, "execution": "adaptive",
                      "id": f"adaptive-w{width}-in{input_target}-out{output_target}-r{rate}"})
    return {"schema_version": 1, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "measurement_contract": {
                "input_lengths": "targets; actual tokenizer counts must come from each done event",
                "output_lengths": "caps; actual generated counts and finish reasons must be recorded",
                "memory": "preflight values are estimates; MLX allocator values are measured separately",
                "repetitions": 3,
                "warmup": "one excluded request per workload class and runtime instance",
            },
            "budget_bytes": budget_bytes, "safety_fraction": safety_fraction,
            "cases": cases}


def workload_for_case(case, requests=30):
    if type(requests) is not int or requests < 1:
        raise ValueError("requests must be positive")
    # The stable repeated fragment produces a close target with the current
    # tokenizer. Actual prompt_tokens in the server terminal event is authoritative.
    repeats = max(1, case["input_target_tokens"] - 16)
    prompt = "Analyze bounded GPU inference behavior: " + " data" * repeats
    return [{"prompt": prompt, "max_tokens": case["output_tokens"],
             "class": f"target-in{case['input_target_tokens']}-out{case['output_tokens']}"}
            for _ in range(requests)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/v06_matrix.json")
    parser.add_argument("--budget-mib", type=int, default=512)
    parser.add_argument("--safety-fraction", type=float, default=.8)
    parser.add_argument("--safe-only", action="store_true")
    args = parser.parse_args()
    try:
        matrix = build_matrix(budget_bytes=args.budget_mib * 1024**2,
                              safety_fraction=args.safety_fraction)
    except ValueError as exc:
        parser.error(str(exc))
    if args.safe_only:
        matrix["cases"] = [case for case in matrix["cases"] if case["preflight_safe"]]
    write_json(args.output, matrix)
    print(json.dumps({"cases": len(matrix["cases"]),
                      "safe": sum(case["preflight_safe"] for case in matrix["cases"]),
                      "output": args.output}, indent=2))


if __name__ == "__main__":
    main()
