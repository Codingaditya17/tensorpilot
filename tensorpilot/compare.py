"""Compare matched repeated runs; preserve variability and per-class tradeoffs."""
import argparse
import json
import math
from pathlib import Path
import statistics

from .reporting import write_json

MEASURES = ("completed_output_tokens_per_s", "p50_client_latency_s", "p95_client_latency_s",
            "p99_client_latency_s", "p95_client_ttft_s", "p95_queue_wait_s", "max_queue_wait_s")


def _signature(report):
    metadata = report["metadata"]
    server = metadata["server"]
    config = metadata["config"]
    if config.get("duration_s") is not None or metadata.get("recovery"):
        raise ValueError("policy comparison requires complete fixed-count runs, not soak/recovered windows")
    if not metadata.get("workload_sha256") or not server.get("artifact_sha256"):
        raise ValueError("comparison requires workload and model artifact fingerprints")
    # Exclude policy/instance identity, output filenames and run timestamps.
    return (metadata["workload_sha256"], metadata.get("client_source_sha256"), server["artifact_sha256"],
            tuple((key, config.get(key)) for key in ("requests", "concurrency", "rate", "timeout", "compact", "journal_fsync")),
            tuple((key, server.get(key)) for key in (
                "model", "model_revision", "memory_limit_bytes", "max_prompt_tokens",
                "max_output_tokens", "sampling", "prompt_format", "mlx_version",
                "mlx_lm_version", "server_platform", "server_python", "queue_capacity", "max_queue_wait_s",
                "source_sha256", "dependencies", "mlx_device", "memory_profiling", "stall_timeout_s")))


def _distribution(values):
    values = [value for value in values if value is not None]
    if not values:
        return {"median": None, "min": None, "max": None, "values": []}
    return {"median": statistics.median(values), "min": min(values), "max": max(values), "values": values}


def _aggregate(summaries):
    return {"successful_requests": sum(item["successful_requests"] for item in summaries),
            "failed_requests": sum(item["failed_requests"] for item in summaries),
            "rejected_requests": sum(item["rejected_requests"] for item in summaries),
            "metrics": {key: _distribution([item.get(key) for item in summaries]) for key in MEASURES}}


def compare_reports(reports, baseline="fifo", candidate="shortest-budget"):
    if not reports:
        raise ValueError("no reports supplied")
    expected = _signature(reports[0])
    if any(_signature(report) != expected for report in reports[1:]):
        raise ValueError("workload, model artifact, environment, or experiment configuration differs")
    grouped = {}
    for report in reports:
        policy = report["metadata"]["server"].get("scheduler")
        if policy not in (baseline, candidate):
            raise ValueError(f"unexpected scheduler policy: {policy}")
        grouped.setdefault(policy, []).append(report)
    if baseline == candidate or set(grouped) != {baseline, candidate}:
        raise ValueError("two distinct scheduler policies are required")
    if len(grouped[baseline]) != len(grouped[candidate]):
        raise ValueError("policies must have equal repetition counts")
    labels = set(reports[0].get("by_class", {}))
    if any(set(report.get("by_class", {})) != labels for report in reports):
        raise ValueError("request classes differ between reports")
    by_policy = {}
    for policy, runs in grouped.items():
        by_policy[policy] = {"repetitions": len(runs), "overall": _aggregate([r["summary"] for r in runs]),
                             "by_class": {label: _aggregate([r["by_class"][label] for r in runs])
                                          for label in sorted(labels)}}

    def changes(base, newer):
        result = {}
        for key in MEASURES:
            before = base["metrics"][key]["median"]
            after = newer["metrics"][key]["median"]
            result[key] = (after / before - 1) if before is not None and before > 0 and after is not None else None
        return result

    return {"schema_version": 1, "baseline": baseline, "candidate": candidate,
            "workload_sha256": reports[0]["metadata"]["workload_sha256"],
            "artifact_sha256": reports[0]["metadata"]["server"]["artifact_sha256"],
            "by_policy": by_policy,
            "relative_change": {"overall": changes(by_policy[baseline]["overall"], by_policy[candidate]["overall"]),
                                 "by_class": {label: changes(by_policy[baseline]["by_class"][label],
                                                              by_policy[candidate]["by_class"][label]) for label in sorted(labels)}},
            "interpretation": "Relative changes compare medians of per-run metrics. Negative latency change is lower latency; positive throughput change is higher throughput. Small repeated runs do not establish statistical significance or generalize to other workloads."}


def regression_gate(comparison, max_p95_regression):
    if not math.isfinite(max_p95_regression) or max_p95_regression < 0:
        raise ValueError("regression threshold must be finite and nonnegative")
    base = comparison["by_policy"][comparison["baseline"]]["overall"]
    candidate = comparison["by_policy"][comparison["candidate"]]["overall"]
    change = comparison["relative_change"]["overall"]["p95_client_latency_s"]
    reasons = []
    if change is None:
        reasons.append("insufficient successful latency measurements")
    elif change > max_p95_regression:
        reasons.append("overall P95 latency exceeds the allowed regression")
    if candidate["failed_requests"] > base["failed_requests"]:
        reasons.append("candidate has more failed requests")
    return {"passed": not reasons, "max_p95_regression": max_p95_regression, "reasons": reasons}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+")
    parser.add_argument("--output", default="results/comparison.json")
    parser.add_argument("--max-p95-regression", type=float)
    args = parser.parse_args()
    try:
        comparison = compare_reports([json.loads(Path(path).read_text()) for path in args.reports])
        comparison["source_reports"] = args.reports
        if args.max_p95_regression is not None:
            comparison["gate"] = regression_gate(comparison, args.max_p95_regression)
    except (ValueError, KeyError) as exc:
        parser.error(str(exc))
    write_json(args.output, comparison)
    print(json.dumps(comparison["relative_change"], indent=2))
    print(f"Saved: {args.output}")
    if comparison.get("gate", {}).get("passed") is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
