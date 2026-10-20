"""Compare matched direct-worker and control-plane reports without hiding variance."""
import argparse
import json
from pathlib import Path
import statistics

from .reporting import write_json

MEASURES = ("completed_output_tokens_per_s", "p50_client_latency_s",
            "p95_client_latency_s", "p99_client_latency_s", "p95_client_ttft_s",
            "p95_queue_wait_s", "p95_inter_token_latency_s")


def _artifact(report):
    server = report["metadata"]["server"]
    if server.get("service") == "tensorpilot-control-plane":
        if server.get("uniform_artifact_sha256"):
            return server["uniform_artifact_sha256"]
        artifacts = {worker.get("artifact_sha256") for worker in server.get("workers", [])}
        if None in artifacts or len(artifacts) != 1:
            raise ValueError("cluster comparison requires one pinned artifact across all workers")
        return artifacts.pop()
    artifact = server.get("artifact_sha256")
    if not artifact:
        raise ValueError("direct report is missing an artifact fingerprint")
    return artifact


def compare(baseline, cluster):
    if not baseline or len(baseline) != len(cluster):
        raise ValueError("balanced nonempty repetitions are required")
    reports = baseline + cluster
    workload = reports[0]["metadata"].get("workload_sha256")
    artifact = _artifact(reports[0])
    config_keys = ("requests", "concurrency", "rate", "timeout", "max_tokens")
    signature = tuple((key, reports[0]["metadata"]["config"].get(key)) for key in config_keys)
    for report in reports[1:]:
        if (report["metadata"].get("workload_sha256") != workload or _artifact(report) != artifact or
                tuple((key, report["metadata"]["config"].get(key)) for key in config_keys) != signature):
            raise ValueError("workload, artifact, or load configuration differs")

    def group(items):
        return {key: {"median": statistics.median(report["summary"][key] for report in items),
                      "values": [report["summary"][key] for report in items]}
                for key in MEASURES}

    baseline_group, cluster_group = group(baseline), group(cluster)
    changes = {key: cluster_group[key]["median"] / baseline_group[key]["median"] - 1
               for key in MEASURES}
    return {"schema_version": 1, "baseline": "direct-single-worker",
            "candidate": "two-worker-control-plane", "repetitions": len(baseline),
            "workload_sha256": workload, "artifact_sha256": artifact,
            "groups": {"direct-single-worker": baseline_group,
                       "two-worker-control-plane": cluster_group},
            "relative_change": changes,
            "interpretation": "Positive throughput is better; negative latency is better. Local worker processes share one Apple GPU and are not independent GPU nodes. Alternating sequential runs do not remove thermal or ordering effects."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", nargs="+", required=True)
    parser.add_argument("--cluster", nargs="+", required=True)
    parser.add_argument("--output", default="results/v07_cluster_comparison.json")
    args = parser.parse_args()
    try:
        result = compare([json.loads(Path(path).read_text()) for path in args.baseline],
                         [json.loads(Path(path).read_text()) for path in args.cluster])
    except (ValueError, KeyError) as exc:
        parser.error(str(exc))
    result["source_reports"] = {"baseline": args.baseline, "cluster": args.cluster}
    write_json(args.output, result)
    print(json.dumps(result["relative_change"], indent=2))


if __name__ == "__main__":
    main()
