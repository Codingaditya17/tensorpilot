import pytest

from tensorpilot.cluster_compare import compare


def report(kind, throughput=100, latency=.2, artifact="sha"):
    summary = {"completed_output_tokens_per_s": throughput,
               "p50_client_latency_s": latency / 2, "p95_client_latency_s": latency,
               "p99_client_latency_s": latency * 1.1, "p95_client_ttft_s": latency / 3,
               "p95_queue_wait_s": latency / 4, "p95_inter_token_latency_s": latency / 5}
    server = ({"artifact_sha256": artifact} if kind == "direct" else
              {"service": "tensorpilot-control-plane",
               "workers": [{"artifact_sha256": artifact}, {"artifact_sha256": artifact}]})
    return {"metadata": {"workload_sha256": "workload", "server": server,
                         "config": {"requests": 30, "concurrency": 4, "rate": 0,
                                    "timeout": 60, "max_tokens": 64}}, "summary": summary}


def test_cluster_comparison_preserves_values_and_relative_change():
    result = compare([report("direct", 100), report("direct", 110)],
                     [report("cluster", 120, .3), report("cluster", 130, .3)])
    assert result["groups"]["direct-single-worker"]["completed_output_tokens_per_s"]["values"] == [100, 110]
    assert result["relative_change"]["completed_output_tokens_per_s"] == pytest.approx(125 / 105 - 1)
    assert result["relative_change"]["p95_client_latency_s"] == pytest.approx(.5)


def test_cluster_comparison_rejects_mixed_artifacts_and_unbalanced_runs():
    mixed = report("cluster")
    mixed["metadata"]["server"]["workers"][1]["artifact_sha256"] = "other"
    with pytest.raises(ValueError, match="one pinned artifact"):
        compare([report("direct")], [mixed])
    with pytest.raises(ValueError, match="balanced"):
        compare([report("direct"), report("direct")], [report("cluster")])
