import asyncio

import pytest

from tensorpilot.synthetic_cluster import run_scenario


def test_synthetic_cluster_exercises_network_fault_recovery_and_churn():
    result = asyncio.run(run_scenario(workers=6, requests=120, concurrency=12,
                                      delay_ms=5, jitter_ms=1, fault_workers=1,
                                      fault_after_s=.03, fault_duration_s=.08, seed=7))
    assert result["configuration"]["transport"] == "real-loopback-tcp"
    assert not result["configuration"]["gpu_allocated"]
    assert result["summary"]["successful_requests"] > 0
    assert 0 <= result["summary"]["error_rate"] < 1
    assert result["faults"]["detection_s"] < 2
    assert result["faults"]["recovery_s"] < 2
    assert result["faults"]["recovered_instance_id"].endswith("generation-2")
    assert result["final_health"]["worker_counts"]["registered"] == 6


def test_synthetic_cluster_rejects_unsafe_sizes():
    with pytest.raises(ValueError, match="workers"):
        asyncio.run(run_scenario(workers=65, requests=1, concurrency=1))
