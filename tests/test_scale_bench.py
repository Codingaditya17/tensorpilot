import pytest

from tensorpilot.scale_bench import percentile, run_benchmark


def test_scale_benchmark_accounts_for_all_placements_and_workers():
    result = run_benchmark(workers=8, placements=80, in_flight=24)
    measured = result["results"]
    assert measured["selected_workers"] == 8
    assert measured["assignments_per_worker"]["min"] == 10
    assert measured["assignments_per_worker"]["max"] == 10
    assert measured["placements_per_s"] > 0
    assert measured["snapshot_bytes"] > 0
    assert measured["health_summary"]["bytes"] < 1024
    assert measured["worker_pagination"]["pages"] == 1
    assert not result["configuration"]["gpu_allocated"]


def test_scale_benchmark_has_hard_safety_bounds():
    with pytest.raises(ValueError, match="workers"):
        run_benchmark(workers=10_001, placements=1, in_flight=1)
    with pytest.raises(ValueError, match="in_flight"):
        run_benchmark(workers=1, placements=2, in_flight=3)


def test_percentile_interpolates_and_handles_empty_input():
    assert percentile([], .5) is None
    assert percentile([1, 2, 3, 4], .5) == 2.5
