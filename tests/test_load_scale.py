"""Load generator: arrival processes, request accounting, cancellation, timeouts, SLO goodput and percentiles."""
import argparse
import asyncio
import json
import statistics
from types import SimpleNamespace

import pytest
from aiohttp.test_utils import TestServer

from tensorpilot import load
from tensorpilot.server import ENGINE, create_app
from tensorpilot.workloads import validate_workload
from test_serving import FakeBackend


def args(**overrides):
    values = dict(url="", requests=1, concurrency=1, rate=0, max_tokens=8, timeout=10, workload="uniform",
                  workload_file=None, seed=7, duration_s=None, max_requests=1000, journal=None, journal_fsync=False,
                  compact=False, overwrite=True, arrival="uniform", burst_size=8, slo=None)
    values.update(overrides)
    return argparse.Namespace(**values)


def item(prompt="hello", max_tokens=6, cls="c", **extra):
    return {"prompt": prompt, "max_tokens": max_tokens, "class": cls, **extra}


async def run_plan(tmp_path, plan, *, queue_size=8, **overrides):
    app = create_app(FakeBackend, queue_size=queue_size, warmup=False, max_queue_wait_s=1.0)
    async with TestServer(app) as server:
        path = tmp_path / "plan.json"
        path.write_text(json.dumps(validate_workload(plan)))
        result = await load.run(args(url=str(server.make_url("")), requests=len(plan), workload_file=str(path),
                                     **overrides))
        engine = app[ENGINE]
        for _ in range(100):
            if engine.active_count == 0 and engine.queue.qsize() == 0:
                break
            await asyncio.sleep(.02)
        return result, engine


def outcomes(result):
    return [record["outcome"] for record in result["requests"]]


def test_every_request_gets_exactly_one_outcome_and_a_client_cancel_is_not_an_error(tmp_path):
    plan = [item(max_tokens=6), item(max_tokens=60, cancel_after_tokens=2), item(max_tokens=6)]
    result, engine = asyncio.run(run_plan(tmp_path, plan))
    assert outcomes(result) == ["completed", "client_cancelled", "completed"]
    summary = result["summary"]
    assert summary["requests_total"] == 3 and summary["successful_requests"] == 2
    assert summary["client_cancelled_requests"] == 1 and summary["failed_requests"] == 0
    assert summary["error_rate"] == 0 and summary["outcomes"] == {"client_cancelled": 1, "completed": 2}
    cancelled = result["requests"][1]
    assert cancelled["received_output_tokens"] == 2 and "error" not in cancelled
    assert engine.active_count == 0, "the server must release a stream the client abandoned"


def test_a_time_based_cancel_drops_the_request_wherever_it_is_and_frees_the_worker(tmp_path):
    plan = [item(max_tokens=200, cancel_after_s=0.03), item(max_tokens=4)]
    result, engine = asyncio.run(run_plan(tmp_path, plan))
    assert outcomes(result) == ["client_cancelled", "completed"]
    assert result["requests"][0]["client_elapsed_s"] < 0.5
    assert engine.active_count == 0 and engine.queue.qsize() == 0


def test_a_request_that_outlives_its_own_deadline_is_a_timeout_not_a_transport_failure(tmp_path):
    plan = [item(max_tokens=200, timeout_s=0.06), item(max_tokens=4)]
    result, _ = asyncio.run(run_plan(tmp_path, plan))
    assert outcomes(result) == ["timed_out", "completed"]
    summary = result["summary"]
    assert summary["timed_out_requests"] == 1 and summary["failed_requests"] == 1
    assert summary["successful_requests"] == 1


def test_overload_is_counted_as_rejection_with_the_accepted_requests_still_completing(tmp_path):
    plan = [item(max_tokens=40) for _ in range(10)]
    result, _ = asyncio.run(run_plan(tmp_path, plan, queue_size=1, concurrency=10))
    counts = result["summary"]["outcomes"]
    assert counts.get("rejected", 0) >= 1 and counts.get("completed", 0) >= 1
    assert sum(counts.values()) == 10
    assert result["summary"]["rejected_requests"] == counts["rejected"]
    assert all(r["error_code"] == "overloaded" for r in result["requests"] if r["outcome"] == "rejected")


def test_per_request_timeout_and_cancel_fields_are_validated():
    assert validate_workload([item(cancel_after_tokens=3, cancel_after_s=1.5, timeout_s=9, prompt_tokens=4)])
    with pytest.raises(ValueError):
        validate_workload([item(cancel_after_tokens=0)])


# ----- arrival processes ----------------------------------------------------------------------------

def test_closed_loop_has_no_schedule_and_uniform_is_exactly_spaced():
    assert load.arrival_offsets(args(rate=0), 5) is None
    assert load.arrival_offsets(args(rate=4), 5) == [0, .25, .5, .75, 1.0]


def test_poisson_arrivals_are_seeded_have_the_requested_mean_rate_and_are_not_uniform():
    first = load.arrival_offsets(args(rate=20, arrival="poisson", seed=3), 6000)
    again = load.arrival_offsets(args(rate=20, arrival="poisson", seed=3), 6000)
    other = load.arrival_offsets(args(rate=20, arrival="poisson", seed=4), 6000)
    assert first == again and first != other
    gaps = [b - a for a, b in zip(first, first[1:])]
    assert statistics.fmean(gaps) == pytest.approx(1 / 20, rel=0.05)
    assert statistics.pstdev(gaps) == pytest.approx(statistics.fmean(gaps), rel=0.1), "exponential gaps: sd ~ mean"


def test_burst_arrivals_release_groups_together_at_the_same_average_rate():
    offsets = load.arrival_offsets(args(rate=10, arrival="burst", burst_size=5), 20)
    assert offsets[:5] == [0] * 5 and offsets[5:10] == [.5] * 5 and offsets[15:] == [1.5] * 5
    assert offsets[-1] + 0.5 == pytest.approx(20 / 10)


def test_arrival_arguments_are_validated():
    with pytest.raises(ValueError, match="nonzero --rate"):
        load.validate_config(args(arrival="poisson", rate=0))
    with pytest.raises(ValueError, match="arrival must be"):
        load.validate_config(args(arrival="zipf", rate=1))
    with pytest.raises(ValueError, match="burst_size"):
        load.validate_config(args(arrival="burst", rate=1, burst_size=0))
    with pytest.raises(ValueError, match="slo"):
        load.validate_config(args(slo={"latency": 1}))
    with pytest.raises(ValueError, match="positive finite"):
        load.validate_config(args(slo={"ttft_s": -1}))


# ----- summary statistics --------------------------------------------------------------------------

def record(index, *, ttft, elapsed, tokens=10, itl=None, offset=0.0, ok=True, outcome=None, **extra):
    if itl is None:
        itl = [(elapsed - ttft) / max(tokens - 1, 1)] * (tokens - 1) if ttft is not None else []
    base = {"index": index, "ok": ok, "class": "c", "dispatch_offset_s": offset, "client_ttft_s": ttft,
            "client_elapsed_s": elapsed, "client_inter_token_s": itl,
            "client_tpot_s": statistics.fmean(itl) if itl else None,
            "client_last_token_s": elapsed, "received_output_tokens": tokens,
            "server": {"generated_tokens": tokens, "queue_wait_s": 0.01, "inter_token_latency_s": itl}}
    base.update(extra)
    if outcome:
        base["outcome"] = outcome
    return base


def test_goodput_counts_only_requests_that_meet_every_declared_threshold():
    records = [record(0, ttft=.1, elapsed=1.0), record(1, ttft=.5, elapsed=1.0),      # too slow to first token
               record(2, ttft=.1, elapsed=3.0), record(3, ttft=.2, elapsed=1.5)]      # too slow end to end / fine
    summary = load.summarize(records, wall_s=10, slo={"ttft_s": .3, "e2e_s": 2.0})
    slo = summary["slo"]
    assert slo["good_requests"] == 2 and slo["attainment"] == 0.5
    assert slo["goodput_requests_per_s"] == pytest.approx(0.2)
    assert slo["goodput_output_tokens_per_s"] == pytest.approx(2.0)
    assert summary["successful_requests_per_s"] == pytest.approx(0.4), "raw throughput counts the slow ones too"


def test_the_tpot_threshold_uses_mean_inter_token_latency():
    fast, slow = record(0, ttft=.1, elapsed=1.0, itl=[.02] * 9), record(1, ttft=.1, elapsed=1.0, itl=[.2] * 9)
    assert load.summarize([fast, slow], 5, slo={"tpot_s": .05})["slo"]["good_requests"] == 1


def test_failed_requests_are_never_good_and_cancellations_do_not_dilute_attainment():
    failed = record(1, ttft=None, elapsed=.2, ok=False, outcome="failed", error_code="boom")
    cancelled = record(2, ttft=.1, elapsed=.2, ok=False, outcome="client_cancelled")
    summary = load.summarize([record(0, ttft=.1, elapsed=.5), failed, cancelled], 5, slo={"e2e_s": 9})
    assert summary["slo"]["attainment"] == 0.5 and summary["error_rate"] == 0.5
    assert summary["client_cancelled_requests"] == 1 and summary["errors_by_code"] == {"boom": 1}


def test_percentiles_are_computed_over_completed_requests():
    records = [record(i, ttft=0.1 + i / 100, elapsed=1 + i / 10) for i in range(100)]
    summary = load.summarize(records, 50)
    assert summary["p50_client_latency_s"] == pytest.approx(statistics.median(1 + i / 10 for i in range(100)))
    assert summary["p99_client_latency_s"] == pytest.approx(10.801, abs=0.01)
    assert summary["p99_client_ttft_s"] == pytest.approx(1.0801, abs=0.001)
    assert summary["p50_client_tpot_s"] is not None and summary["p95_client_inter_token_s"] is not None


def test_wall_clock_and_active_window_throughput_are_reported_separately():
    # Two requests of 100 tokens; the first token appears at 5 s and the last at 6 s: a 1 s active window.
    records = [record(0, ttft=1.0, elapsed=2.0, tokens=100, offset=4.0, client_last_token_s=2.0),
               record(1, ttft=1.0, elapsed=2.0, tokens=100, offset=4.0, client_last_token_s=2.0)]
    summary = load.summarize(records, wall_s=20)
    assert summary["completed_output_tokens_per_s"] == pytest.approx(10.0)
    assert summary["active_window_s"] == pytest.approx(1.0)
    assert summary["active_window_output_tokens_per_s"] == pytest.approx(200.0)


def test_older_records_without_outcomes_still_summarize():
    legacy = {"index": 0, "ok": True, "class": "c", "client_elapsed_s": 1.0, "client_ttft_s": .1,
              "server": {"generated_tokens": 4, "queue_wait_s": 0.0}}
    summary = load.summarize([legacy, {**legacy, "index": 1, "ok": False, "error_code": "overloaded"}], 2)
    assert summary["successful_requests"] == 1 and summary["rejected_requests"] == 1
    assert summary["outcomes"] == {"completed": 1, "failed": 1}


def test_a_workload_spec_can_drive_the_runner_directly(tmp_path):
    spec = {"seed": 1, "classes": [{"name": "s", "prompt_tokens": 5, "output_tokens": 4}]}
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))

    async def scenario():
        app = create_app(FakeBackend, queue_size=8, warmup=False)
        async with TestServer(app) as server:
            return await load.run(args(url=str(server.make_url("")), requests=5, workload_spec=str(path),
                                       concurrency=2))

    result = asyncio.run(scenario())
    assert result["summary"]["successful_requests"] == 5
    assert all(r["prompt_tokens"] == 5 for r in result["requests"])


def test_the_namespace_gpu_bench_builds_still_works(tmp_path):
    """gpu_bench constructs its own Namespace without the new attributes; defaults must apply."""
    old = SimpleNamespace(url="", requests=1, concurrency=1, rate=0, max_tokens=4, timeout=10, workload="uniform",
                          workload_file=None, seed=1, duration_s=None, max_requests=10, journal=None,
                          journal_fsync=False, compact=False, overwrite=True)

    async def scenario():
        app = create_app(FakeBackend, queue_size=8, warmup=False)
        async with TestServer(app) as server:
            old.url = str(server.make_url(""))
            return await load.run(old)

    assert asyncio.run(scenario())["summary"]["successful_requests"] == 1
