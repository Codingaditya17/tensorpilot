import asyncio
import json
from types import ModuleType, SimpleNamespace
import sys

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
import pytest

from tensorpilot.artifacts import artifact_fingerprint, resolve_model
from tensorpilot.compare import compare_reports, regression_gate
from tensorpilot.experiment import ensure_port_free, run_order
from tensorpilot.load import run
from tensorpilot.reporting import write_json
from tensorpilot.workloads import build_stress_workload, build_workload, fingerprint, validate_workload


def test_mixed_plan_is_seeded_with_fixed_composition():
    first = build_workload(30, kind="mixed", seed=42)
    assert first == build_workload(30, kind="mixed", seed=42)
    second = build_workload(30, kind="mixed", seed=43)
    assert fingerprint(first) != fingerprint(second)
    assert sorted(item["class"] for item in first) == sorted(item["class"] for item in second)
    assert sum(item["class"] == "short" for item in first) == 18
    assert {item["max_tokens"] for item in first} == {16, 64, 128}
    validate_workload(first)


def test_stress_workload_has_controlled_context_and_output_shape():
    plan = build_stress_workload(3, prompt_repeats=8, output_tokens=32)
    assert len(plan) == 3
    assert {item["max_tokens"] for item in plan} == {32}
    assert {item["class"] for item in plan} == {"context-r8-out32"}
    validate_workload(plan)
    with pytest.raises(ValueError):
        build_stress_workload(1, prompt_repeats=1000)


@pytest.mark.parametrize("plan", [[], [{}], [{"prompt": "hi", "max_tokens": True, "class": "x"}],
                                 [{"prompt": "", "max_tokens": 1, "class": "x"}]])
def test_invalid_workloads(plan):
    with pytest.raises(ValueError):
        validate_workload(plan)


def test_artifact_digest_tracks_content_and_is_independent_of_directory_name(tmp_path):
    left, right = tmp_path / "left", tmp_path / "right"
    left.mkdir()
    right.mkdir()
    for folder in (left, right):
        (folder / "config.json").write_text('{"model": "stub"}')
        (folder / "model.safetensors").write_bytes(b"synthetic-weights")
    original = artifact_fingerprint(left)
    assert original == artifact_fingerprint(right)
    (left / ".DS_Store").write_text("ignored")
    assert original == artifact_fingerprint(left)
    (left / "model.safetensors").write_bytes(b"modified-weights")
    assert original != artifact_fingerprint(left)
    path, metadata = resolve_model(str(right))
    assert path == right
    assert metadata["model_source"] == "local_directory"
    assert metadata["artifact_sha256"] == artifact_fingerprint(right)
    with pytest.raises(ValueError):
        resolve_model(str(right), revision="main")


def test_remote_model_revision_and_offline_flag_are_recorded(tmp_path, monkeypatch):
    snapshot = tmp_path / ("a" * 40)
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}")
    calls = []
    module = ModuleType("huggingface_hub")

    def download(**kwargs):
        calls.append(kwargs)
        return str(snapshot)

    module.snapshot_download = download
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    _, metadata = resolve_model("test/model", revision="v1", offline=True)
    assert metadata["model_revision"] == "a" * 40
    assert calls[0]["local_files_only"] is True
    assert calls[0]["revision"] == "v1"


def test_atomic_report_does_not_destroy_previous_report_on_encoding_error(tmp_path):
    path = tmp_path / "report.json"
    write_json(path, {"old": True})
    with pytest.raises(ValueError):
        write_json(path, {"bad": float("nan")})
    assert json.loads(path.read_text()) == {"old": True}
    assert list(tmp_path.iterdir()) == [path]


def report(policy, latency=.2, throughput=100):
    summary = {"successful_requests": 30, "failed_requests": 0, "rejected_requests": 0,
               "p95_client_latency_s": latency, "completed_output_tokens_per_s": throughput}
    return {"metadata": {"workload_sha256": "matched-plan", "config": {"requests": 30, "concurrency": 6,
                                                                           "rate": 0, "timeout": 60},
                         "server": {"artifact_sha256": "matched-artifact", "scheduler": policy}},
            "summary": summary, "by_class": {"short": dict(summary), "long": dict(summary)}}


def test_comparison_preserves_run_variation_and_per_class_changes():
    result = compare_reports([report("fifo", .2, 100), report("shortest-budget", .1, 110),
                              report("fifo", .4, 100), report("shortest-budget", .2, 110)])
    assert result["by_policy"]["fifo"]["overall"]["metrics"]["p95_client_latency_s"]["values"] == [.2, .4]
    assert result["relative_change"]["overall"]["p95_client_latency_s"] == pytest.approx(-.5)
    assert result["relative_change"]["by_class"]["long"]["completed_output_tokens_per_s"] == pytest.approx(.1)
    assert regression_gate(result, .1)["passed"]


@pytest.mark.parametrize("field,value", [("workload_sha256", "different"),
                                        ("artifact_sha256", "different"), ("concurrency", 3)])
def test_comparison_refuses_mismatched_runs(field, value):
    left, right = report("fifo"), report("shortest-budget")
    if field == "workload_sha256":
        right["metadata"][field] = value
    elif field == "concurrency":
        right["metadata"]["config"][field] = value
    else:
        right["metadata"]["server"][field] = value
    with pytest.raises(ValueError, match="differs"):
        compare_reports([left, right])


def test_comparison_requires_balanced_repetitions_and_provenance():
    with pytest.raises(ValueError, match="equal repetition"):
        compare_reports([report("fifo"), report("fifo"), report("shortest-budget")])
    left = report("fifo")
    del left["metadata"]["server"]["artifact_sha256"]
    with pytest.raises(ValueError, match="fingerprints"):
        compare_reports([left, report("shortest-budget")])


def test_regression_gate_fails_latency_and_error_regressions():
    left, right = report("fifo", .2), report("shortest-budget", .3)
    right["summary"]["failed_requests"] = 1
    result = regression_gate(compare_reports([left, right]), .1)
    assert not result["passed"]
    assert len(result["reasons"]) == 2


def test_regression_gate_rejects_faster_but_less_available_candidate():
    left, right = report("fifo", .2), report("shortest-budget", .1)
    right["summary"]["rejected_requests"] = 1
    result = regression_gate(compare_reports([left, right]), .1)
    assert not result["passed"]
    assert result["reasons"] == ["candidate has more rejected requests"]


@pytest.mark.parametrize("partial", ["duration", "recovery"])
def test_regression_comparison_refuses_partial_or_different_observation_windows(partial):
    baseline, candidate = report("fifo"), report("shortest-budget")
    if partial == "duration":
        candidate["metadata"]["config"]["duration_s"] = 30
    else:
        candidate["metadata"]["recovery"] = {"run_status": "interrupted"}
    with pytest.raises(ValueError, match="complete fixed-count"):
        compare_reports([baseline, candidate])


def test_alternating_policy_order():
    assert run_order(2) == [(1, "fifo"), (1, "shortest-budget"),
                            (2, "shortest-budget"), (2, "fifo")]


def execution_report(mode, capacity=2):
    result = report("fifo")
    result["metadata"]["server"].update(execution_mode=mode,
        max_active_sequences=capacity if mode == "batch" else 1,
        prefill_step_size=256 if mode == "batch" else 2048)
    return result


def test_execution_comparison_and_gate():
    result = compare_reports([execution_report("serial"), execution_report("batch")], axis="execution_mode")
    assert result["by_variant"]["batch"]["repetitions"] == 1
    assert regression_gate(result, .05)["passed"]
    assert run_order(2, ("serial", "batch")) == [(1, "serial"), (1, "batch"), (2, "batch"), (2, "serial")]


def test_policy_comparison_rejects_different_execution():
    left, right = execution_report("serial"), execution_report("batch")
    right["metadata"]["server"]["scheduler"] = "shortest-budget"
    with pytest.raises(ValueError, match="differs"):
        compare_reports([left, right])
    with pytest.raises(ValueError, match="differs"):
        compare_reports([left, right], axis="execution_mode")


def test_execution_comparison_rejects_inconsistent_capacity():
    with pytest.raises(ValueError, match="execution settings"):
        compare_reports([execution_report("serial"), execution_report("serial"),
                         execution_report("batch", 2), execution_report("batch", 4)], axis="execution_mode")


def test_load_rejects_mismatched_stream_token_counts():
    async def scenario():
        app = web.Application()

        async def health(request):
            return web.json_response({"ready": True})

        async def broken(request):
            events = [{"type": "token", "output": True, "text": "x", "token_id": 1},
                      {"type": "done", "generated_tokens": 2, "queue_wait_s": 0}]
            return web.Response(text="".join(json.dumps(event) + "\n" for event in events))

        app.router.add_get("/health", health)
        app.router.add_post("/v1/generate", broken)
        async with TestClient(TestServer(app)) as client:
            args = SimpleNamespace(url=str(client.make_url("/")), requests=1, concurrency=1,
                                   rate=0, max_tokens=1, timeout=5, output="unused")
            result = await run(args)
            assert result["summary"]["failed_requests"] == 1
            assert result["requests"][0]["error_code"] == "token_count_mismatch"
            with pytest.raises(RuntimeError, match="already in use"):
                await ensure_port_free(client.server.port)
    asyncio.run(scenario())
