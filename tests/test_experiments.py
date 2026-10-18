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
from tensorpilot.workloads import build_workload, fingerprint, validate_workload


def test_mixed_plan_is_seeded_with_fixed_composition():
    first = build_workload(30, kind="mixed", seed=42)
    assert first == build_workload(30, kind="mixed", seed=42)
    second = build_workload(30, kind="mixed", seed=43)
    assert fingerprint(first) != fingerprint(second)
    assert sorted(item["class"] for item in first) == sorted(item["class"] for item in second)
    assert sum(item["class"] == "short" for item in first) == 18
    assert {item["max_tokens"] for item in first} == {16, 64, 128}
    validate_workload(first)


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
