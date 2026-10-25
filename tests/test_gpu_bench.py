"""Benchmark configuration, planning and evidence layout. The end-to-end run uses a CPU worker."""
import asyncio
import json
import threading

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from aiohttp import web  # noqa: E402
from tiny_model import make_tiny_model_dir  # noqa: E402
from tensorpilot import gpu_bench  # noqa: E402
from tensorpilot.cuda_backend import CUDABackend  # noqa: E402
from tensorpilot.server import create_app  # noqa: E402

REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"


def base_config():
    return {
        "schema": gpu_bench.SCHEMA,
        "target": {"model": "tiny-local", "revision": REVISION},
        "assumptions": {"status": "placeholder_unmeasured", "decode_tokens_per_s": 60,
                        "prefill_tokens_per_s": 2000},
        "max_session_minutes": 5,
        "experiments": [{"id": "baseline", "kind": "closed_loop", "requests": 4, "prompt_repeats": 1,
                         "max_tokens": 4, "repeats": 1, "warmup_requests": 1, "timeout_s": 30,
                         "max_minutes": 2}],
    }


def test_config_rejects_unsafe_or_ambiguous_shapes():
    cases = {
        "schema": lambda c: c.update(schema="other"),
        "unknown field": lambda c: c["experiments"][0].update(surprise=1),
        "duplicate id": lambda c: c["experiments"].append(dict(c["experiments"][0])),
        "unbounded requests": lambda c: c["experiments"][0].update(requests=10**6),
        "bool as integer": lambda c: c["experiments"][0].update(max_tokens=True),
        "rates on closed loop": lambda c: c["experiments"][0].update(rates=[1.0]),
        "missing assumptions": lambda c: c.pop("assumptions"),
        "planned without reason": lambda c: c["experiments"].append(
            {"id": "later", "kind": "two_gpu_planned", "status": "planned", "minimum_gpus": 2}),
        "unknown kind": lambda c: c["experiments"][0].update(kind="magic"),
    }
    for name, mutate in cases.items():
        config = base_config()
        mutate(config)
        with pytest.raises(ValueError):
            gpu_bench.validate_config(config)


def test_plan_reports_assumptions_and_budget_without_touching_a_gpu():
    config = base_config()
    summary = gpu_bench.plan(config, budget_minutes=1)
    assert summary["rows"][0]["estimated_gpu_minutes"] > 0
    assert summary["session_overhead_minutes"] == gpu_bench.DEFAULT_SESSION_OVERHEAD_MINUTES
    assert summary["within_budget"] is False
    assert "not measurements" in summary["note"]


def test_measured_or_missing_assumption_status_is_rejected():
    config = base_config()
    config["assumptions"]["status"] = "measured"
    with pytest.raises(ValueError, match="placeholder_unmeasured"):
        gpu_bench.validate_config(config)
    config = base_config()
    del config["experiments"][0]["max_minutes"]
    with pytest.raises(ValueError, match="max_minutes"):
        gpu_bench.validate_config(config)


def test_run_requires_a_session_budget(tmp_path):
    config = base_config()
    del config["max_session_minutes"]
    with pytest.raises(ValueError, match="session budget"):
        gpu_bench.run(config, url="http://127.0.0.1:1", output_dir=tmp_path / "budget")


def test_abort_rules_stop_on_error_rate_oom_and_unready_worker():
    def report(error_rate=0.0, oom_before=0, oom_after=0, ready=True):
        return {"summary": {"error_rate": error_rate},
                "metadata": {"server": {"last_measured_memory": {"cuda_oom_events": oom_before}},
                             "server_after": {"ready": ready,
                                              "last_measured_memory": {"cuda_oom_events": oom_after}}}}
    assert gpu_bench._abort_reason(report(), .05) is None
    assert gpu_bench._abort_reason(report(error_rate=.2), .05).startswith("error_rate")
    assert gpu_bench._abort_reason(report(oom_before=0, oom_after=1), .05) == "cuda_out_of_memory"
    assert gpu_bench._abort_reason(report(ready=False), .05) == "worker_not_ready_after_run"


def test_time_cap_and_session_budget_are_enforced_without_a_server(tmp_path, monkeypatch):
    async def slow_run(args):
        await asyncio.sleep(10)

    monkeypatch.setattr(gpu_bench.load, "run", slow_run)
    config = base_config()
    config["experiments"][0]["warmup_requests"] = 0
    config["experiments"].append({**config["experiments"][0], "id": "second"})
    manifest = gpu_bench.run(config, url="http://127.0.0.1:1", output_dir=tmp_path / "capped",
                             session_minutes=0.0005)
    first = manifest["experiments"]["baseline"]
    assert first["status"] == "aborted" and first["abort_reason"] == "time_cap"
    second = manifest["experiments"]["second"]
    assert second["status"] == "skipped" and second["abort_reason"] == "session_budget"


def test_run_refuses_unpinned_revision_and_planned_experiments(tmp_path):
    config = base_config()
    config["target"]["revision"] = "main"
    with pytest.raises(ValueError, match="40-character"):
        gpu_bench.run(config, url="http://127.0.0.1:1", output_dir=tmp_path / "a")
    config = base_config()
    config["experiments"].append({"id": "two", "kind": "two_gpu_planned", "status": "planned",
                                  "minimum_gpus": 2, "reason": "needs hardware"})
    with pytest.raises(ValueError, match="cannot run"):
        gpu_bench.run(config, url="http://127.0.0.1:1", output_dir=tmp_path / "b", only=["two"])


class ThreadedWorker:
    """Run the production worker app on its own event loop so the synchronous runner can call it."""

    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        self.runner = web.AppRunner(create_app(lambda: CUDABackend(str(self.model_dir), device="cpu",
                                                                  allow_cpu=True, dtype="float32"),
                                               queue_size=8, max_tokens=16, warmup=True,
                                               metadata={"model": "tiny-local", "backend": "cuda"},
                                               instance_id="bench-worker-gen1"), access_log=None)
        self.call(self.runner.setup())
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        self.call(site.start())
        self.url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
        return self

    def call(self, coroutine):
        return asyncio.run_coroutine_threadsafe(coroutine, self.loop).result(60)

    def __exit__(self, *exc_info):
        self.call(self.runner.cleanup())
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(10)


def test_run_writes_one_report_per_repeat_with_server_provenance(tmp_path):
    model_dir = make_tiny_model_dir(tmp_path / "model")
    config = base_config()
    config["experiments"].append({"id": "paced", "kind": "open_loop", "requests": 3, "prompt_repeats": 1,
                                  "max_tokens": 3, "rates": [50.0], "repeats": 1, "warmup_requests": 0,
                                  "concurrency": 4, "timeout_s": 30, "max_minutes": 2})
    with ThreadedWorker(model_dir) as worker:
        manifest = gpu_bench.run(config, url=worker.url, output_dir=tmp_path / "evidence",
                                 config_path="inline.json")
    evidence = tmp_path / "evidence"
    assert set(manifest["experiments"]) == {"baseline", "paced"}
    assert (evidence / "manifest.json").exists()
    report = json.loads((evidence / "baseline" / "repeat01_closed.json").read_text())
    assert report["summary"]["successful_requests"] == 4
    assert report["summary"]["failed_requests"] == 0
    assert report["metadata"]["server"]["backend"] == "cuda"
    assert report["metadata"]["server"]["artifact_sha256"]
    assert report["metadata"]["server"]["gpu_telemetry"] is None
    assert report["config_sha256"] == gpu_bench.config_fingerprint(config)
    assert report["limitations"] and "one sequence at a time" in report["limitations"][0]
    paced = json.loads((evidence / "paced" / "repeat01_rate50p0.json").read_text())
    assert paced["experiment"]["arrival"] == "rate50p0"
    assert paced["summary"]["successful_requests"] == 3
    assert (evidence / "baseline" / "repeat01_closed.requests.jsonl").exists()
    with pytest.raises(FileExistsError):
        gpu_bench.run(config, url="http://127.0.0.1:1", output_dir=evidence)


def test_shipped_benchmark_configs_validate_and_keep_placeholders_labeled():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "benchmarks" / "gpu"
    configs = sorted(root.glob("*.json"))
    assert {path.name for path in configs} >= {"v09-first-30min.json", "v09-single-gpu.json",
                                              "v09-two-gpu-future.json"}
    for path in configs:
        config = gpu_bench.load_config(path)
        assert config["assumptions"]["status"] == "placeholder_unmeasured", path.name
        summary = gpu_bench.plan(config)
        assert summary["assumptions_status"] == "placeholder_unmeasured"
