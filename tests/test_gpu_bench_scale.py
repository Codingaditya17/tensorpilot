"""The v1.0 evaluation path through gpu_bench: workload specs, arrival processes, SLOs, telemetry and counters.

Runs a real batching worker on CPU with a tiny model. It checks the harness and its evidence, not GPU performance.
"""
import asyncio
import copy
import json
import threading

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from aiohttp import web  # noqa: E402
from tiny_model import make_tiny_model_dir  # noqa: E402
from tensorpilot import gpu_bench  # noqa: E402
from tensorpilot.cuda_backend import CUDABackend  # noqa: E402
from tensorpilot.cuda_batch import CUDABatchAdapterFactory  # noqa: E402
from tensorpilot.server import create_app  # noqa: E402

REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
SPEC = {"seed": 3, "classes": [{"name": "short", "weight": 3, "prompt_tokens": {"dist": "uniform", "min": 3, "max": 12},
                                "output_tokens": {"dist": "uniform", "min": 4, "max": 10}},
                               {"name": "long", "weight": 1, "prompt_tokens": 30, "output_tokens": 12}],
        "cancel": {"fraction": 0.25, "after_tokens": 2}}


def config():
    return {"schema": gpu_bench.SCHEMA, "target": {"model": "tiny-local", "revision": REVISION},
            "assumptions": {"status": "placeholder_unmeasured", "decode_tokens_per_s": 60, "prefill_tokens_per_s": 2000},
            "max_session_minutes": 5, "telemetry": {"interval_s": 0.05},
            "experiments": [
                {"id": "sweep-c4", "kind": "closed_loop", "requests": 24, "concurrency": 4, "timeout_s": 30,
                 "max_minutes": 2, "estimated_minutes": 1, "workload_spec": SPEC, "warmup_requests": 2,
                 "slo": {"ttft_s": 10, "e2e_s": 30}},
                {"id": "poisson", "kind": "open_loop", "requests": 10, "rates": [40.0], "arrival": "poisson",
                 "concurrency": 16, "timeout_s": 30, "max_minutes": 2, "estimated_minutes": 1,
                 "workload_spec": SPEC, "telemetry": False}]}


class BatchWorker:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        app = create_app(lambda: CUDABackend(str(self.model_dir), device="cpu", allow_cpu=True, dtype="float32"),
                         queue_size=32, max_tokens=32, warmup=False, execution="batch", batch_size=4,
                         prefill_step_size=16, batch_adapter_factory=CUDABatchAdapterFactory(kv_budget_bytes=64 * 1024**2),
                         metadata={"model": "tiny-local", "backend": "cuda"}, instance_id="scale-worker")
        self.runner = web.AppRunner(app, access_log=None)
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


@pytest.fixture(scope="module")
def evidence(tmp_path_factory):
    root = tmp_path_factory.mktemp("scale-evidence")
    model_dir = make_tiny_model_dir(root / "model")
    with BatchWorker(model_dir) as worker:
        manifest = gpu_bench.run(config(), url=worker.url, output_dir=root / "out", config_path="inline.json")
    return root / "out", manifest


def load_report(directory, experiment, name):
    return json.loads((directory / experiment / name).read_text())


def test_both_experiments_complete_and_the_manifest_records_the_worker_mode(evidence):
    directory, manifest = evidence
    assert {k: v["status"] for k, v in manifest["experiments"].items()} == {"sweep-c4": "completed",
                                                                          "poisson": "completed"}
    assert manifest["server_at_start"]["execution_mode"] == "batch"
    assert any("fixed-slot KV pool" in line for line in manifest["limitations"])
    assert not any("one sequence at a time" in line for line in manifest["limitations"])


def test_cancellation_is_accounted_separately_and_goodput_is_reported(evidence):
    report = load_report(*evidence[:1], "sweep-c4", "repeat01_closed.json")
    summary = report["summary"]
    assert summary["requests_total"] == 24
    assert summary["client_cancelled_requests"] > 0 and summary["successful_requests"] > 0
    assert summary["failed_requests"] == 0 and summary["error_rate"] == 0
    assert sum(summary["outcomes"].values()) == 24
    assert summary["slo"]["good_requests"] == summary["successful_requests"], "generous SLO: every completion is good"
    assert report["experiment"]["arrival_process"] == "closed_loop"


def test_each_report_carries_window_telemetry_and_server_counters(evidence):
    directory, manifest = evidence
    report = load_report(directory, "sweep-c4", "repeat01_closed.json")
    window = report["telemetry"]
    assert window["window_s"] > 0 and window["server_samples"] >= 2, "a window always holds its two endpoints"
    assert window["active_sequences"] is not None and window["kv_rows_active"] is not None
    delta = report["server_counters_delta"]
    assert delta["tensorpilot_jobs_total{outcome=completed}"] > 0
    assert any(key.startswith("tensorpilot_admission_decisions_total") for key in delta)
    events = [json.loads(line)["event"] for line in
              (directory / "sweep-c4" / "telemetry" / "events.jsonl").read_text().splitlines()]
    assert events[0] == "collector_started" and "warmup_start" in events and "repeat01_closed_start" in events
    assert events.index("repeat01_closed_start") < events.index("repeat01_closed_end")
    overhead = manifest["experiments"]["sweep-c4"]["telemetry_overhead"]
    assert overhead["server_polls"] >= 1 and overhead["gpu_status"].startswith("unavailable")


def test_small_samples_are_labeled_and_the_batch_limitation_text_is_used(evidence):
    report = load_report(*evidence[:1], "sweep-c4", "repeat01_closed.json")
    assert "small sample" in report["statistics_caveat"] and "maximum" in report["statistics_caveat"]
    assert any("fixed-slot KV pool" in line for line in report["limitations"])


def test_poisson_arrivals_are_recorded_and_telemetry_can_be_switched_off_per_experiment(evidence):
    directory, manifest = evidence
    report = load_report(directory, "poisson", "repeat01_rate40p0.json")
    assert report["experiment"]["arrival_process"] == "poisson" and "poisson" in report["metadata"]["arrival_mode"]
    assert "telemetry" not in report and not (directory / "poisson" / "telemetry").exists()
    assert "telemetry_overhead" not in manifest["experiments"]["poisson"]
    assert report["summary"]["requests_total"] == 10


def test_the_runner_reports_which_requests_were_cancelled_by_design(evidence):
    directory, _ = evidence
    journal = [json.loads(line) for line in (directory / "sweep-c4" / "repeat01_closed.requests.jsonl")
               .read_text().splitlines() if '"type":"request"' in line]
    cancelled = [r["record"] for r in journal if r["record"]["outcome"] == "client_cancelled"]
    assert cancelled and all(r["received_output_tokens"] == r["cancel_after_tokens"] == 2 for r in cancelled)


# ----- configuration -------------------------------------------------------------------------------

def invalid(mutator, match=None):
    cfg = copy.deepcopy(config())
    mutator(cfg)
    with pytest.raises(ValueError, match=match):
        gpu_bench.validate_config(cfg)


def test_the_new_experiment_fields_are_validated():
    invalid(lambda c: c["experiments"][0].pop("estimated_minutes"), "estimated_minutes")
    invalid(lambda c: c["experiments"][0].update(arrival="poisson"), "only valid for open loops")
    invalid(lambda c: c["experiments"][1].update(arrival="zipf"), "uniform, poisson or burst")
    invalid(lambda c: c["experiments"][1].update(burst_size=0), "burst_size")
    invalid(lambda c: c["experiments"][0].update(slo={"latency": 1}), "slo")
    invalid(lambda c: c["experiments"][0].update(slo={"ttft_s": -1}), "slo")
    invalid(lambda c: c["experiments"][0].update(telemetry="yes"), "telemetry")
    invalid(lambda c: c["experiments"][0].update(workload_spec={"classes": []}), "invalid workload_spec")
    invalid(lambda c: c.update(telemetry={"interval_s": 0}), "telemetry")
    invalid(lambda c: c.update(telemetry={"interval_s": 1, "extra": 1}), "telemetry")


def test_the_concurrency_ceiling_now_covers_the_128_client_sweep_but_not_beyond_256():
    cfg = config()
    cfg["experiments"][0]["concurrency"] = 128
    gpu_bench.validate_config(cfg)
    cfg["experiments"][0]["concurrency"] = 257
    with pytest.raises(ValueError, match="concurrency"):
        gpu_bench.validate_config(cfg)


def test_a_spec_experiments_declared_minutes_drive_the_plan_and_cap_at_max_minutes():
    plan = gpu_bench.plan(config())
    assert {row["id"]: row["estimated_gpu_minutes"] for row in plan["rows"]} == {"sweep-c4": 1.0, "poisson": 1.0}
    cfg = config()
    cfg["experiments"][0]["estimated_minutes"] = 99
    assert gpu_bench.plan(cfg)["rows"][0]["estimated_gpu_minutes"] == 2.0


def test_counter_parsing_and_deltas():
    text = ('# TYPE tensorpilot_jobs counter\ntensorpilot_jobs_total{outcome="completed"} 5.0\n'
            'tensorpilot_jobs_total{outcome="cancelled"} 1.0\ntensorpilot_unrelated_total 9.0\n')
    before = gpu_bench.parse_counters(text)
    after = gpu_bench.parse_counters(text.replace("5.0", "9.0"))
    assert before == {"tensorpilot_jobs_total{outcome=completed}": 5.0, "tensorpilot_jobs_total{outcome=cancelled}": 1.0}
    assert gpu_bench.counter_delta(before, after) == {"tensorpilot_jobs_total{outcome=completed}": 4.0}
    assert gpu_bench.counter_delta(None, after) is None


def test_statistics_caveat_distinguishes_small_and_larger_samples():
    assert "small sample" in gpu_bench.statistics_caveat({"successful_requests": 20})
    assert "small sample" not in gpu_bench.statistics_caveat({"successful_requests": 500})


# ----- the vLLM-side stage: the same harness against an OpenAI-compatible server ---------------------------

class OpenAIServer:
    def __init__(self):
        from test_load_openai import FakeVLLM
        self.fake = FakeVLLM(chunks=6, delay=0.002)
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        self.runner = web.AppRunner(self.fake.app, access_log=None)
        asyncio.run_coroutine_threadsafe(self.runner.setup(), self.loop).result(30)
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        asyncio.run_coroutine_threadsafe(site.start(), self.loop).result(30)
        self.url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
        return self

    def __exit__(self, *exc_info):
        asyncio.run_coroutine_threadsafe(self.runner.cleanup(), self.loop).result(30)
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(10)


def openai_config():
    cfg = config()
    cfg["api"], cfg["model_name"] = "openai", "Qwen/Qwen2.5-7B-Instruct"
    cfg["experiments"] = [cfg["experiments"][0]]
    cfg["experiments"][0]["workload_spec"] = {"seed": 1, "classes": [{"name": "s", "prompt_tokens": 6,
                                                                       "output_tokens": 6}]}
    return cfg


def test_the_same_harness_runs_an_openai_compatible_stage_with_its_own_limitations(tmp_path):
    with OpenAIServer() as server:
        manifest = gpu_bench.run(openai_config(), url=server.url, output_dir=tmp_path / "vllm", config_path="inline.json")
    directory = tmp_path / "vllm"
    assert manifest["api"] == "openai" and manifest["experiments"]["sweep-c4"]["status"] == "completed"
    report = load_report(directory, "sweep-c4", "repeat01_closed.json")
    assert report["summary"]["successful_requests"] > 0 and report["summary"]["p95_queue_wait_s"] is None
    assert report["server_counters_delta"] is None
    assert report["metadata"]["server"]["service"] == "openai-compatible"
    assert any("different engine" in line for line in report["limitations"])
    assert not any("one sequence at a time" in line or "fixed-slot" in line for line in report["limitations"])
    assert server.fake.bodies and all(body["temperature"] == 0 and body["repetition_penalty"] == 1.0
                                      for body in server.fake.bodies)


def test_gpu_only_telemetry_is_collected_for_a_server_without_a_health_document(tmp_path):
    with OpenAIServer() as server:
        gpu_bench.run(openai_config(), url=server.url, output_dir=tmp_path / "vllm", config_path="inline.json")
    report = load_report(tmp_path / "vllm", "sweep-c4", "repeat01_closed.json")
    assert report["telemetry"]["server_samples"] == 0
    overhead = json.loads((tmp_path / "vllm" / "manifest.json").read_text())["experiments"]["sweep-c4"]["telemetry_overhead"]
    assert overhead["server_polls"] == 0 and overhead["server_poll_errors"] == 0


def test_openai_config_requires_a_model_name_and_a_valid_api():
    invalid(lambda c: c.update(api="openai"), "model_name")
    invalid(lambda c: c.update(api="grpc"), "api must be")
    cfg = openai_config()
    gpu_bench.validate_config(cfg)
