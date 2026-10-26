"""Integrity of the v1.0 evaluation suite: generated files, workload parity, and fit on the target card."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tensorpilot import gpu_bench, suite
from tensorpilot.model_plan import assess
from tensorpilot.workloads import build_spec_workload

GPU_DIR = Path(__file__).resolve().parents[1] / "benchmarks" / "gpu"
RTX_4090 = 25252724736
CONFIGS = {stage: suite.config_for(stage) for stage in suite.stage_names()}


def model_config(key):
    if key == "14b":
        return SimpleNamespace(model_type="qwen2", hidden_size=5120, intermediate_size=13824, num_hidden_layers=48,
                               num_attention_heads=40, num_key_value_heads=8, vocab_size=152064,
                               tie_word_embeddings=False, max_position_embeddings=32768, use_sliding_window=False,
                               layer_types=["full_attention"] * 48)
    if key == "7b":
        return SimpleNamespace(model_type="qwen2", hidden_size=3584, intermediate_size=18944, num_hidden_layers=28,
                               num_attention_heads=28, num_key_value_heads=4, vocab_size=152064,
                               tie_word_embeddings=False, max_position_embeddings=32768, use_sliding_window=False,
                               layer_types=["full_attention"] * 28)
    return SimpleNamespace(model_type="qwen2", hidden_size=896, intermediate_size=4864, num_hidden_layers=24,
                           num_attention_heads=14, num_key_value_heads=2, vocab_size=151936,
                           tie_word_embeddings=True, max_position_embeddings=32768, use_sliding_window=False,
                           layer_types=["full_attention"] * 24)


DOWNLOAD_BYTES = {"0.5b": 988097824, "7b": int(15.23e9), "14b": int(29.54e9)}


@pytest.mark.parametrize("stage", suite.stage_names())
def test_the_committed_config_is_exactly_what_the_generator_produces(stage):
    committed = (GPU_DIR / f"v10-{stage}.json").read_text()
    assert committed == suite.render(CONFIGS[stage]), "run: python -m tensorpilot.suite --write benchmarks/gpu"


@pytest.mark.parametrize("stage", suite.stage_names())
def test_every_stage_is_a_valid_bounded_experiment_config(stage):
    config = CONFIGS[stage]
    gpu_bench.validate_config(config)
    ids = [e["id"] for e in config["experiments"]]
    assert len(ids) == len(set(ids)) and ids[0] == "cold-start"
    for experiment in config["experiments"]:
        assert 0 < experiment["estimated_minutes"] <= experiment["max_minutes"] <= 240, experiment["id"]
        assert experiment["timeout_s"] <= 120
    assert config["assumptions"]["status"] == "placeholder_unmeasured"
    assert gpu_bench.plan(config)["within_budget"]
    assert config["max_session_minutes"] == sum(e["max_minutes"] for e in config["experiments"]) + 5


@pytest.mark.parametrize("stage", ["0.5b-batch", "7b-bf16-batch", "14b-nf4-batch", "vllm-7b-bf16", "vllm-14b-nf4"])
def test_the_concurrency_sweep_covers_1_to_128_clients(stage):
    sweep = {e["concurrency"]: e for e in CONFIGS[stage]["experiments"] if e["id"].startswith("sweep-c")}
    assert sorted(sweep) == [1, 8, 16, 32, 64, 128]
    assert sweep[128]["requests"] >= 600 and sweep[1]["requests"] <= 100, "c=1 is a small sample, and is labeled so"
    assert all(e["workload_spec"] == suite.MIXED for e in sweep.values())


@pytest.mark.parametrize("tp_stage, vllm_stage", [("7b-bf16-batch", "vllm-7b-bf16"), ("14b-nf4-batch", "vllm-14b-nf4")])
def test_vllm_runs_the_identical_workloads_as_tensorpilot(tp_stage, vllm_stage):
    ours = {e["id"]: e for e in CONFIGS[tp_stage]["experiments"]}
    theirs = {e["id"]: e for e in CONFIGS[vllm_stage]["experiments"]}
    assert theirs and set(theirs) <= set(ours)
    comparable = ("kind", "requests", "concurrency", "repeats", "rates", "arrival", "burst_size", "workload_spec",
                  "timeout_s", "slo", "warmup_requests")
    for name, experiment in theirs.items():
        for field in comparable:
            assert experiment.get(field) == ours[name].get(field), f"{name}.{field} differs between engines"
    assert CONFIGS[vllm_stage]["target"]["model"] == CONFIGS[tp_stage]["target"]["model"]
    assert CONFIGS[vllm_stage]["target"]["revision"] == CONFIGS[tp_stage]["target"]["revision"]
    assert CONFIGS[vllm_stage]["api"] == "openai" and "api" not in CONFIGS[tp_stage]


def test_only_tensorpilot_runs_the_experiments_that_need_its_admission_and_cancellation_internals():
    extras = {"cancellation", "timeouts", "overload", "recovery-after-overload"}
    assert extras <= {e["id"] for e in CONFIGS["7b-bf16-batch"]["experiments"]}
    assert not extras & {e["id"] for e in CONFIGS["vllm-7b-bf16"]["experiments"]}


def max_prompt_tokens(workload):
    best = 0
    for entry in workload["classes"]:
        distribution = entry["prompt_tokens"]
        if isinstance(distribution, int):
            best = max(best, distribution)
        elif distribution["dist"] == "uniform":
            best = max(best, distribution["max"])
        elif distribution["dist"] == "choice":
            best = max(best, max(distribution["values"]))
    return best


@pytest.mark.parametrize("stage", suite.stage_names())
def test_no_prompt_exceeds_the_stages_configured_limit_and_outputs_stay_within_the_cap(stage):
    config = CONFIGS[stage]
    limit = config["target"]["max_prompt_tokens"]
    for experiment in config["experiments"]:
        assert max_prompt_tokens(experiment["workload_spec"]) <= limit, f"{experiment['id']} overruns the prompt cap"
        plan = build_spec_workload(experiment["workload_spec"], 400)
        assert max(item["max_tokens"] for item in plan) <= 512
        assert max(item["prompt_tokens"] for item in plan) <= limit


@pytest.mark.parametrize("stage", [s for s in suite.stage_names() if suite.STAGES[s]["engine"] == "tensorpilot"
                                   and suite.STAGES[s]["execution"] == "batch"])
def test_every_batching_stage_has_a_kv_pool_that_fits_the_4090(stage):
    parameters = suite.STAGES[stage]
    key = parameters["model"]
    verdict = assess(model_config(key), repo=suite.MODELS[key]["repo"], revision="r" * 40, license="apache-2.0",
                     weights_bytes=DOWNLOAD_BYTES[key], gpu_total_bytes=RTX_4090, dtype="bfloat16",
                     prompt_tokens=parameters["max_prompt_tokens"], output_tokens=512,
                     requested_capacity=parameters["batch_size"], quantization=parameters["quantization"])
    assert verdict["verdict"] == "fits", verdict.get("reason")
    assert verdict["rows_granted"] >= min(parameters["batch_size"], 12), \
        f"{stage}: only {verdict['rows_granted']} slots fit; the stage's concurrency points would be meaningless"


def test_the_14b_stage_requests_more_slots_than_it_can_hold_and_the_suite_says_so():
    stage = suite.STAGES["14b-nf4-batch"]
    verdict = assess(model_config("14b"), repo="m", revision="r" * 40, license="x", weights_bytes=DOWNLOAD_BYTES["14b"],
                     gpu_total_bytes=RTX_4090, dtype="bfloat16", prompt_tokens=stage["max_prompt_tokens"],
                     output_tokens=512, requested_capacity=stage["batch_size"], quantization="nf4")
    assert verdict["rows_granted"] < stage["batch_size"] and verdict["pool_limited_by"] == "vram"
    assert 20 <= verdict["rows_granted"] <= 40


def test_slo_thresholds_scale_with_model_size_and_the_overload_run_has_none():
    slos = {k: suite.MODELS[k]["slo"] for k in suite.MODELS}
    assert slos["0.5b"]["e2e_s"] < slos["7b"]["e2e_s"] < slos["14b"]["e2e_s"]
    overload = next(e for e in CONFIGS["7b-bf16-batch"]["experiments"] if e["id"] == "overload")
    assert "slo" not in overload and overload["abort_error_rate"] == 1.0 and overload["rates"] == [400]


def test_the_telemetry_overhead_pair_is_identical_except_for_the_sampler():
    experiments = {e["id"]: e for e in CONFIGS["7b-bf16-batch"]["experiments"]}
    on, off = experiments["overhead-telemetry-on"], experiments["overhead-telemetry-off"]
    assert off["telemetry"] is False and "telemetry" not in on
    differing = {k for k in on.keys() | off.keys() if on.get(k) != off.get(k)}
    assert differing == {"id", "description", "telemetry"}


def test_budgets_are_stated_with_caps_far_above_expectations_but_bounded():
    total_expected = sum(c["expected_experiment_minutes"] for c in CONFIGS.values())
    total_cap = sum(c["max_session_minutes"] - 5 for c in CONFIGS.values())
    assert 200 < total_expected < 400 and total_expected < total_cap
    assert "TOTAL" in suite.plan_text()


def test_the_command_line_regenerates_identical_files(tmp_path, monkeypatch, capsys):
    import sys
    monkeypatch.setattr(sys, "argv", ["suite", "--write", str(tmp_path)])
    assert suite.main() == 0
    for stage in suite.stage_names():
        assert (tmp_path / f"v10-{stage}.json").read_text() == (GPU_DIR / f"v10-{stage}.json").read_text()
    assert json.loads((tmp_path / "v10-7b-bf16-batch.json").read_text())["target"]["quantization"] == "none"


@pytest.mark.parametrize("stage", suite.stage_names())
def test_realized_prompts_match_what_each_experiment_declares_so_nothing_is_silently_shortened(stage):
    """Fitting under a limit is not enough: a clamped plan would still fit while misreporting its own contents."""
    for experiment in CONFIGS[stage]["experiments"]:
        plan = build_spec_workload(experiment["workload_spec"], experiment["requests"])
        for entry in experiment["workload_spec"]["classes"]:
            declared = entry["prompt_tokens"]
            realized = {item["prompt_tokens"] for item in plan if item["class"] == entry["name"]}
            if not realized:
                continue
            if isinstance(declared, int):
                assert realized == {declared}, experiment["id"]
            elif declared["dist"] == "choice":
                assert realized <= set(declared["values"]) and len(realized) == len(set(declared["values"])), experiment["id"]
            elif declared["dist"] == "uniform":
                assert declared["min"] <= min(realized) and max(realized) <= declared["max"], experiment["id"]
        assert all(len(item["prompt"].split()) == item["prompt_tokens"] for item in plan), experiment["id"]
