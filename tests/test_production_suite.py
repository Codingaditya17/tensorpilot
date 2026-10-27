"""The production-scale plan keeps the promises made about it: levels, volume, parity between engines, and the budget."""
import pytest

from tensorpilot import suite

LEVELS = (1, 32, 64, 128, 256, 512, 1024)
PROD = ["7b-bf16-prod-serial", "7b-bf16-prod", "14b-nf4-prod", "vllm-7b-bf16-prod", "vllm-14b-nf4-prod"]


def experiments(stage):
    return {e["id"]: e for e in suite.config_for(stage)["experiments"]}


@pytest.mark.parametrize("stage", ["7b-bf16-prod", "14b-nf4-prod", "vllm-7b-bf16-prod", "vllm-14b-nf4-prod"])
def test_every_batching_stage_sweeps_all_seven_concurrency_levels(stage):
    found = experiments(stage)
    for level in LEVELS:
        assert found[f"scale-c{level}"]["concurrency"] == level
        assert found[f"scale-c{level}"]["requests"] >= level, "every client must get at least one request"


def test_the_7b_sweep_alone_is_at_least_ten_thousand_requests_and_nothing_is_hidden_by_the_abort_rule():
    sweep = [e for e in experiments("7b-bf16-prod").values() if e["id"].startswith("scale-c")]
    assert sum(e["requests"] * e["repeats"] for e in sweep) >= 10_000
    assert all(e["abort_error_rate"] >= 0.9 for e in sweep), "saturation must be measured, not aborted early"


def test_a_closed_loop_point_with_more_clients_than_slots_is_admitted_not_rejected_by_construction():
    env = suite.stage_env("7b-bf16-prod")
    assert int(env["V10_QUEUE_SIZE"]) >= 1024, "1,024 clients must fit the queue so the sweep measures latency, not rejection"
    assert int(env["V10_BACKLOG"]) >= 1024


@pytest.mark.parametrize("pair", [("7b-bf16-prod-serial", "7b-bf16-prod"), ("7b-bf16-prod", "vllm-7b-bf16-prod"),
                                  ("14b-nf4-prod", "vllm-14b-nf4-prod")])
def test_comparison_experiments_are_identical_across_engines(pair):
    a, b = experiments(pair[0]), experiments(pair[1])
    shared = [name for name in a if name.startswith("cmp-") and name in b]
    assert {"cmp-c1", "cmp-c8", "cmp-c32"} <= set(shared)
    for name in shared:
        for key in ("workload_spec", "requests", "concurrency", "repeats", "warmup_requests", "timeout_s", "slo"):
            assert a[name][key] == b[name][key], f"{name}.{key} differs between {pair}"
        assert a[name]["repeats"] >= 3, "the vLLM comparison needs at least three measured repetitions"


def test_scale_and_ladder_experiments_are_identical_between_tensorpilot_and_vllm():
    for tp, vl in (("7b-bf16-prod", "vllm-7b-bf16-prod"), ("14b-nf4-prod", "vllm-14b-nf4-prod")):
        a, b = experiments(tp), experiments(vl)
        planning_only = ("estimated_minutes", "max_minutes", "description")
        for name in [n for n in a if n.startswith(("scale-c", "ladder", "sustained"))]:
            assert {k: v for k, v in a[name].items() if k not in planning_only} == \
                   {k: v for k, v in b[name].items() if k not in planning_only}, name


def test_vllm_stages_run_only_the_workloads_that_make_sense_for_a_foreign_engine():
    for stage in ("vllm-7b-bf16-prod", "vllm-14b-nf4-prod"):
        assert not {"cancellation", "timeouts", "overload", "recovery-after-overload"} & set(experiments(stage))


def test_the_7b_stage_carries_the_safety_experiments_and_the_14b_stage_does_not_repeat_them():
    assert {"cancellation", "timeouts", "overload", "recovery-after-overload"} <= set(experiments("7b-bf16-prod"))
    assert not {"overload", "cancellation"} & set(experiments("14b-nf4-prod"))


def test_slos_are_declared_per_model_and_match_the_stated_targets():
    assert suite.MODELS["7b"]["slo"] == {"ttft_s": 2.0, "e2e_s": 20.0, "tpot_s": 0.15}
    for stage in PROD:
        for e in experiments(stage).values():
            if "slo" in e:
                assert e["slo"] == suite.MODELS[suite.STAGES[stage]["model"]]["slo"]


def test_packing_is_on_for_the_batching_stages_and_off_for_serial_and_vllm_stage_environments():
    assert suite.stage_env("7b-bf16-prod")["V10_PREFILL_PACK"] == suite.PREFILL_PACK_TOKENS > 0
    assert suite.stage_env("7b-bf16-prod-serial")["V10_PREFILL_PACK"] == 0
    assert suite.stage_env("0.5b-quick-batch")["V10_PREFILL_PACK"] == 0, "the A/B baseline must be the unpacked path"
    assert suite.stage_env("0.5b-quick-batch-packed")["V10_PREFILL_PACK"] > 0
    assert suite.stage_env("probe-7b")["V10_ENGINE"] == "probe"


def test_the_packed_ab_stage_runs_the_same_experiments_as_its_baseline():
    base, packed = experiments("0.5b-quick-batch"), experiments("0.5b-quick-batch-packed")
    assert base == packed


def test_the_core_plan_fits_nine_dollars_in_the_worst_case_at_the_assumed_price():
    plan = suite.budget_plan()
    assert plan["fits_budget_worst_case"] and plan["worst_total_usd"] <= 9.0
    assert plan["every_expected_stage_can_start"]
    assert plan["price_is_assumption"] is True
    assert plan["expected_total_usd"] < plan["worst_total_usd"]


def test_a_dearer_pod_is_scaled_down_or_flagged_never_silently_over_budget():
    plan = suite.budget_plan(price_per_hour=1.80)
    assert plan["worst_total_usd"] <= 9.0 + 1e-6 or not plan["fits_budget_worst_case"]
    assert plan["scaled_to_fit"], "the plan must say it was squeezed"
    assert not plan["every_expected_stage_can_start"], "squeezing must be reported as stages the ledger would refuse"


def test_the_14b_vllm_run_is_an_optional_session_outside_the_nine_dollars():
    plan = suite.budget_plan()
    assert "vllm-14b-nf4-prod" not in sum((s["stages"] for s in plan["sessions"].values()), [])
    assert "14b-vllm" in plan["optional_sessions"]


def test_total_planned_requests_clear_ten_thousand_by_a_wide_margin():
    total = 0
    for stage in ("7b-bf16-prod", "14b-nf4-prod", "7b-bf16-prod-serial"):
        for e in suite.config_for(stage)["experiments"]:
            total += e["requests"] * e.get("repeats", 1) * (len(e["rates"]) if "rates" in e else 1)
    assert total > 15_000


# ----- the lean re-run of the vLLM comparison (the 7B session's vLLM stage failed to start) -----

LEAN = [("7b-bf16-cmp", "7b-bf16-prod"), ("vllm-7b-bf16-cmp", "vllm-7b-bf16-prod")]


@pytest.mark.parametrize("lean, parent", LEAN)
def test_a_lean_stage_copies_the_prod_experiments_unchanged(lean, parent):
    """Identical definitions mean identical workload hashes, so archived -prod runs and re-run lean runs pair directly."""
    full = {e["id"]: e for e in suite.config_for(parent)["experiments"]}
    kept = suite.config_for(lean)["experiments"]
    assert kept and all(e == full[e["id"]] for e in kept)
    assert suite.config_for(lean)["target"] == {**suite.config_for(parent)["target"]}


@pytest.mark.parametrize("lean", ["7b-bf16-cmp", "vllm-7b-bf16-cmp"])
def test_a_lean_stage_has_every_comparison_point_the_acceptance_criteria_name_with_enough_repeats(lean):
    import json
    from pathlib import Path
    criteria = json.loads((Path(__file__).resolve().parents[1] / "benchmarks" / "acceptance" / "criteria.json").read_text())
    by_id = {e["id"]: e for e in suite.config_for(lean)["experiments"]}
    for point in criteria["vllm"]["points"]:
        assert point in by_id and by_id[point].get("repeats", 1) >= criteria["vllm"]["min_repeats"], point


def test_the_lean_vllm_stage_adds_the_plateau_points_and_nothing_tensorpilot_only():
    ids = {e["id"] for e in suite.config_for("vllm-7b-bf16-cmp")["experiments"]}
    assert {"scale-c64", "scale-c256", "scale-c1024"} <= ids
    assert not ids & {"cancellation", "timeouts", "overload", "recovery-after-overload"}
    assert suite.stage_env("vllm-7b-bf16-cmp")["V10_ENGINE"] == "vllm"
    assert suite.STAGES["vllm-7b-bf16-cmp"]["batch_size"] == suite.STAGES["7b-bf16-prod"]["batch_size"], "same 48 sequences"


def test_the_rerun_session_is_cheap_enough_to_fit_a_three_dollar_balance_at_the_measured_price():
    plan = suite.budget_plan(0.91, 3.0, suite.RERUN_SESSIONS)
    row = plan["sessions"]["7b-vllm-rerun"]
    assert row["stages"] == ["vllm-7b-bf16-cmp", "7b-bf16-cmp", "quality-7b"], "vLLM first: it is the one that failed"
    assert plan["fits_budget_worst_case"] and plan["every_expected_stage_can_start"]
    assert row["worst_cost_usd"] < 2.2 and row["expected_cost_usd"] < 1.6


def test_the_core_sessions_are_unchanged_by_the_rerun_definitions():
    assert set(suite.SESSIONS) == {"gate", "7b", "14b"}
