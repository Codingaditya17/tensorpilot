import pytest

from tensorpilot.matrix import build_matrix, estimated_kv_bytes, workload_for_case
from tensorpilot.workloads import validate_workload


def test_matrix_covers_required_context_output_width_and_runtime_axes():
    matrix = build_matrix(arrival_rates=(0,))
    cases = matrix["cases"]
    assert {case["input_target_tokens"] for case in cases} == {128, 512, 2048}
    assert {case["output_tokens"] for case in cases} == {128, 256, 512}
    assert {case["max_batch_width"] for case in cases} == {1, 2, 4, 8}
    assert {case["execution"] for case in cases} == {"batch", "adaptive"}
    assert len(cases) == 72


def test_preflight_marks_large_cases_without_confusing_estimate_with_measurement():
    matrix = build_matrix(arrival_rates=(0,), budget_bytes=32 * 1024**2)
    assert any(not case["preflight_safe"] for case in matrix["cases"])
    assert all(case["estimate_kind"].endswith("not_measured_memory") for case in matrix["cases"])
    assert estimated_kv_bytes(128, 128, 2) > 0


def test_case_workload_is_valid_and_records_targets_as_labels():
    case = build_matrix(input_tokens=(128,), output_tokens=(512,), widths=(2,),
                        arrival_rates=(0,))["cases"][0]
    plan = workload_for_case(case, requests=3)
    assert len(validate_workload(plan)) == 3
    assert {item["max_tokens"] for item in plan} == {512}
    assert {item["class"] for item in plan} == {"target-in128-out512"}


def test_matrix_rejects_invalid_safety_fraction():
    with pytest.raises(ValueError):
        build_matrix(safety_fraction=0)
