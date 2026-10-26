"""The go/no-go for packed prefill, decided from two stage directories by a rule fixed in advance."""
import itertools
import json

import pytest

from tensorpilot import packing_gate


def stage(tmp_path, name, *, tps, ttft, failed=0, invariance=0.96875, gate=True, interrupted=False, status="completed",
          skip=()):
    directory = tmp_path / name
    bench = directory / "bench"
    directory.mkdir(parents=True, exist_ok=True)
    experiments = {}
    for point in ("quick-c1", "quick-c8", "quick-c32", "quick-c128"):
        if point in skip:
            continue
        folder = bench / point
        folder.mkdir(parents=True)
        summary = {"completed_output_tokens": tps[point] * 10, "wall_s": 10.0, "requests_total": 20, "successful_requests": 20 - failed,
                   "failed_requests": failed, "p95_client_ttft_s": ttft[point]}
        (folder / "repeat01_closed.json").write_text(json.dumps({"summary": summary}))
        experiments[point] = {"status": status}
    (bench / "manifest.json").write_text(json.dumps({"experiments": experiments, "interrupted": interrupted}))
    if invariance is not None:
        entry = {"exact_match_fraction": invariance}
        (directory / "invariance.json").write_text(json.dumps({"together_vs_solo": entry, "staggered_vs_solo": entry}))
    (directory / "memory-gate.json").write_text(json.dumps({"ok": gate, "checks": [{"detail": "ok"}]}))
    return directory


_runs = itertools.count()
BASE_TPS = {"quick-c1": 50.0, "quick-c8": 280.0, "quick-c32": 640.0, "quick-c128": 760.0}
BASE_TTFT = {"quick-c1": 0.03, "quick-c8": 0.2, "quick-c32": 1.2, "quick-c128": 5.4}


def faster(factor):
    return {k: v * (factor if k in ("quick-c32", "quick-c128") else 1.0) for k, v in BASE_TPS.items()}


def decide(tmp_path, **packed):
    run = tmp_path / f"run{next(_runs)}"
    a = stage(run, "u", tps=BASE_TPS, ttft=BASE_TTFT)
    kwargs = {"tps": BASE_TPS, "ttft": BASE_TTFT, **packed}
    b = stage(run, "p", **kwargs)
    from tensorpilot.quick_compare import stage_facts
    return packing_gate.judge(stage_facts(a), stage_facts(b))


def test_a_clear_gain_with_no_regression_means_use_packing(tmp_path):
    verdict = decide(tmp_path, tps=faster(1.4), ttft={**BASE_TTFT, "quick-c128": 2.0})
    assert verdict["decision"] == "USE_PACKING" and verdict["helped_at_c128"] and verdict["no_regression"]


def test_no_measurable_gain_means_run_unpacked_because_an_unmeasured_path_has_no_reason_to_ride_along(tmp_path):
    verdict = decide(tmp_path, tps=faster(1.01))
    assert verdict["decision"] == "NO_GAIN" and verdict["run_7b_stages_with"] == "PREFILL_PACK_TOKENS=0"


def test_a_ttft_gain_alone_counts_as_a_gain(tmp_path):
    assert decide(tmp_path, ttft={**BASE_TTFT, "quick-c128": 4.0})["decision"] == "USE_PACKING"


@pytest.mark.parametrize("change", [
    {"tps": {**BASE_TPS, "quick-c8": 200.0}},                                   # slower at c=8
    {"ttft": {**BASE_TTFT, "quick-c32": 1.6}},                                  # TTFT regression at c=32
    {"failed": 1},                                                              # any failed request
    {"invariance": 0.9},                                                        # below the floor
    {"invariance": 0.9, "tps": faster(2.0)},                                    # a gain does not excuse it
    {"gate": False}, {"interrupted": True}, {"status": "aborted"}, {"skip": ("quick-c128",)}, {"invariance": None}])
def test_any_regression_or_missing_evidence_means_run_unpacked(tmp_path, change):
    if "tps" in change and change["tps"].get("quick-c32") == BASE_TPS["quick-c32"] * 2.0:
        change = {**change, "ttft": BASE_TTFT}
    verdict = decide(tmp_path, **change)
    assert verdict["decision"] == "REGRESSION" and verdict["run_7b_stages_with"] == "PREFILL_PACK_TOKENS=0", verdict["checks"]


def test_one_request_fewer_identical_is_tolerated_but_not_two(tmp_path):
    assert decide(tmp_path, invariance=0.96875 - 1 / 32, tps=faster(1.4))["decision"] == "USE_PACKING"
    assert decide(tmp_path, invariance=0.96875 - 2 / 32, tps=faster(1.4))["decision"] == "REGRESSION"


def test_the_command_line_prints_the_decision_and_writes_json(tmp_path, monkeypatch, capsys):
    import sys
    a = stage(tmp_path, "u", tps=BASE_TPS, ttft=BASE_TTFT)
    b = stage(tmp_path, "p", tps=faster(1.5), ttft={**BASE_TTFT, "quick-c128": 2.0})
    monkeypatch.setattr(sys, "argv", ["packing_gate", str(a), str(b), "--output", str(tmp_path / "v.json")])
    assert packing_gate.main() == 0
    assert "PACKING GATE: USE_PACKING" in capsys.readouterr().out
    assert json.loads((tmp_path / "v.json").read_text())["schema"] == "tensorpilot.packing-verdict/v1"
