"""Batch invariance measurement, including a worker that is deliberately wrong only when requests share a batch."""
import asyncio
import json
import sys

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from tiny_model import make_tiny_model_dir  # noqa: E402
from test_gpu_bench_scale import BatchWorker  # noqa: E402
from tensorpilot import batch_invariance  # noqa: E402
from tensorpilot.server import ENGINE  # noqa: E402

TINY = {"seed": 4, "classes": [{"name": "short", "prompt_tokens": {"dist": "uniform", "min": 4, "max": 20},
                                "output_tokens": {"dist": "uniform", "min": 6, "max": 14}}]}


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    return make_tiny_model_dir(tmp_path_factory.mktemp("invariance-model"))


def measure(worker, tmp_path, prompts=12):
    return asyncio.run(batch_invariance.measure(worker.url, prompts=prompts, timeout=60, workdir=tmp_path, spec=TINY,
                                                head_start_s=0.05))


def test_a_correct_cpu_worker_gives_every_request_the_same_tokens_however_it_is_batched(model_dir, tmp_path):
    with BatchWorker(model_dir) as worker:
        report = measure(worker, tmp_path)
    for name in ("together_vs_solo", "staggered_vs_solo"):
        result = report[name]
        assert result["comparable"] == 12 and result["exact_matches"] == 12, (name, result["diverged_requests"])
        assert result["exact_match_fraction"] == 1.0 and result["earliest_divergence"] is None
    assert report["solo_summary"]["failed_requests"] == 0 and report["together_summary"]["failed_requests"] == 0
    assert report["schema"] == batch_invariance.SCHEMA and report["reading_guide"]


def test_a_worker_that_is_wrong_only_when_co_batched_is_caught_near_the_start_of_the_sequence(model_dir, tmp_path):
    with BatchWorker(model_dir) as worker:
        adapter = worker.runner.app[ENGINE].adapter
        original = adapter._forward

        def corrupt_when_shared(input_ids, positions, mask, write, kv_rows, width, last_only):
            logits = original(input_ids, positions, mask, write, kv_rows, width, last_only)
            if input_ids.shape[0] > 1 and input_ids.shape[1] == 1:  # a decode step shared by several sequences
                logits[0, -1, 7] += 1000.0                          # row 0 is forced onto token 7
            return logits

        adapter._forward = corrupt_when_shared
        report = measure(worker, tmp_path)
    shared = report["together_vs_solo"]
    assert shared["diverged"] >= 1 and shared["exact_match_fraction"] < 1.0
    assert shared["earliest_divergence"] <= 2, "a real defect shows at the start, not as a late near-tie"
    assert report["staggered_vs_solo"]["diverged"] >= 1


def row(tokens, ok=True):
    return {"ok": ok, "token_ids": tokens}


def test_comparison_reports_exact_prefix_and_first_divergence():
    reference = {0: row([1, 2, 3, 4]), 1: row([5, 6, 7, 8]), 2: row([9, 9, 9]), 3: row([1], ok=False)}
    other = {0: row([1, 2, 3, 4]), 1: row([5, 6, 0, 0]), 2: row([9, 9, 9, 9]), 3: row([1])}
    result = batch_invariance.compare_runs(reference, other)
    assert result["requests"] == 4 and result["comparable"] == 3 and result["not_comparable"] == 1
    assert result["exact_matches"] == 1 and result["diverged"] == 2
    assert result["earliest_divergence"] == 2 and result["median_divergence"] in (2, 3)
    assert {r["index"] for r in result["diverged_requests"]} == {1, 2}
    assert result["mean_prefix_fraction"] == pytest.approx((1.0 + 0.5 + 0.75) / 3)


def test_a_missing_request_is_not_silently_counted_as_a_match():
    result = batch_invariance.compare_runs({0: row([1, 2])}, {})
    assert result["comparable"] == 0 and result["not_comparable"] == 1 and result["exact_match_fraction"] is None


def test_the_command_line_writes_the_report_and_validates_its_arguments(model_dir, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["inv", "--url", "http://127.0.0.1:1", "--prompts", "2", "--output", "x.json"])
    with pytest.raises(SystemExit):
        batch_invariance.main()
    monkeypatch.setattr(batch_invariance, "SPEC", TINY)
    with BatchWorker(model_dir) as worker:
        output = tmp_path / "invariance.json"
        monkeypatch.setattr(sys, "argv", ["inv", "--url", worker.url, "--prompts", "8", "--output", str(output)])
        assert batch_invariance.main() == 0
    saved = json.loads(output.read_text())
    assert saved["together_vs_solo"]["exact_matches"] == 8
    assert "identical to solo" in capsys.readouterr().out
