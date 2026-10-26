"""Fidelity capture and comparison, on tiny models over CPU. The metrics are exercised, not model quality."""
import copy
import json
import math
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("bitsandbytes")

from tiny_model import make_tiny_model_dir  # noqa: E402
from tensorpilot import quality  # noqa: E402
from tensorpilot.cuda_backend import CUDABackend  # noqa: E402

CORPUS = {"a": " ".join(f"w{i % 40}" for i in range(0, 60, 1)), "b": " ".join(f"w{(i * 7) % 50}" for i in range(50))}
PROMPTS = ["hello world the a", "w1 w2 w3 w4 w5", "the the a world hello w9"]


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    return make_tiny_model_dir(tmp_path_factory.mktemp("quality-model"))


def backend(model_dir, mode="none", seed=None):
    loaded = CUDABackend(str(model_dir), device="cpu", allow_cpu=True, dtype="float32", quantization=mode)
    if seed is not None:  # a different model with the same architecture and vocabulary
        from tiny_model import make_tiny_model_dir as rebuild
        import tempfile
        other = CUDABackend(str(rebuild(tempfile.mkdtemp(), seed=seed)), device="cpu", allow_cpu=True, dtype="float32")
        loaded.model = other.model
    return loaded


def take(loaded, **options):
    return quality.capture(loaded, corpus=CORPUS, prompts=PROMPTS, generate_tokens=10, **options)


def test_a_model_compared_with_itself_is_identical(model_dir):
    first = take(backend(model_dir))
    report = quality.compare(first, copy.deepcopy(first))
    assert report["top1_agreement"] == 1.0 and report["top5_overlap"] == 1.0
    assert report["perplexity_ratio_test_over_reference"] == pytest.approx(1.0)
    assert report["kl_reference_to_test_nats_topk_support"] == pytest.approx(0.0, abs=1e-9)
    assert report["greedy_exact_match_fraction"] == 1.0 and report["greedy_mean_prefix_agreement"] == 1.0
    assert report["positions_compared"] > 80 and report["sequences_compared"] == 2


def test_the_metrics_detect_a_different_model(model_dir):
    reference = take(backend(model_dir))
    other = quality.compare(reference, take(backend(model_dir, seed=5)))
    assert other["top1_agreement"] < 1.0 and other["kl_reference_to_test_nats_topk_support"] > 0.01
    assert other["perplexity_ratio_test_over_reference"] != pytest.approx(1.0, abs=1e-3)


def test_nf4_against_its_unquantized_original_uses_exact_reference_logprobs(model_dir):
    reference = take(backend(model_dir))
    test = take(backend(model_dir, "nf4"), reference=reference)
    assert test["quantization"]["format"] == "nf4" and reference["quantization"] is None
    entry = test["teacher_forced"]["a"]
    assert len(entry["logprob_at_reference_topk"]) == len(entry["top1"])
    report = quality.compare(reference, test)
    assert report["kl_is_approximate"] is False
    assert 0 <= report["top1_agreement"] <= 1 and report["kl_reference_to_test_nats_topk_support"] > 0
    assert math.isfinite(report["perplexity_ratio_test_over_reference"])
    assert report["test"]["quantization"]["format"] == "nf4" and report["reference"]["quantization"] is None


def test_reference_logprobs_agree_with_the_models_own_top_k_where_they_overlap(model_dir):
    reference = take(backend(model_dir))
    test = take(backend(model_dir, "nf4"), reference=reference)
    entry, ref_entry = test["teacher_forced"]["b"], reference["teacher_forced"]["b"]
    checked = 0
    for position, ref_ids in enumerate(ref_entry["topk_ids"]):
        own = dict(zip(entry["topk_ids"][position], entry["topk_logprobs"][position]))
        for column, token in enumerate(ref_ids):
            if token in own:
                assert entry["logprob_at_reference_topk"][position][column] == pytest.approx(own[token], abs=1e-5)
                checked += 1
    assert checked > 100


def test_without_reference_logprobs_the_kl_is_flagged_as_approximate(model_dir):
    report = quality.compare(take(backend(model_dir)), take(backend(model_dir, "nf4")))
    assert report["kl_is_approximate"] is True


def synthetic(test_logprobs):
    ids, logprobs = [1, 2], [math.log(.5), math.log(.5)]
    sequence = {"token_ids": [9, 8, 7], "logprob_of_next": [-1.0, -1.0], "top1": [1, 1], "topk_ids": [ids, ids],
                "topk_logprobs": [logprobs, logprobs]}
    return {"schema": quality.SCHEMA, "corpus_sha256": "x", "teacher_forced": {"s": sequence}, "generation": {},
            "test_logprobs": test_logprobs}


def test_kl_matches_a_hand_computed_value():
    reference = synthetic(None)
    test = copy.deepcopy(reference)
    test["teacher_forced"]["s"]["logprob_at_reference_topk"] = [[math.log(.25), math.log(.75)]] * 2
    expected = .5 * math.log(.5 / .25) + .5 * math.log(.5 / .75)  # 0.1438 nats
    report = quality.compare(reference, test)
    assert report["kl_reference_to_test_nats_topk_support"] == pytest.approx(expected, abs=1e-9)
    assert report["kl_is_approximate"] is False


def test_incomparable_captures_are_refused(model_dir):
    capture = take(backend(model_dir))
    other_corpus = copy.deepcopy(capture)
    other_corpus["corpus_sha256"] = "different"
    with pytest.raises(ValueError, match="different corpora"):
        quality.compare(capture, other_corpus)
    shifted = copy.deepcopy(capture)
    shifted["teacher_forced"]["a"]["token_ids"][3] += 1
    with pytest.raises(ValueError, match="token ids differ"):
        quality.compare(capture, shifted)
    with pytest.raises(ValueError, match="tensorpilot.quality/v1"):
        quality.compare({"schema": "other"}, capture)


def test_a_reference_that_tokenized_differently_is_rejected_at_capture_time(model_dir):
    reference = take(backend(model_dir))
    reference["teacher_forced"]["a"]["token_ids"][2] += 1
    with pytest.raises(ValueError, match="tokenized this text differently"):
        take(backend(model_dir, "nf4"), reference=reference)


def test_greedy_divergence_is_reported_as_a_prefix_not_just_a_mismatch():
    base = synthetic(None)
    base["generation"] = {"p": {"prompt": "x", "token_ids": [1, 2, 3, 4, 5, 6, 7, 8]}}
    test = copy.deepcopy(base)
    test["generation"]["p"]["token_ids"] = [1, 2, 3, 4, 9, 9, 9, 9]
    report = quality.compare(base, test)
    assert report["greedy_exact_match_fraction"] == 0 and report["greedy_mean_prefix_agreement"] == .5
    assert report["greedy_first_divergence_index_mean"] == 4


def test_the_command_line_captures_and_compares(model_dir, tmp_path, monkeypatch, capsys):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("\n\n".join(CORPUS.values()))
    base = ["tensorpilot-quality", "capture", "--model", str(model_dir), "--device", "cpu", "--allow-cpu",
            "--dtype", "float32", "--corpus-file", str(corpus), "--generate-tokens", "6"]
    ref, nf4 = tmp_path / "ref.json", tmp_path / "nf4.json"
    for argv in (base + ["--output", str(ref)],
                 base + ["--quantization", "nf4", "--reference", str(ref), "--output", str(nf4)]):
        monkeypatch.setattr(sys, "argv", argv)
        assert quality.main() == 0
        assert "saved" in capsys.readouterr().out
    monkeypatch.setattr(sys, "argv", ["tensorpilot-quality", "compare", str(ref), str(nf4)])
    assert quality.main() == 0
    report = json.loads(capsys.readouterr().out)
    assert report["kl_is_approximate"] is False and report["sequences_compared"] == 2
    assert report["test"]["quantization"]["format"] == "nf4"
