import random

import pytest

from tensorpilot.workloads import (MAX_PROMPT_TOKENS, SINGLE_TOKEN_WORDS, build_spec_workload, fingerprint,
                                   prompt_of_tokens, sample, validate_workload)

SPEC = {"seed": 5,
        "classes": [{"name": "short", "weight": 3, "prompt_tokens": {"dist": "uniform", "min": 8, "max": 64},
                     "output_tokens": {"dist": "choice", "values": [16, 32], "weights": [1, 1]}},
                    {"name": "long", "weight": 1, "prompt_tokens": 512, "output_tokens": 128}],
        "cancel": {"fraction": .25, "after_tokens": {"dist": "uniform", "min": 2, "max": 6}},
        "timeout_s": 30}


def test_a_spec_and_seed_give_the_same_plan_every_time():
    first, second = build_spec_workload(SPEC, 300), build_spec_workload(SPEC, 300)
    assert first == second and fingerprint(first) == fingerprint(second)
    assert fingerprint(build_spec_workload({**SPEC, "seed": 6}, 300)) != fingerprint(first)


def test_the_plan_honors_class_mix_lengths_cancellation_and_validates():
    plan = validate_workload(build_spec_workload(SPEC, 2000))
    shorts = [item for item in plan if item["class"] == "short"]
    assert 0.70 < len(shorts) / len(plan) < 0.80
    assert all(8 <= item["prompt_tokens"] <= 64 and item["max_tokens"] in (16, 32) for item in shorts)
    assert all(item["prompt_tokens"] == 512 and item["max_tokens"] == 128 for item in plan if item["class"] == "long")
    cancelled = [item for item in plan if "cancel_after_tokens" in item]
    assert 0.2 < len(cancelled) / len(plan) < 0.3 and all(2 <= item["cancel_after_tokens"] <= 6 for item in cancelled)
    assert all(item["timeout_s"] == 30 for item in plan)


def test_a_prompt_has_exactly_the_requested_number_of_words():
    assert len(prompt_of_tokens(37, random.Random(1)).split()) == 37
    assert prompt_of_tokens(1, random.Random(1)).startswith(" ")
    for bad in (0, MAX_PROMPT_TOKENS + 1, 2.5):
        with pytest.raises(ValueError):
            prompt_of_tokens(bad, random.Random(1))


def test_the_longest_prompt_stays_under_the_service_byte_limit():
    assert len(prompt_of_tokens(MAX_PROMPT_TOKENS, random.Random(1)).encode()) <= 16384


def test_each_word_is_one_qwen_token_so_prompt_length_is_exact():
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", local_files_only=True,
                                                  revision="7ae557604adf67be50417f59c2c2f167def9a775")
    except Exception:
        pytest.skip("Qwen tokenizer is not in the local Hugging Face cache")
    assert all(len(tokenizer.encode(" " + word)) == 1 for word in SINGLE_TOKEN_WORDS)
    for count in (1, 17, 256, 1024, 2048):
        assert len(tokenizer.encode(prompt_of_tokens(count, random.Random(count)))) == count


@pytest.mark.parametrize("distribution", [{"dist": "zipf"}, {"dist": "uniform", "min": 5, "max": 1},
                                          {"dist": "choice", "values": []},
                                          {"dist": "choice", "values": [1, 2], "weights": [1]}, "text"])
def test_malformed_distributions_are_rejected(distribution):
    with pytest.raises(ValueError):
        sample(distribution, random.Random(0))


def test_lognormal_is_clipped_to_its_bounds():
    rng = random.Random(2)
    values = [sample({"dist": "lognormal", "median": 100, "sigma": 1.5, "min": 20, "max": 300}, rng)
              for _ in range(500)]
    assert min(values) >= 20 and max(values) <= 300 and len(set(values)) > 50


@pytest.mark.parametrize("spec, requests", [({}, 5), ({"classes": []}, 5), (SPEC, 0), (SPEC, 100001),
                                            ({"classes": [{"name": "a", "prompt_tokens": 5, "output_tokens": 5},
                                                          {"name": "a", "prompt_tokens": 5, "output_tokens": 5}]}, 5)])
def test_malformed_specs_are_rejected(spec, requests):
    with pytest.raises(ValueError):
        build_spec_workload(spec, requests)


def test_validation_accepts_optional_fields_and_rejects_bad_ones():
    base = {"prompt": " hello", "max_tokens": 4, "class": "c"}
    assert validate_workload([{**base, "cancel_after_s": .5, "timeout_s": 3, "prompt_tokens": 1}])
    for bad in ({"cancel_after_tokens": 0}, {"timeout_s": -1}, {"cancel_after_s": float("nan")},
                {"surprise": 1}, {"prompt_tokens": 1.5}):
        with pytest.raises(ValueError):
            validate_workload([{**base, **bad}])
