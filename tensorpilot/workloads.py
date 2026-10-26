"""Deterministic workload plans and portable content fingerprints."""
import hashlib
import json
import math
import random

from .bench import PROMPTS


def fingerprint(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_workload(requests, max_tokens=64, *, kind="uniform", seed=42):
    if type(requests) is not int or requests < 1:
        raise ValueError("requests must be positive")
    if type(max_tokens) is not int or max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    if kind not in ("uniform", "mixed"):
        raise ValueError("workload must be uniform or mixed")
    if kind == "uniform":
        return [{"prompt": PROMPTS[index % len(PROMPTS)], "max_tokens": max_tokens,
                 "class": "uniform"} for index in range(requests)]
    # Repeatable 60/20/20 mix. Shuffle composition rather than sampling it, so
    # repeated runs cannot accidentally compare different class proportions.
    classes = ["short", "short", "short", "medium", "long"]
    labels = [classes[index % len(classes)] for index in range(requests)]
    random.Random(seed).shuffle(labels)
    budgets = {"short": 16, "medium": 64, "long": 128}
    prompts = {
        "short": "Name one reason a bounded inference queue is useful. Be concise.",
        "medium": "Explain the tradeoffs between throughput and latency in GPU model serving.",
        "long": "Give a detailed numbered explanation of admission control, FIFO scheduling, "
                "head-of-line blocking, fairness, deadlines and overload recovery in model serving.",
    }
    return [{"prompt": prompts[label], "max_tokens": budgets[label], "class": label} for label in labels]


def build_stress_workload(requests, *, prompt_repeats=32, output_tokens=64):
    """Build controlled context/output plans; actual tokenizer counts come from the server."""
    if type(requests) is not int or not 1 <= requests <= 100000:
        raise ValueError("requests must be in 1..100000")
    if type(prompt_repeats) is not int or not 1 <= prompt_repeats <= 256:
        raise ValueError("prompt_repeats must be in 1..256")
    if type(output_tokens) is not int or not 1 <= output_tokens <= 512:
        raise ValueError("output_tokens must be in 1..512")
    observation = "GPU inference needs bounded memory and predictable recovery. "
    prompt = "Summarize these observations without adding facts:\n" + observation * prompt_repeats
    if len(prompt.encode()) > 16384:
        raise ValueError("constructed prompt exceeds service byte limit")
    label = f"context-r{prompt_repeats}-out{output_tokens}"
    return [{"prompt": prompt, "max_tokens": output_tokens, "class": label}
            for _ in range(requests)]


# Words that are exactly one token (with a leading space) in the Qwen2.5 tokenizer. A prompt of N of them is
# therefore N tokens by construction, so context length is controlled without a tokenizer at run time.
# tests/test_workloads.py re-verifies this against the real tokenizer when it is cached.
SINGLE_TOKEN_WORDS = (
    "the of and to in is that for it as was with be by on not he this are or his from at which but have an had "
    "they you were their one all we can her has there been if more when will would who so no what up out about "
    "into than them only other new some could time these two may then do first any my now such like our over man "
    "me even most made after also did many before must through back years where much your way well down should "
    "because each just those people how too little state good very make world still own see men work long get "
    "here between both life being under never day same another know while last might us great old year off come "
    "since against go came right used take three").split()

ITEM_REQUIRED = {"prompt", "max_tokens", "class"}
ITEM_OPTIONAL = {"prompt_tokens", "cancel_after_tokens", "cancel_after_s", "timeout_s"}
MAX_PROMPT_TOKENS = 2700  # keeps the prompt under the service's 16384-byte limit


def prompt_of_tokens(count, rng):
    """A prompt of exactly ``count`` tokens for Qwen tokenizers, drawn deterministically from ``rng``."""
    if type(count) is not int or not 1 <= count <= MAX_PROMPT_TOKENS:
        raise ValueError(f"prompt token count must be an integer in 1..{MAX_PROMPT_TOKENS}")
    return "".join(" " + SINGLE_TOKEN_WORDS[rng.randrange(len(SINGLE_TOKEN_WORDS))] for _ in range(count))


def sample(distribution, rng, *, integer=True):
    """Draw from a small declarative distribution: fixed, uniform, choice or lognormal."""
    if isinstance(distribution, (int, float)) and not isinstance(distribution, bool):
        distribution = {"dist": "fixed", "value": distribution}
    if not isinstance(distribution, dict):
        raise ValueError("a distribution must be a number or an object")
    kind = distribution.get("dist")
    if kind == "fixed":
        value = distribution["value"]
    elif kind == "uniform":
        low, high = distribution["min"], distribution["max"]
        if low > high:
            raise ValueError("uniform min cannot exceed max")
        value = rng.uniform(low, high)
    elif kind == "choice":
        values, weights = distribution["values"], distribution.get("weights")
        if not values or (weights is not None and (len(weights) != len(values) or min(weights) < 0
                                                    or not sum(weights) > 0)):
            raise ValueError("choice needs values and matching nonnegative weights")
        value = rng.choices(values, weights=weights)[0]
    elif kind == "lognormal":
        value = rng.lognormvariate(math.log(distribution["median"]), distribution["sigma"])
        value = min(max(value, distribution.get("min", 0)), distribution.get("max", math.inf))
    else:
        raise ValueError(f"unknown distribution {kind!r}")
    return int(round(value)) if integer else float(value)


def build_spec_workload(spec, requests):
    """Build a workload from a declarative spec. The same spec and seed give the same plan everywhere.

    spec: {"seed": int, "classes": [{"name", "weight", "prompt_tokens", "output_tokens"}], and optionally
    "cancel": {"fraction", "after_tokens"} and/or {"fraction", "after_s"}, "timeout_s": distribution}.
    """
    if type(requests) is not int or not 1 <= requests <= 100000:
        raise ValueError("requests must be in 1..100000")
    if not isinstance(spec, dict) or not spec.get("classes"):
        raise ValueError("a workload spec needs a nonempty classes list")
    rng = random.Random(spec.get("seed", 42))
    classes = spec["classes"]
    weights = [entry.get("weight", 1) for entry in classes]
    names = [entry["name"] for entry in classes]
    if len(set(names)) != len(names) or not all(isinstance(name, str) and 1 <= len(name) <= 32 for name in names):
        raise ValueError("class names must be unique strings of 1..32 characters")
    cancel = spec.get("cancel") or {}
    plan = []
    for _ in range(requests):
        entry = rng.choices(classes, weights=weights)[0]
        prompt_tokens = max(1, sample(entry["prompt_tokens"], rng))
        if prompt_tokens > MAX_PROMPT_TOKENS:
            # Never clamp silently: a plan that claims longer prompts than it contains would misreport its own evidence.
            raise ValueError(f"class {entry['name']!r} asks for a {prompt_tokens}-token prompt; the service accepts "
                             f"{MAX_PROMPT_TOKENS} at most (its 16384-byte prompt limit)")
        item = {"prompt": prompt_of_tokens(prompt_tokens, rng),
                "max_tokens": max(1, min(sample(entry["output_tokens"], rng), 512)),
                "class": entry["name"], "prompt_tokens": prompt_tokens}
        if cancel and rng.random() < cancel.get("fraction", 0):
            if "after_tokens" in cancel:
                item["cancel_after_tokens"] = max(1, sample(cancel["after_tokens"], rng))
            if "after_s" in cancel:
                item["cancel_after_s"] = max(0.001, sample(cancel["after_s"], rng, integer=False))
        if "timeout_s" in spec:
            item["timeout_s"] = max(0.001, sample(spec["timeout_s"], rng, integer=False))
        plan.append(item)
    return plan


def validate_workload(plan, *, max_requests=100000):
    if not isinstance(plan, list) or not 1 <= len(plan) <= max_requests:
        raise ValueError(f"workload must contain 1..{max_requests} request objects")
    for item in plan:
        if not isinstance(item, dict) or not ITEM_REQUIRED <= set(item) <= ITEM_REQUIRED | ITEM_OPTIONAL:
            raise ValueError("each workload item requires prompt, max_tokens and class")
        if not isinstance(item["prompt"], str) or not item["prompt"].strip() or len(item["prompt"].encode()) > 16384:
            raise ValueError("workload prompt must contain 1..16384 UTF-8 bytes")
        if type(item["max_tokens"]) is not int or not 1 <= item["max_tokens"] <= 512:
            raise ValueError("workload max_tokens must be an integer in 1..512")
        if not isinstance(item["class"], str) or not 1 <= len(item["class"]) <= 32:
            raise ValueError("workload class must contain 1..32 characters")
        for field in ("prompt_tokens", "cancel_after_tokens"):
            if field in item and (type(item[field]) is not int or item[field] < 1):
                raise ValueError(f"workload {field} must be a positive integer")
        for field in ("cancel_after_s", "timeout_s"):
            if field in item and (type(item[field]) not in (int, float) or not math.isfinite(item[field])
                                  or item[field] <= 0):
                raise ValueError(f"workload {field} must be a positive finite number")
    return plan
