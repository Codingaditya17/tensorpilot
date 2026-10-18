"""Deterministic workload plans and portable content fingerprints."""
import hashlib
import json
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
    if type(output_tokens) is not int or not 1 <= output_tokens <= 256:
        raise ValueError("output_tokens must be in 1..256")
    observation = "GPU inference needs bounded memory and predictable recovery. "
    prompt = "Summarize these observations without adding facts:\n" + observation * prompt_repeats
    if len(prompt.encode()) > 16384:
        raise ValueError("constructed prompt exceeds service byte limit")
    label = f"context-r{prompt_repeats}-out{output_tokens}"
    return [{"prompt": prompt, "max_tokens": output_tokens, "class": label}
            for _ in range(requests)]


def validate_workload(plan, *, max_requests=100000):
    if not isinstance(plan, list) or not 1 <= len(plan) <= max_requests:
        raise ValueError(f"workload must contain 1..{max_requests} request objects")
    for item in plan:
        if not isinstance(item, dict) or set(item) != {"prompt", "max_tokens", "class"}:
            raise ValueError("each workload item requires prompt, max_tokens and class")
        if not isinstance(item["prompt"], str) or not item["prompt"].strip() or len(item["prompt"].encode()) > 16384:
            raise ValueError("workload prompt must contain 1..16384 UTF-8 bytes")
        if type(item["max_tokens"]) is not int or not 1 <= item["max_tokens"] <= 256:
            raise ValueError("workload max_tokens must be an integer in 1..256")
        if not isinstance(item["class"], str) or not 1 <= len(item["class"]) <= 32:
            raise ValueError("workload class must contain 1..32 characters")
    return plan
