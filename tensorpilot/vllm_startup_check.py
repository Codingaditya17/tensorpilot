"""Exercise a freshly started vLLM server with the sampling paths its warm-up and its first random-sampling request depend on.

The benchmark is greedy (temperature 0, no top-k or top-p), and vLLM skips its FlashInfer sampler for greedy requests, so a benchmark
alone would not touch the kernel whose missing nvcc killed the second vLLM attempt. This sends a few short completions that do take
that path, and checks each answer. It is the last step of the early startup stage, run on a small model before any 7B work is paid for.

    TENSORPILOT_API_KEY=... python -m tensorpilot.vllm_startup_check --url http://127.0.0.1:8081 --model Qwen/... --output out.json

The key is read from the environment so it never appears in a process listing. Exit status: 0 all requests answered, 1 otherwise.
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

CASES = (
    ("greedy", {"temperature": 0}),
    ("top_k_top_p", {"temperature": 0.8, "top_p": 0.9, "top_k": 20, "seed": 7}),
    ("top_k", {"temperature": 0.7, "top_k": 40, "seed": 11}),
    ("top_p", {"temperature": 0.7, "top_p": 0.8, "seed": 13}),
)


def complete(url, model, key, extra, timeout):
    if not url.startswith(("http://", "https://")):
        raise ValueError("the URL must be http or https")
    body = {"model": model, "prompt": "The three primary colors are", "max_tokens": 16, **extra}
    request = urllib.request.Request(url.rstrip("/") + "/v1/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # nosec B310 - scheme checked above
        return json.loads(response.read())


def check(answer):
    """Return a problem description, or None when the answer is a real completion."""
    try:
        choice = answer["choices"][0]
        tokens = answer["usage"]["completion_tokens"]
    except (KeyError, IndexError, TypeError):
        return "malformed response"
    if not isinstance(choice.get("text"), str) or not choice["text"].strip():
        return "empty text"
    if tokens < 1 or not choice.get("finish_reason"):
        return f"{tokens} tokens, finish_reason {choice.get('finish_reason')!r}"
    return None


def run(url, model, key, timeout=300):
    results = []
    for name, extra in CASES:
        started = time.perf_counter()
        try:
            problem = check(complete(url, model, key, extra, timeout))
        except (urllib.error.URLError, OSError, ValueError) as exc:
            problem = f"{type(exc).__name__}: {exc}"
        results.append({"case": name, "ok": problem is None, "problem": problem,
                        "seconds": round(time.perf_counter() - started, 2)})
    return {"schema": "tensorpilot.vllm-startup-check/v1", "model": model, "ok": all(r["ok"] for r in results), "cases": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    report = run(args.url, args.model, os.environ.get("TENSORPILOT_API_KEY", ""), args.timeout)
    with open(args.output, "w") as handle:
        json.dump(report, handle, indent=2)
    for r in report["cases"]:
        print(f"startup check {r['case']}: {'ok' if r['ok'] else 'FAILED ' + str(r['problem'])} ({r['seconds']} s)", file=sys.stderr)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
