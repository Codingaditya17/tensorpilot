"""Local MLX inference benchmarks: token latency and reproducible configuration."""
import argparse
import json
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
PROMPTS = [
    "Explain why GPU memory bandwidth can limit transformer inference in two sentences.",
    "What is the difference between prefill and decode in an autoregressive language model?",
    "Describe one tradeoff between inference throughput and tail latency.",
]


def summarize(records):
    if not records:
        raise ValueError("No records")
    completed = [r for r in records if r["ok"]]
    total_duration = sum(r["elapsed_s"] for r in records)
    latencies = sorted(r["elapsed_s"] for r in completed)
    def percentile(p):
        if not latencies:
            return None
        rank = (len(latencies) - 1) * p
        lower = int(rank)
        upper = min(lower + 1, len(latencies)-1)
        return latencies[lower] + (latencies[upper] - latencies[lower]) * (rank - lower)
    return {
        "successful_requests": len(completed),
        "failed_requests": len(records) - len(completed),
        "median_request_latency_s": statistics.median(latencies) if latencies else None,
        "p95_request_latency_s": percentile(.95),
        "total_generated_tokens": sum(r["generated_tokens"] for r in completed),
        "aggregate_output_tokens_per_s": sum(r["generated_tokens"] for r in completed) / total_duration if total_duration > 0 else None,
        "note": "Sequential benchmark: aggregate throughput uses summed request durations, not concurrent wall clock.",
    }


def run(model_name, n_requests, max_tokens, warmup):
    try:
        import mlx.core as mx
        from mlx_lm import load, generate
    except ImportError as exc:
        raise SystemExit("Install dependencies: pip install -r requirements.txt") from exc

    print(f"Loading {model_name} ... (first run downloads model weights)")
    model, tokenizer = load(model_name)
    if warmup:
        print("Warming up model ...")
        generate(model, tokenizer, prompt=PROMPTS[0], max_tokens=8, verbose=False)
        mx.eval(mx.array([1]))

    records = []
    for index in range(n_requests):
        prompt = PROMPTS[index % len(PROMPTS)]
        start = time.perf_counter()
        try:
            # mlx_lm's generate returns a finished string; we intentionally DO NOT claim TTFT.
            output = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
            elapsed = time.perf_counter() - start
            # Re-tokenizing output is an approximation; use a streaming/token-aware
            # generator for exact output token accounting in the next milestone.
            generated_tokens = len(tokenizer.encode(output, add_special_tokens=False))
            record = {"index": index, "ok": True, "elapsed_s": elapsed, "generated_tokens": generated_tokens,
                      "estimated_tokens_per_s": generated_tokens / elapsed if elapsed else None,
                      "prompt": prompt, "response_preview": output[:160]}
            print(f"[{index+1}/{n_requests}] {elapsed:.2f}s | approx {generated_tokens} tokens | {record['estimated_tokens_per_s']:.1f} tok/s")
        except Exception as exc:
            record = {"index": index, "ok": False, "elapsed_s": time.perf_counter()-start,
                      "generated_tokens": 0, "error": repr(exc)}
            print(f"[{index+1}/{n_requests}] failed: {exc}")
        records.append(record)
    return {"metadata": {"model": model_name, "run_at_utc": datetime.now(timezone.utc).isoformat(),
                         "platform": platform.platform(), "machine": platform.machine(),
                         "requests": n_requests, "max_tokens": max_tokens, "warmup": warmup,
                         "concurrency": 1, "token_counts": "approximate via re-tokenization",
                         "timings": "end-to-end only; first-token latency not measured"},
            "summary": summarize(records), "requests": records}


def main():
    parser = argparse.ArgumentParser(description="TensorPilot: MLX local inference baseline")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--requests", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--output", default="results/baseline.json")
    args = parser.parse_args()
    if args.requests <= 0 or args.max_tokens <= 0:
        parser.error("--requests and --max-tokens must be positive")
    result = run(args.model, args.requests, args.max_tokens, not args.no_warmup)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2))
    print("\nSummary:", json.dumps(result["summary"], indent=2))
    print(f"Saved: {destination}")


if __name__ == "__main__":
    main()
