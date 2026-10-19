"""Capture a bounded steady-state MLX workload as an Apple Metal GPU trace."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from .backend import MLXBackend
from .bench import DEFAULT_MODEL
from .reporting import write_json


def run_capture(backend, trace_path, prompts, max_tokens):
    trace_path = Path(trace_path)
    if trace_path.suffix != ".gputrace":
        raise ValueError("capture path must end in .gputrace")
    if trace_path.exists():
        raise FileExistsError(f"capture already exists: {trace_path}")
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    # Excluded warmup ensures model loading/compilation is not confused with the
    # captured steady-state window.
    list(backend.stream("Warm up TensorPilot.", min(8, max_tokens)))
    backend.mx.synchronize()
    before = backend.memory()
    records = []
    capture_started = time.perf_counter()
    backend.mx.metal.start_capture(str(trace_path))
    try:
        for index, prompt in enumerate(prompts):
            started = time.perf_counter()
            tokens = list(backend.stream(prompt, max_tokens))
            backend.mx.synchronize()
            records.append({"index": index, "prompt_utf8_bytes": len(prompt.encode()),
                            "prompt_tokens": tokens[-1].prompt_tokens if tokens else None,
                            "sampled_tokens_including_eos": len(tokens),
                            "output_tokens": sum(token.output for token in tokens),
                            "finish_reason": tokens[-1].finish_reason if tokens else None,
                            "synchronized_elapsed_s": time.perf_counter() - started})
    finally:
        backend.mx.metal.stop_capture()
    backend.mx.synchronize()
    return {"schema_version": 1, "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            "measurement_source": "MLX Metal capture plus synchronized model-thread wall clock",
            "trace": str(trace_path), "model": backend.provenance,
            "max_tokens": max_tokens, "requests": records,
            "capture_window_s": time.perf_counter() - capture_started,
            "memory_before": before, "memory_after": backend.memory(),
            "limitations": "Inspect kernel names/durations in Xcode Instruments. Metal traces are Apple-specific and do not predict ROCm behavior."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output", default="results/v06_kernel_profile.gputrace")
    parser.add_argument("--requests", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--memory-gib", type=float, default=6)
    args = parser.parse_args()
    if not 1 <= args.requests <= 16 or not 1 <= args.max_tokens <= 128:
        parser.error("capture limits: requests in 1..16 and max-tokens in 1..128")
    if not 0 < args.memory_gib <= 8:
        parser.error("memory-gib must be in (0, 8]")
    try:
        backend = MLXBackend(args.model, int(args.memory_gib * 1024**3), 2048,
                             revision=args.revision, offline=args.offline)
        prompts = ["Explain one GPU inference bottleneck in one sentence."
                   for _ in range(args.requests)]
        result = run_capture(backend, args.output, prompts, args.max_tokens)
        manifest = str(Path(args.output).with_suffix(".json"))
        write_json(manifest, result)
    except (ValueError, OSError, RuntimeError, FileExistsError) as exc:
        parser.exit(1, f"Capture failed: {exc}\n")
    print(json.dumps({"trace": args.output, "manifest": manifest,
                      "requests": len(result["requests"])}, indent=2))


if __name__ == "__main__":
    main()
