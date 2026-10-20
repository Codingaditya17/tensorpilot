"""GPU readiness gate: device visibility, CUDA stack, VRAM, and one smoke inference.

Each check records its own evidence. Any failed required check makes the report
not ready and the command exits nonzero. The gate starts nothing, rents nothing,
and deploys nothing; it only inspects the machine it runs on.
"""
import argparse
import asyncio
import json
import platform
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .environment import dependency_versions, source_fingerprint
from .gpu_telemetry import sample_gpus

SCHEMA = "tensorpilot.gpu-preflight/v1"
REVISION = re.compile(r"[0-9a-f]{40}")
GIB = 1024**3


def check(name, passed, detail, **evidence):
    return {"name": name, "status": "pass" if passed else "fail", "detail": detail, **evidence}


def probe_torch():
    """Describe the PyTorch CUDA stack without importing it unless it is installed."""
    try:
        import torch
    except ImportError as exc:
        return {"importable": False, "error": type(exc).__name__}
    info = {"importable": True, "torch_version": torch.__version__, "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(), "devices": []}
    if info["cuda_available"]:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            free, total = torch.cuda.mem_get_info(index)
            info["devices"].append({"index": index, "name": properties.name,
                                    "compute_capability": f"{properties.major}.{properties.minor}",
                                    "free_bytes": free, "total_bytes": total})
    return info


def revision_check(revision):
    if revision is None or not REVISION.fullmatch(revision):
        return check("pinned_revision", False,
                     "a 40-character Hugging Face commit SHA is required for reproducible evidence",
                     revision=revision)
    return check("pinned_revision", True, "model revision is an immutable commit", revision=revision)


def vram_check(torch_info, min_vram_gib):
    devices = torch_info.get("devices") or []
    if not devices:
        return check("vram", False, "no CUDA device reported by PyTorch")
    threshold = min_vram_gib * GIB
    short = [device["index"] for device in devices
             if device["total_bytes"] < threshold or device["free_bytes"] < threshold]
    detail = (f"every device has at least {min_vram_gib} GiB total and free"
              if not short else f"devices below {min_vram_gib} GiB total or free: {short}")
    return check("vram", not short, detail, devices=devices)


async def run_preflight(*, model, revision, min_vram_gib, prompt, smoke_tokens, load_backend,
                        sampler=sample_gpus, torch_probe=probe_torch, skip_load=False):
    checks = []
    telemetry = await sampler()
    devices = telemetry.get("devices", [])
    checks.append(check("nvidia_smi_visible", telemetry["status"] == "measured" and bool(devices),
                        telemetry.get("reason", "nvidia-smi reports devices") if not devices else
                        f"nvidia-smi reports {len(devices)} device(s)", telemetry=telemetry))
    torch_info = torch_probe()
    cuda_ok = bool(torch_info.get("cuda_available")) and torch_info.get("cuda_runtime") is not None
    checks.append(check("torch_cuda", cuda_ok,
                        "PyTorch CUDA is available" if cuda_ok else
                        f"PyTorch CUDA unavailable ({torch_info.get('error', 'cuda_available is false')})",
                        torch=torch_info))
    checks.append(vram_check(torch_info, min_vram_gib))
    checks.append(revision_check(revision))
    smoke = None
    if skip_load:
        checks.append(check("model_smoke", False, "skipped by --skip-load; a smoke inference is required"))
    else:
        smoke = await asyncio.to_thread(_smoke, load_backend, prompt, smoke_tokens)
        checks.append(check("model_smoke", smoke["ok"], smoke["detail"], **{
            key: value for key, value in smoke.items() if key not in ("ok", "detail")}))
    return {"schema": SCHEMA, "ready": all(item["status"] == "pass" for item in checks),
            "checks": checks, "model": model, "revision": revision,
            "min_vram_gib": min_vram_gib, "telemetry": telemetry, "torch": torch_info}


def _smoke(load_backend, prompt, smoke_tokens):
    try:
        started = time.perf_counter()
        backend = load_backend()
        load_s = time.perf_counter() - started
        first_token_s = None
        tokens = []
        began = time.perf_counter()
        for token in backend.stream(prompt, smoke_tokens):
            if first_token_s is None and token.output:
                first_token_s = time.perf_counter() - began
            if token.output:
                tokens.append(token.token_id)
        generation_s = time.perf_counter() - began
        memory = backend.memory()
        provenance = backend.provenance
    except Exception as exc:  # Preflight reports the failure; it never hides it.
        return {"ok": False, "detail": f"smoke inference failed: {type(exc).__name__}"}
    passed = bool(tokens)
    return {"ok": passed, "detail": f"generated {len(tokens)} output token(s)" if passed else
            "no output tokens were produced", "model_load_s": load_s, "first_token_s": first_token_s,
            "generation_s": generation_s, "output_tokens": len(tokens), "memory": memory,
            "artifact_sha256": provenance.get("artifact_sha256"),
            "model_revision": provenance.get("model_revision"),
            "gpu": provenance.get("gpu")}


def make_backend_loader(args):
    def load():
        from .cuda_backend import CUDABackend
        return CUDABackend(args.model, revision=args.revision, offline=args.offline, device=args.device,
                           dtype=args.dtype, memory_fraction=args.cuda_memory_fraction,
                           cache_dir=args.cache_dir, max_prompt_tokens=args.max_prompt_tokens)
    return load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Transformers checkpoint, local directory or repository")
    parser.add_argument("--revision", help="40-character Hugging Face commit SHA")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--cache-dir")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--cuda-memory-fraction", type=float, default=.9)
    parser.add_argument("--max-prompt-tokens", type=int, default=2048)
    parser.add_argument("--min-vram-gib", type=float, default=16)
    parser.add_argument("--prompt", default="Explain why GPU memory bandwidth limits decoding.")
    parser.add_argument("--smoke-tokens", type=int, default=16)
    parser.add_argument("--skip-load", action="store_true", help="run device checks without loading the model")
    parser.add_argument("--output", default="results/v09_gpu/preflight.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 0 < args.min_vram_gib <= 1024 or not 1 <= args.smoke_tokens <= 128:
        parser.error("min-vram-gib must be in (0, 1024]; smoke-tokens in 1..128")
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        parser.error("output already exists; choose a new path or use --overwrite")
    report = asyncio.run(run_preflight(
        model=args.model, revision=args.revision, min_vram_gib=args.min_vram_gib,
        prompt=args.prompt, smoke_tokens=args.smoke_tokens, load_backend=make_backend_loader(args),
        skip_load=args.skip_load))
    report.update({"generated_at_utc": datetime.now(timezone.utc).isoformat(),
                   "host": {"platform": platform.platform(), "machine": platform.machine(),
                            "python": platform.python_version()},
                   "source_sha256": source_fingerprint(), "dependencies": dependency_versions()})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    for item in report["checks"]:
        print(f"[{item['status'].upper()}] {item['name']}: {item['detail']}")
    print(f"ready: {str(report['ready']).lower()}  report: {output}")
    sys.exit(0 if report["ready"] else 1)


if __name__ == "__main__":
    main()
