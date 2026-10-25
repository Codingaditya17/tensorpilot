"""GPU readiness gate: host, driver, CUDA stack, model revision, storage, VRAM and a smoke test.

Each check records its own evidence. Any failed required check makes the report
not ready and the command exits nonzero. The gate starts nothing, rents nothing,
and deploys nothing; it only inspects the machine and the Hugging Face repository.

Driver policy, from NVIDIA's CUDA compatibility documentation:
- CUDA 12.8 toolkit GA lists Linux driver 570.26 or newer.
- CUDA 12.x minor-version compatibility runs on drivers from 525 up to 580.
So a driver below 525 is a hard failure, a driver from 525 to 570.25 is a warning
that the smoke test must confirm, and 570.26 or newer passes. The CUDA initialization
check and the smoke inference decide whether the installed stack actually works.
"""
import argparse
import asyncio
import importlib.metadata
import json
import platform
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .environment import dependency_versions, source_fingerprint
from .gpu_telemetry import sample_gpus

SCHEMA = "tensorpilot.gpu-preflight/v2"
REVISION = re.compile(r"[0-9a-f]{40}")
GIB = 1024**3
MIN_DRIVER = (525,)
CUDA_12_8_DRIVER = (570, 26)
# Pins from requirements-cuda.in / requirements-cuda.txt. The preflight compares the
# installed versions against these, so a drifted environment cannot pass silently.
EXPECTED_PINS = {"torch": "2.9.1", "transformers": "5.20.1"}
# Headroom for activations, the caching allocator and a short KV cache.
VRAM_HEADROOM_BYTES = int(1.5 * GIB)
# Downloads need the weights once; the factor of two leaves room for the temporary copy.
STORAGE_FACTOR = 2
STORAGE_MARGIN_BYTES = 2 * GIB


def check(name, passed, detail, **evidence):
    return {"name": name, "status": "pass" if passed else "fail", "detail": detail, **evidence}


def warn(name, detail, **evidence):
    return {"name": name, "status": "warn", "detail": detail, **evidence}


def parse_driver(version):
    parts = re.findall(r"\d+", version or "")
    if not parts:
        return None
    return tuple(int(part) for part in parts[:2])


def driver_check(telemetry):
    devices = telemetry.get("devices") or []
    if not devices:
        return check("driver", False, "no NVIDIA driver version reported (nvidia-smi unavailable)")
    raw = devices[0].get("driver_version")
    version = parse_driver(raw)
    if version is None:
        return check("driver", False, f"unparseable driver version {raw!r}", driver_version=raw)
    if version < MIN_DRIVER:
        return check("driver", False,
                     f"driver {raw} is below 525, the minimum for CUDA 12.x compatibility",
                     driver_version=raw)
    if version < CUDA_12_8_DRIVER:
        return warn("driver", f"driver {raw} is in the CUDA 12.x minor-version compatibility range "
                    "(525 to 579); the CUDA 12.8 toolkit lists 570.26 or newer. The smoke test must pass "
                    "and some features may be unavailable on this driver", driver_version=raw)
    return check("driver", True, f"driver {raw} meets the CUDA 12.8 toolkit requirement (570.26)",
                 driver_version=raw)


def pins_check(installed=None):
    installed = installed or {}
    found = {}
    for name, expected in EXPECTED_PINS.items():
        if name in installed:
            found[name] = installed[name]
            continue
        try:
            found[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            found[name] = None
    mismatched = {name: {"expected": expected, "installed": found[name]}
                  for name, expected in EXPECTED_PINS.items()
                  if (found[name] or "").split("+")[0] != expected}
    if mismatched:
        return check("dependency_pins", False, f"installed versions differ from the lock: {mismatched}",
                     installed=found)
    return check("dependency_pins", True, "torch and transformers match the locked versions", installed=found)


def revision_check(revision):
    if revision is None or not REVISION.fullmatch(revision):
        return check("pinned_revision", False,
                     "a 40-character Hugging Face commit SHA is required for reproducible evidence",
                     revision=revision)
    return check("pinned_revision", True, "model revision is an immutable commit", revision=revision)


def remote_revision_check(model, revision, *, fetch_info):
    """Confirm the pinned commit exists upstream and report its weight bytes.

    ``fetch_info`` returns the Hub's model info for (repo, revision); it is injected so
    tests and offline runs do not touch the network.
    """
    try:
        info = fetch_info(model, revision)
    except Exception as exc:  # Network, auth and missing-repo failures are reported, not hidden.
        return check("remote_revision", False, f"could not read {model}@{revision}: {type(exc).__name__}"), None
    resolved = getattr(info, "sha", None)
    if resolved != revision:
        return check("remote_revision", False,
                     f"upstream resolves {model} to {resolved}, not the pinned {revision}",
                     resolved=resolved), None
    weights = sum((getattr(f, "size", 0) or 0) for f in (info.siblings or [])
                  if getattr(f, "rfilename", "").endswith(".safetensors"))
    if weights <= 0:
        return check("remote_revision", False, "no .safetensors weights listed for this revision"), None
    return check("remote_revision", True, "pinned commit exists upstream",
                 resolved=resolved, weight_bytes=weights), weights


def local_weight_bytes(path):
    directory = Path(path)
    total = sum(item.stat().st_size for item in directory.rglob("*.safetensors"))
    return total or None


def storage_check(cache_dir, weight_bytes):
    target = Path(cache_dir or Path.home() / ".cache" / "huggingface")
    target.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(target).free
    required = weight_bytes * STORAGE_FACTOR + STORAGE_MARGIN_BYTES
    return check("storage", free >= required,
                 f"{free / GIB:.1f} GiB free at {target}; need {required / GIB:.1f} GiB",
                 free_bytes=free, required_bytes=required, cache_dir=str(target))


def vram_check(torch_info, weight_bytes, min_vram_gib):
    devices = torch_info.get("devices") or []
    if not devices:
        return check("vram", False, "no CUDA device reported by PyTorch")
    device = devices[0]
    required = max(weight_bytes + VRAM_HEADROOM_BYTES, min_vram_gib * GIB) if weight_bytes else min_vram_gib * GIB
    fits = device["free_bytes"] >= required and device["total_bytes"] >= required
    return check("vram", fits,
                 f"device {device['index']} has {device['free_bytes'] / GIB:.1f} GiB free of "
                 f"{device['total_bytes'] / GIB:.1f} GiB; need {required / GIB:.1f} GiB "
                 f"(weights plus headroom, floor {min_vram_gib} GiB)",
                 devices=devices, required_bytes=required)


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


def fetch_hub_info(model, revision):
    from huggingface_hub import HfApi
    return HfApi().model_info(model, revision=revision, files_metadata=True)


async def run_preflight(*, model, revision, min_vram_gib, prompt, smoke_tokens, load_backend,
                        cache_dir=None, sampler=sample_gpus, torch_probe=probe_torch,
                        fetch_info=fetch_hub_info, skip_remote=False, skip_load=False,
                        installed_pins=None, local_model=False):
    checks = []
    telemetry = await sampler()
    devices = telemetry.get("devices", [])
    checks.append(check("nvidia_smi_visible", telemetry["status"] == "measured" and bool(devices),
                        telemetry.get("reason", "nvidia-smi reports devices") if not devices else
                        f"nvidia-smi reports {len(devices)} device(s)", telemetry=telemetry))
    checks.append(driver_check(telemetry))
    torch_info = torch_probe()
    cuda_ok = bool(torch_info.get("cuda_available")) and torch_info.get("cuda_runtime") is not None
    checks.append(check("torch_cuda", cuda_ok,
                        "PyTorch CUDA is available" if cuda_ok else
                        f"PyTorch CUDA unavailable ({torch_info.get('error', 'cuda_available is false')})",
                        torch=torch_info))
    checks.append(pins_check(installed_pins))
    checks.append(revision_check(revision))
    weight_bytes = None
    if local_model:
        weight_bytes = local_weight_bytes(model)
        checks.append(check("local_weights", weight_bytes is not None,
                            "local model directory contains .safetensors weights" if weight_bytes else
                            "no .safetensors weights in the local directory", weight_bytes=weight_bytes))
    elif skip_remote:
        checks.append(check("remote_revision", False, "skipped by --skip-remote; remote verification is required"))
    elif revision and REVISION.fullmatch(revision):
        remote, weight_bytes = await asyncio.to_thread(remote_revision_check, model, revision,
                                                       fetch_info=fetch_info)
        checks.append(remote)
    if weight_bytes:
        checks.append(storage_check(cache_dir, weight_bytes))
    checks.append(vram_check(torch_info, weight_bytes, min_vram_gib))
    smoke = None
    if skip_load:
        checks.append(check("model_smoke", False, "skipped by --skip-load; a smoke inference is required"))
    else:
        smoke = await asyncio.to_thread(_smoke, load_backend, prompt, smoke_tokens)
        checks.append(check("model_smoke", smoke["ok"], smoke["detail"], **{
            key: value for key, value in smoke.items() if key not in ("ok", "detail")}))
    return {"schema": SCHEMA, "ready": all(item["status"] != "fail" for item in checks),
            "warnings": [item["name"] for item in checks if item["status"] == "warn"],
            "checks": checks, "model": model, "revision": revision, "weight_bytes": weight_bytes,
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
    parser.add_argument("--model", required=True, help="Transformers checkpoint: repository or local directory")
    parser.add_argument("--revision", help="40-character Hugging Face commit SHA")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--cache-dir")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--cuda-memory-fraction", type=float, default=.9)
    parser.add_argument("--max-prompt-tokens", type=int, default=2048)
    parser.add_argument("--min-vram-gib", type=float, default=8,
                        help="floor for free device memory; the model-fit requirement is checked as well")
    parser.add_argument("--prompt", default="Explain why GPU memory bandwidth limits decoding.")
    parser.add_argument("--smoke-tokens", type=int, default=16)
    parser.add_argument("--skip-load", action="store_true", help="run device checks without loading the model")
    parser.add_argument("--skip-remote", action="store_true", help="do not contact the Hugging Face Hub")
    parser.add_argument("--output", default="results/v09_gpu/preflight.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 0 < args.min_vram_gib <= 1024 or not 1 <= args.smoke_tokens <= 128:
        parser.error("min-vram-gib must be in (0, 1024]; smoke-tokens in 1..128")
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        parser.error("output already exists; choose a new path or use --overwrite")
    local_model = Path(args.model).expanduser().is_dir()
    report = asyncio.run(run_preflight(
        model=args.model, revision=args.revision, min_vram_gib=args.min_vram_gib,
        prompt=args.prompt, smoke_tokens=args.smoke_tokens, load_backend=make_backend_loader(args),
        cache_dir=args.cache_dir, skip_remote=args.skip_remote, skip_load=args.skip_load,
        local_model=local_model))
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
