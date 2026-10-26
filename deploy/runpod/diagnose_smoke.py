"""One-off diagnostic for a failed preflight `model_smoke`: capture the full traceback.

Runs the same CUDABackend load and greedy stream as the preflight, once per dtype, so the
failing frame and the dtype dependence are both visible. Self-contained: it needs only the installed
tensorpilot package, so it works on a Pod running an image that predates --traceback-file.

Safety: it starts no server, opens no port and makes no network call beyond the pinned model
download the preflight already did. Credential values (env names containing KEY, TOKEN, SECRET,
PASSWORD, CREDENTIAL, plus Hugging Face tokens and bearer headers) are redacted before anything is
printed or written. The report is written with mode 0600.

    python3 diagnose_smoke.py [--output FILE]
"""
import argparse
import gc
import os
import platform
import re
import sys
import traceback

SECRET_NAME = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL", re.IGNORECASE)
SECRET_PATTERNS = (re.compile(r"hf_[A-Za-z0-9]{16,}"), re.compile(r"(?i)bearer\s+\S+"))
SAFE_ENV = ("CUDA_VISIBLE_DEVICES", "PYTORCH_CUDA_ALLOC_CONF", "HF_HOME", "TENSORPILOT_CACHE_DIR",
            "TENSORPILOT_MODEL", "TENSORPILOT_REVISION", "TENSORPILOT_RESULTS_ROOT")


def redact(text):
    for name, value in os.environ.items():
        if len(value) >= 8 and SECRET_NAME.search(name):
            text = text.replace(value, "[REDACTED]")
    for pattern in SECRET_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=os.environ.get("TENSORPILOT_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"))
    parser.add_argument("--revision", default=os.environ.get("TENSORPILOT_REVISION"))
    parser.add_argument("--cache-dir", default=os.environ.get("TENSORPILOT_CACHE_DIR"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-cpu", action="store_true", help="for local testing only")
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--prompt", default="Explain why GPU memory bandwidth limits decoding.")
    parser.add_argument("--output", default=os.path.join(os.environ.get("TENSORPILOT_RESULTS_ROOT", "."),
                                                           "smoke-diagnosis.txt"))
    args = parser.parse_args()

    lines = []

    def emit(text=""):
        lines.append(text)
        print(redact(text), flush=True)

    import torch
    import transformers
    from tensorpilot.cuda_backend import CUDABackend

    emit(f"python {platform.python_version()} torch {torch.__version__} (cuda {torch.version.cuda}) "
         f"transformers {transformers.__version__}")
    emit("env: " + ", ".join(f"{name}={os.environ[name]}" for name in SAFE_ENV if name in os.environ))
    emit(f"cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        emit(f"device0={torch.cuda.get_device_name(0)} capability={torch.cuda.get_device_capability(0)} "
             f"free_mib={free // 2**20} total_mib={total // 2**20}")
    emit(f"model={args.model} revision={args.revision} device={args.device}")

    outcome = {}
    for dtype in ("float16", "bfloat16", "float32"):
        emit(f"\n=== dtype {dtype}")
        backend = None
        try:
            backend = CUDABackend(args.model, revision=args.revision, cache_dir=args.cache_dir,
                                  device=args.device, dtype=dtype, allow_cpu=args.allow_cpu)
            emit("load: ok")
            count = sum(1 for token in backend.stream(args.prompt, args.tokens) if token.output)
            emit(f"generate: ok, {count} output token(s)")
            outcome[dtype] = "ok"
        except Exception as exc:
            outcome[dtype] = f"FAIL {type(exc).__name__}"
            emit(f"FAILED: {type(exc).__name__}")
            emit(traceback.format_exc()[-8000:])
        finally:
            backend = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    emit("\nsummary: " + ", ".join(f"{dtype}={result}" for dtype, result in outcome.items()))
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w") as handle:
        handle.write(redact("\n".join(lines)) + "\n")
    print(f"\nreport written (mode 0600): {args.output}")
    sys.exit(0 if all(result == "ok" for result in outcome.values()) else 1)


if __name__ == "__main__":
    main()
