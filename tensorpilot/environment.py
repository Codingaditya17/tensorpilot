"""Small reproducibility metadata helpers; no host identity or secrets."""
import hashlib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def source_fingerprint():
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.py")):
        digest.update(path.name.encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def dependency_versions():
    result = {}
    for name in ("mlx", "mlx-lm", "aiohttp", "prometheus-client", "huggingface-hub", "transformers", "tokenizers", "torch",
                 "bitsandbytes", "accelerate"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result
