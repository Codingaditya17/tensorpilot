"""Resolve immutable model provenance before loading the inference runtime."""
import hashlib
import json
from pathlib import Path
import re

MODEL_PATTERNS = ["*.json", "*.safetensors", "*.py", "*.model", "*.tiktoken",
                  "*.txt", "*.jsonl", "*.jinja"]


def artifact_fingerprint(directory):
    root = Path(directory)
    entries = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and any(path.match(pattern) for pattern in MODEL_PATTERNS):
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            entries.append({"file": path.relative_to(root).as_posix(), "sha256": digest})
    if not entries:
        raise ValueError("model directory contains no supported artifact files")
    manifest = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(manifest).hexdigest()


def resolve_model(model, *, revision=None, offline=False, cache_dir=None):
    path = Path(model).expanduser()
    local = path.exists()
    if local:
        if not path.is_dir():
            raise ValueError("local model path must be a directory")
        if revision:
            raise ValueError("revision is only supported for a Hugging Face repository")
    else:
        from huggingface_hub import snapshot_download
        path = Path(snapshot_download(repo_id=model, revision=revision, cache_dir=cache_dir,
                                      local_files_only=offline, allow_patterns=MODEL_PATTERNS))
    resolved_revision = path.name if not local and re.fullmatch(r"[0-9a-f]{40}", path.name) else None
    return path, {"model_source": "local_directory" if local else "huggingface",
                  "requested_revision": revision, "model_revision": resolved_revision,
                  "artifact_sha256": artifact_fingerprint(path)}
