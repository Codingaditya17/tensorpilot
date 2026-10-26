"""Model compatibility and VRAM-fit planning, computed from a checkpoint's real config and weight size.

Nothing here loads weights or touches a GPU. The numbers are estimates derived from ``config.json`` and
the safetensors file sizes; the verdicts are what the planner expects, and a real run (the preflight and
``cuda_peak_allocated_bytes``) is what confirms them. A model is reported as servable only if TensorPilot's
batched forward pass actually implements its architecture.

    python -m tensorpilot.model_plan --model Qwen/Qwen2.5-7B-Instruct --vram-gib 24
"""
import argparse
import json
import sys
from types import SimpleNamespace

from . import quantization as quant
from .cuda_batch import GIB, DEFAULT_HEADROOM_BYTES, UnsupportedModel, plan_kv_pool, read_geometry

RTX_4090_TOTAL_BYTES = 25252724736          # measured by PyTorch on the first RTX 4090 run
CONTEXT_POINTS = (512, 1024, 2048, 4096, 8192)
# bitsandbytes dequantizes one weight matrix to the compute dtype per matmul. Budget three of the largest in flight.
DEQUANT_SCRATCH_MATRICES = 3
DTYPE_BYTES = {"bfloat16": 2, "float16": 2, "float32": 4}


def activation_estimate_bytes(config, *, batch, chunk, dtype_bytes):
    """Rough transient activation memory for one forward pass. An estimate, never a measurement.

    Per token a decoder layer keeps the MLP intermediate (gate, up, product) and a handful of
    hidden-sized tensors alive; the vocabulary-sized logits exist only for the sampled position.
    """
    tokens = max(batch, chunk)
    per_token = (3 * config.intermediate_size + 8 * config.hidden_size) * dtype_bytes
    logits = batch * config.vocab_size * 4
    return tokens * per_token + logits


def assess(config, *, repo, revision, license, weights_bytes, gpu_total_bytes=RTX_4090_TOTAL_BYTES,
           memory_fraction=0.9, headroom_bytes=DEFAULT_HEADROOM_BYTES, dtype="bfloat16",
           prompt_tokens=1024, output_tokens=512, requested_capacity=64, prefill_chunk=256, quantization="none"):
    dtype_bytes = DTYPE_BYTES[dtype]
    download_bytes = weights_bytes
    if quantization != "none":
        weights_bytes = quant.estimate_resident_bytes(config, quantization, dtype_bytes)
    cap_bytes = int(gpu_total_bytes * memory_fraction)
    entry = {"model": repo, "revision": revision, "license": license, "model_type": getattr(config, "model_type", None),
             "quantization": quantization, "download_gib": round(download_bytes / GIB, 2),
             "weights_gib": round(weights_bytes / GIB, 2), "allocator_cap_gib": round(cap_bytes / GIB, 2),
             "context_length": getattr(config, "max_position_embeddings", None), "dtype": dtype,
             "tokenizer_note": "byte-level BPE; the server tokenizes raw text with default special tokens "
                               "and applies no chat template"}
    try:
        geometry = read_geometry(config)
    except UnsupportedModel as error:
        return {**entry, "verdict": "unsupported_architecture", "reason": str(error)}
    bytes_per_token = geometry.kv_bytes_per_token(dtype_bytes)
    entry.update(layers=geometry.layers, q_heads=geometry.q_heads, kv_heads=geometry.kv_heads,
                 head_dim=geometry.head_dim, kv_bytes_per_token=bytes_per_token,
                 kv_kib_per_token=round(bytes_per_token / 1024, 2))
    headroom = headroom_bytes
    if quantization != "none":
        largest = config.hidden_size * config.intermediate_size * dtype_bytes
        entry["dequant_scratch_gib"] = round(DEQUANT_SCRATCH_MATRICES * largest / GIB, 2)
        headroom += DEQUANT_SCRATCH_MATRICES * largest
    row_tokens = min(prompt_tokens + output_tokens, getattr(config, "max_position_embeddings", 10**9))
    entry["activation_estimate_mib_decode_batch"] = round(activation_estimate_bytes(
        config, batch=requested_capacity, chunk=1, dtype_bytes=dtype_bytes) / 1024**2, 1)
    entry["activation_estimate_mib_prefill_chunk"] = round(activation_estimate_bytes(
        config, batch=1, chunk=prefill_chunk, dtype_bytes=dtype_bytes) / 1024**2, 1)
    try:
        plan = plan_kv_pool(weights_bytes=weights_bytes, bytes_per_token=bytes_per_token, row_tokens=row_tokens,
                            requested_capacity=requested_capacity, total_vram_bytes=gpu_total_bytes,
                            memory_fraction=memory_fraction, headroom_bytes=headroom)
    except ValueError as error:
        return {**entry, "verdict": "does_not_fit", "reason": str(error),
                "note": "TensorPilot's only quantized mode is nf4; there is no offloading or tensor parallelism, "
                        "so a model whose resident weights do not fit cannot be served on this GPU"}
    entry.update(verdict="fits", row_tokens=row_tokens, rows_granted=plan.capacity, rows_requested=requested_capacity,
                 pool_gib=round(plan.pool_bytes / GIB, 2), pool_limited_by=plan.limited_by,
                 kv_budget_gib=round(plan.budget_bytes / GIB, 2))
    sweep = {}
    for context in CONTEXT_POINTS:
        if context > (getattr(config, "max_position_embeddings", 10**9)):
            continue
        row = context * bytes_per_token
        rows = max(0, plan.budget_bytes // row - 1)
        sweep[str(context)] = {"row_gib": round(row / GIB, 3), "max_concurrent_sequences": int(rows),
                               "holds_128_concurrent": bool(rows >= 128)}
    entry["max_concurrent_sequences_by_context"] = sweep
    return entry


def fetch(repo, revision=None):
    """Read config.json and weight sizes from the Hub. Metadata only; no weights are downloaded."""
    from huggingface_hub import HfApi, hf_hub_download
    from transformers import AutoConfig
    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    path = hf_hub_download(repo, "config.json", revision=info.sha)
    config = AutoConfig.from_pretrained(path.rsplit("/", 1)[0])
    weights = sum(sibling.size or 0 for sibling in info.siblings if sibling.rfilename.endswith(".safetensors"))
    license = next((tag.split(":", 1)[1] for tag in info.tags if tag.startswith("license:")), "unknown")
    return SimpleNamespace(config=config, repo=repo, revision=info.sha, license=license, weights_bytes=weights,
                           gated=bool(info.gated))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", action="append", required=True, help="repeatable Hugging Face repository")
    parser.add_argument("--revision", help="commit SHA (applies to a single --model)")
    parser.add_argument("--vram-gib", type=float, help="override device memory instead of the RTX 4090 value")
    parser.add_argument("--memory-fraction", type=float, default=0.9)
    parser.add_argument("--dtype", choices=tuple(DTYPE_BYTES), default="bfloat16")
    parser.add_argument("--quantization", action="append", choices=quant.MODES,
                        help="repeatable; each model is assessed once per mode (default: none)")
    parser.add_argument("--prompt-tokens", type=int, default=1024)
    parser.add_argument("--output-tokens", type=int, default=512)
    parser.add_argument("--capacity", type=int, default=64)
    parser.add_argument("--output", help="write the matrix as JSON")
    args = parser.parse_args()
    total = int(args.vram_gib * GIB) if args.vram_gib else RTX_4090_TOTAL_BYTES
    rows = []
    for repo in args.model:
        fetched = fetch(repo, args.revision if len(args.model) == 1 else None)
        for mode in args.quantization or ["none"]:
            entry = assess(fetched.config, repo=repo, revision=fetched.revision, license=fetched.license,
                           weights_bytes=fetched.weights_bytes, gpu_total_bytes=total,
                           memory_fraction=args.memory_fraction, dtype=args.dtype,
                           prompt_tokens=args.prompt_tokens, output_tokens=args.output_tokens,
                           requested_capacity=args.capacity, quantization=mode)
            entry["gated"] = fetched.gated
            rows.append(entry)
    document = {"schema": "tensorpilot.model-matrix/v1", "estimate_kind": "derived_from_config_and_file_sizes_"
                "not_measured", "gpu_total_bytes": total, "memory_fraction": args.memory_fraction,
                "headroom_bytes": DEFAULT_HEADROOM_BYTES, "models": rows}
    text = json.dumps(document, indent=2)
    if args.output:
        with open(args.output, "w") as handle:
            handle.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
