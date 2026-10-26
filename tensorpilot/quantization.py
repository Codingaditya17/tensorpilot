"""Explicit 4-bit weight quantization for the CUDA backend (bitsandbytes NF4).

Quantization is a mode the operator chooses (``--quantization nf4``), never a fallback. If the mode is
requested and the stack cannot deliver it, loading fails with the reason; the backend does not quietly serve
bf16 or a different precision.

What NF4 changes: the attention and MLP projection weights are stored as 4-bit NormalFloat blocks with
double-quantized scales, and are dequantized to the compute dtype inside each matmul. The embedding, the
output head, norms and biases stay in the compute dtype. KV cache, activations and sampling are unchanged.

What it does not promise: lower memory does not mean higher speed. A bitsandbytes 4-bit matmul with more than
one token per step dequantizes each weight matrix to the compute dtype before the multiply, which adds memory
traffic that bf16 does not pay. Whether NF4 is faster or slower for a given batch width is a measurement.
"""
from importlib.metadata import PackageNotFoundError, version

MODES = ("none", "nf4")
NF4_BLOCK_SIZE = 64          # bitsandbytes' default 4-bit block size
SECOND_LEVEL_BLOCK = 256     # double quantization: block-scale values are themselves quantized in blocks of 256
# Transformers 5.20.1's own minimums for its bitsandbytes integration.
BITSANDBYTES_MIN = "0.46.1"
ACCELERATE_MIN = "1.1.0"
# Bytes per quantized weight: 4 bits, plus an 8-bit scale per block, plus a float32 scale per 256 blocks.
NF4_BYTES_PER_WEIGHT = 0.5 + 1 / NF4_BLOCK_SIZE + 4 / (NF4_BLOCK_SIZE * SECOND_LEVEL_BLOCK)


class QuantizationUnavailable(RuntimeError):
    """The requested quantization cannot be provided by this environment."""


def validate_mode(mode):
    if mode not in MODES:
        raise ValueError(f"quantization must be one of {MODES}; got {mode!r}")
    return mode


def _version_tuple(text):
    parts = []
    for piece in text.split("+")[0].split("."):
        digits = "".join(character for character in piece if character.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def require_stack(mode):
    """Fail with an actionable message unless bitsandbytes and accelerate can quantize on this host."""
    if validate_mode(mode) == "none":
        return
    for package, minimum in (("bitsandbytes", BITSANDBYTES_MIN), ("accelerate", ACCELERATE_MIN)):
        try:
            installed = version(package)
        except PackageNotFoundError:
            raise QuantizationUnavailable(
                f"--quantization {mode} needs {package}>={minimum}, which is not installed. Install the CUDA "
                "requirements (requirements-cuda.txt). The worker will not fall back to another precision.") from None
        if _version_tuple(installed) < _version_tuple(minimum):
            raise QuantizationUnavailable(f"--quantization {mode} needs {package}>={minimum}; found {installed}")
    try:
        from transformers.integrations import validate_bnb_backend_availability
        validate_bnb_backend_availability(raise_exception=True)
    except Exception as error:
        raise QuantizationUnavailable(
            f"bitsandbytes cannot run on this host: {type(error).__name__}: {error}") from error


def make_config(mode, compute_dtype):
    """The Transformers quantization config for ``mode``; the recipe is fixed so evidence is comparable."""
    if validate_mode(mode) == "none":
        return None
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=compute_dtype,
                              bnb_4bit_use_double_quant=True)


def _state_bytes(state):
    """Bytes held by a bitsandbytes quantization state, including its nested (second-level) state."""
    total = 0
    for name in ("absmax", "code", "offset"):
        tensor = getattr(state, name, None)
        if hasattr(tensor, "numel"):
            total += tensor.numel() * tensor.element_size()
    nested = getattr(state, "state2", None)
    return total + (_state_bytes(nested) if nested is not None else 0)


def resident_weight_bytes(model):
    """Bytes the model's weights occupy, counting bitsandbytes scale tables that ``parameters()`` omits."""
    total = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
    for module in model.modules():
        state = getattr(getattr(module, "weight", None), "quant_state", None)
        if state is not None:
            total += _state_bytes(state)
    return total


def quantized_layer_count(model):
    return sum(1 for module in model.modules() if getattr(getattr(module, "weight", None), "quant_state", None)
               is not None)


def describe(model, mode, compute_dtype):
    """Provenance for evidence: what was quantized, how, with which library versions."""
    if validate_mode(mode) == "none":
        return None
    versions = {}
    for package in ("bitsandbytes", "accelerate"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return {"method": "bitsandbytes", "format": "nf4", "double_quantization": True, "block_size": NF4_BLOCK_SIZE,
            "compute_dtype": str(compute_dtype).replace("torch.", ""), "quantized_linear_layers": quantized_layer_count(model),
            "kept_in_compute_dtype": "embeddings, lm_head, norms, biases",
            "weights_resident_bytes": resident_weight_bytes(model), "library_versions": versions,
            "note": "weights are quantized from the original checkpoint at load time; fidelity versus the "
                    "unquantized model is measured separately (tensorpilot.quality)"}


def estimate_resident_bytes(config, mode, dtype_bytes=2):
    """Planning estimate of resident weight bytes from a decoder config; never a measurement.

    Checked against a real load of Qwen2.5-0.5B (457,187,552 bytes measured) in the tests.
    """
    hidden, inter, layers = config.hidden_size, config.intermediate_size, config.num_hidden_layers
    heads = config.num_attention_heads
    kv_heads = getattr(config, "num_key_value_heads", None) or heads
    head_dim = getattr(config, "head_dim", None) or hidden // heads
    attention = hidden * heads * head_dim * 2 + hidden * kv_heads * head_dim * 2
    projections = layers * (attention + 3 * hidden * inter)
    biases = layers * (heads * head_dim + 2 * kv_heads * head_dim) if config.model_type == "qwen2" else 0
    norms = (2 * layers + 1) * hidden
    embedding = config.vocab_size * hidden
    head = 0 if getattr(config, "tie_word_embeddings", False) else embedding
    kept = (biases + norms + embedding + head) * dtype_bytes
    if validate_mode(mode) == "none":
        return int(projections * dtype_bytes + kept)
    return int(projections * NF4_BYTES_PER_WEIGHT + kept)
