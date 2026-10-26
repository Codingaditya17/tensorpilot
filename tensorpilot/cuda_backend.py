"""Optional Linux/NVIDIA backend: Hugging Face Transformers on PyTorch CUDA.

Decoding is greedy, matching the MLX backend's temperature-0 sampler. The loop
owns an explicit KV cache and yields one token at a time, so every generated
token passes through the engine's cancellation and deadline checkpoints. A
library ``generate`` call could not be interrupted between tokens that way.

PyTorch and Transformers are imported lazily: this module imports on machines
without the CUDA stack, and only constructing a backend requires it.
"""
from .artifacts import resolve_model
from .backend import Token

DTYPES = ("float16", "bfloat16", "float32")
DEFAULT_MEMORY_FRACTION = .9


def cuda_device_summary(torch, device):
    """Static identity of one CUDA device; nothing here is a utilization sample."""
    properties = torch.cuda.get_device_properties(device)
    return {"name": properties.name,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "total_memory_bytes": properties.total_memory,
            "multi_processor_count": properties.multi_processor_count}


def resolve_cuda_device(torch, device, *, allow_cpu=False):
    """Return a ``torch.device`` with an explicit index for a CUDA request.

    PyTorch allocator APIs such as ``set_per_process_memory_fraction`` reject a bare ``cuda`` device.
    ``cuda`` resolves to the current device and ``cuda:N`` is range-checked against what PyTorch can
    see. Indices are PyTorch's own, so they already respect ``CUDA_VISIBLE_DEVICES``, which renumbers
    the visible devices from 0.
    """
    try:
        parsed = torch.device(device)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid device {device!r}: expected 'cuda' or 'cuda:<index>'") from exc
    if parsed.type != "cuda":
        if allow_cpu and parsed.type == "cpu":
            return parsed
        raise ValueError("device must be CUDA; CPU is accepted only by CPU conformance tests")
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot see a CUDA device; check the NVIDIA driver and container runtime")
    count = torch.cuda.device_count()
    index = parsed.index if parsed.index is not None else torch.cuda.current_device()
    if not 0 <= index < count:
        raise ValueError(f"CUDA device index {index} is out of range: PyTorch sees {count} device(s) "
                         "(CUDA_VISIBLE_DEVICES renumbers the visible devices from 0)")
    return torch.device("cuda", index)


class CUDABackend:
    backend_name = "cuda"

    def __init__(self, model_name, *, max_prompt_tokens=2048, revision=None, offline=False,
                 device="cuda", dtype="float16", memory_fraction=DEFAULT_MEMORY_FRACTION,
                 cache_dir=None, allow_cpu=False):
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if dtype not in DTYPES:
            raise ValueError(f"dtype must be one of {DTYPES}")
        if type(max_prompt_tokens) is not int or max_prompt_tokens < 1:
            raise ValueError("max_prompt_tokens must be a positive integer")
        if type(memory_fraction) not in (int, float) or not 0 < memory_fraction <= 1:
            raise ValueError("memory_fraction must be in (0, 1]")
        self.torch = torch
        self.device = resolve_cuda_device(torch, device, allow_cpu=allow_cpu)
        if self.device.type == "cuda":
            # A per-process cap turns runaway allocation into a per-request OOM instead of
            # starving other processes sharing the GPU. The device must carry an explicit index.
            torch.cuda.set_per_process_memory_fraction(memory_fraction, self.device)
        self.max_prompt_tokens = max_prompt_tokens
        self.oom_events = 0
        path, self.provenance = resolve_model(model_name, revision=revision, offline=offline,
                                              cache_dir=cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=getattr(torch, dtype))
        self.model = self.model.to(self.device).eval()
        self.eos_token_ids = _eos_token_ids(self.model, self.tokenizer)
        context = getattr(self.model.config, "max_position_embeddings", None)
        self.context_tokens = int(context) if isinstance(context, int) and context > 0 else None
        gpu = cuda_device_summary(torch, self.device) if self.device.type == "cuda" else None
        self.provenance.update({
            "backend": self.backend_name, "device": str(self.device), "gpu": gpu,
            "model_dtype": dtype, "cuda_memory_fraction": memory_fraction if gpu else None,
            "cuda_runtime": torch.version.cuda, "cudnn_version": torch.backends.cudnn.version(),
            "torch_version": torch.__version__, "transformers_version": transformers.__version__,
            "decoding": "greedy, explicit KV cache, one sequence per backend",
            # Checkpoint sampling defaults (for example repetition penalty) are not applied.
            "generation_config_applied": False, "context_tokens": self.context_tokens,
        })

    def stream(self, prompt, max_tokens):
        torch = self.torch
        ids = self.tokenizer.encode(prompt)
        if not 1 <= len(ids) <= self.max_prompt_tokens:
            raise ValueError("prompt exceeds token limit or encodes empty")
        if self.context_tokens is not None and len(ids) + max_tokens > self.context_tokens:
            # Reject before the first forward pass rather than failing partway through a stream.
            raise ValueError("prompt plus max_tokens exceeds the model context window")
        generated, emitted = [], ""
        outputs = past = None
        try:
            with torch.inference_mode():
                # One unpadded sequence: the mask is all ones and grows by one per decoded token.
                mask = torch.ones((1, len(ids)), dtype=torch.long, device=self.device)
                outputs = self._forward(torch.tensor([ids], device=self.device), mask, None)
                past, logits = outputs.past_key_values, outputs.logits[:, -1, :]
                for step in range(max_tokens):
                    token_id = int(torch.argmax(logits, dim=-1).item())
                    is_eos = token_id in self.eos_token_ids
                    last = is_eos or step == max_tokens - 1
                    finish = ("stop" if is_eos else "length") if last else None
                    if not is_eos:
                        generated.append(token_id)
                    # EOS is yielded for protocol completeness but never counted as output.
                    text, emitted = self._text_delta(generated, emitted, final=last)
                    yield Token(text, token_id, not is_eos, len(ids), finish)
                    if last:
                        return
                    mask = torch.cat([mask, torch.ones((1, 1), dtype=torch.long, device=self.device)], dim=1)
                    outputs = self._forward(torch.tensor([[token_id]], device=self.device), mask, past)
                    past, logits = outputs.past_key_values, outputs.logits[:, -1, :]
        finally:
            # Runs on normal completion, error, deadline and generator close. Drop the cache
            # references so the caching allocator can reuse the memory for the next request.
            outputs = past = logits = None

    def _forward(self, input_ids, attention_mask, past):
        try:
            return self.model(input_ids=input_ids, attention_mask=attention_mask,
                              past_key_values=past, use_cache=True)
        except self.torch.OutOfMemoryError:
            # The request fails with a bounded error; the worker stays up for the next one.
            self.oom_events += 1
            if self.device.type == "cuda":
                self.torch.cuda.empty_cache()
            raise

    def _text_delta(self, generated, emitted, *, final):
        """Return newly printable text. Incomplete UTF-8 is held back until its bytes arrive."""
        text = self.tokenizer.decode(generated)
        if not final and text.endswith("�"):
            return "", emitted
        if not text.startswith(emitted):
            # Detokenizer rewrites are not expected; resynchronize without duplicating text.
            return "", text
        return text[len(emitted):], text

    def memory(self):
        torch = self.torch
        if self.device.type != "cuda":
            cuda = {"cuda_allocated_bytes": None, "cuda_peak_allocated_bytes": None,
                    "cuda_reserved_bytes": None, "cuda_peak_reserved_bytes": None}
        else:
            cuda = {"cuda_allocated_bytes": torch.cuda.memory_allocated(self.device),
                    "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device),
                    "cuda_reserved_bytes": torch.cuda.memory_reserved(self.device),
                    "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(self.device)}
        return {**cuda, "cuda_oom_events": self.oom_events}


def _eos_token_ids(model, tokenizer):
    ids = set()
    raw = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if isinstance(raw, int):
        ids.add(raw)
    elif raw:
        ids.update(int(value) for value in raw)
    if tokenizer.eos_token_id is not None:
        ids.add(int(tokenizer.eos_token_id))
    return frozenset(ids)
