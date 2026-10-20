"""Backend serving contract plus the MLX implementation.

Model access stays on the engine's single dedicated thread. Every backend, MLX or
CUDA, must satisfy ``InferenceBackend``: the serial ``Engine`` relies only on the
members listed there, so a new runtime plugs in without changing scheduling,
admission, cancellation or the HTTP streaming contract.
"""
from dataclasses import dataclass
from contextlib import closing
from typing import Iterator, Protocol

from .artifacts import resolve_model

BACKENDS = ("mlx", "cuda")


@dataclass
class Token:
    text: str
    token_id: int
    output: bool
    prompt_tokens: int | None = None
    finish_reason: str | None = None
    memory_profile: dict | None = None


class InferenceBackend(Protocol):
    """Members the serving engine uses; all calls run on the model-owner thread."""

    backend_name: str
    provenance: dict
    max_prompt_tokens: int

    def stream(self, prompt: str, max_tokens: int) -> Iterator[Token]:
        """Yield one Token per generated token; closing the iterator must release it."""

    def memory(self) -> dict:
        """Return a memory sample. Keys a runtime does not measure are reported as None."""


def cache_profile(cache, mx):
    """Use cache-owned array bytes; do not estimate from process allocator deltas."""
    try:
        allocated = sum(layer.nbytes for layer in cache)
    except (AttributeError, NotImplementedError):
        allocated = None
    return {"cache_allocated_bytes": allocated, "cache_layers": len(cache),
            "mlx_active_bytes": mx.get_active_memory()}


class MLXBackend:
    backend_name = "mlx"

    def __init__(self, model_name, memory_bytes, max_prompt_tokens, *, revision=None, offline=False,
                 profile_memory=False, cache_dir=None):
        import mlx.core as mx
        from mlx_lm import load
        self.mx = mx
        mx.set_memory_limit(memory_bytes)
        mx.set_cache_limit(min(memory_bytes // 8, 512 * 1024**2))
        path, self.provenance = resolve_model(model_name, revision=revision, offline=offline,
                                              cache_dir=cache_dir)
        self.provenance["mlx_device"] = mx.device_info()
        self.provenance["memory_profiling"] = profile_memory
        self.profile_memory = profile_memory
        self.model, self.tokenizer = load(str(path))
        self.max_prompt_tokens = max_prompt_tokens

    def stream(self, prompt, max_tokens):
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler
        from mlx_lm.models.cache import make_prompt_cache
        # Explicitly encode once; no implicit chat template in this baseline.
        ids = self.tokenizer.encode(prompt)
        if len(ids) > self.max_prompt_tokens:
            raise ValueError("prompt exceeds token limit")
        cache = make_prompt_cache(self.model) if self.profile_memory else None
        with closing(stream_generate(
            self.model, self.tokenizer, prompt=ids, max_tokens=max_tokens,
            sampler=make_sampler(temp=0),
            prompt_cache=cache,
        )) as responses:
            for response in responses:
                profile = cache_profile(cache, self.mx) if cache is not None else None
                if profile is not None:
                    profile["sampled_tokens_including_eos"] = response.generation_tokens
                # MLX's final response may contain EOS. Count it separately.
                yield Token(response.text, int(response.token),
                            response.token not in self.tokenizer.eos_token_ids,
                            response.prompt_tokens, response.finish_reason, profile)

    def memory(self):
        return {"mlx_active_bytes": self.mx.get_active_memory(),
                "mlx_peak_bytes_process": self.mx.get_peak_memory()}
