"""Model-thread adapter for opt-in MLX continuous batching."""
from dataclasses import dataclass
import threading
import time

from .backend import Token


@dataclass
class Sequence:
    prompt_tokens: int
    detokenizer: object


@dataclass(frozen=True)
class BatchStep:
    events: list
    phase: str
    synchronized_execution_s: float
    occupancy: int

    def __iter__(self):
        return iter(self.events)

    def __len__(self):
        return len(self.events)


class MLXBatchAdapter:
    def __init__(self, backend, *, capacity=4, prefill_step_size=256, generator_factory=None,
                 synchronize=None):
        if type(capacity) is not int or not 1 <= capacity <= 8:
            raise ValueError("batch capacity must be in 1..8")
        if type(prefill_step_size) is not int or prefill_step_size < 1:
            raise ValueError("prefill_step_size must be positive")
        real_mlx_generator = generator_factory is None
        if real_mlx_generator:
            from mlx_lm.generate import BatchGenerator
            generator_factory = BatchGenerator
        if synchronize is None:
            if real_mlx_generator:
                import mlx.core as mx
                synchronize = mx.synchronize
            else:  # CPU conformance adapters do not execute MLX.
                def synchronize():
                    return None
        self.owner = threading.get_ident()
        self.backend = backend
        self.capacity = capacity
        self.sequences = {}
        self.closed = False
        self.synchronize = synchronize
        self.last_insert_profile = None
        self.generator = generator_factory(
            backend.model, completion_batch_size=capacity, prefill_batch_size=capacity,
            prefill_step_size=prefill_step_size,
            stop_tokens=[[token] for token in sorted(backend.tokenizer.eos_token_ids)],
        )

    def _check_owner(self):
        if threading.get_ident() != self.owner:
            raise RuntimeError("batch adapter must stay on its model-owner thread")
        if self.closed:
            raise RuntimeError("batch adapter is closed")

    def insert(self, prompt, max_tokens):
        self._check_owner()
        if len(self.sequences) >= self.capacity:
            raise RuntimeError("batch capacity exhausted")
        if type(max_tokens) is not int or not 1 <= max_tokens <= 512:
            raise ValueError("invalid output budget")
        tokenize_started = time.perf_counter()
        tokens = self.backend.tokenizer.encode(prompt)
        tokenization_s = time.perf_counter() - tokenize_started
        if not 1 <= len(tokens) <= self.backend.max_prompt_tokens:
            raise ValueError("prompt exceeds token limit or encodes empty")
        # TokenizerWrapper returns a new stateful detokenizer on each access.
        detokenizer = self.backend.tokenizer.detokenizer
        self.synchronize()
        insert_started = time.perf_counter()
        uid = self.generator.insert([tokens], max_tokens=[max_tokens])[0]
        self.synchronize()
        insert_s = time.perf_counter() - insert_started
        self.sequences[uid] = Sequence(len(tokens), detokenizer)
        self.last_insert_profile = {"measurement_source": "synchronized_model_thread_wall_clock",
                                    "tokenization_s": tokenization_s,
                                    "batch_insert_s": insert_s,
                                    "prompt_tokens": len(tokens)}
        return uid

    def step(self):
        self._check_owner()
        if not self.sequences:
            return []
        occupancy = len(self.sequences)
        self.synchronize()
        started = time.perf_counter()
        _, responses = self.generator.next()
        self.synchronize()
        execution_s = time.perf_counter() - started
        events = []
        for response in responses:
            sequence = self.sequences[response.uid]
            output = response.token not in self.backend.tokenizer.eos_token_ids
            if output:
                sequence.detokenizer.add_token(response.token)
            if response.finish_reason is not None:
                sequence.detokenizer.finalize()
            events.append((response.uid, Token(
                sequence.detokenizer.last_segment, int(response.token), output,
                sequence.prompt_tokens, response.finish_reason,
            )))
            if response.finish_reason is not None:
                del self.sequences[response.uid]
        # An empty result can mean prefill progress, not end-of-generation.
        return BatchStep(events, "decode" if events else "prefill", execution_s, occupancy)

    def remove(self, uid):
        self._check_owner()
        if uid not in self.sequences:
            return False
        self.generator.remove([uid])
        del self.sequences[uid]
        return True

    def close(self):
        if threading.get_ident() != self.owner:
            raise RuntimeError("batch cleanup must run on its model-owner thread")
        if self.closed:
            return
        if self.sequences:
            self.generator.remove(list(self.sequences))
            self.sequences.clear()
        self.generator.close()
        self.closed = True
