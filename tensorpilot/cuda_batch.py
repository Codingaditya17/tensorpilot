"""Continuous batching for the CUDA backend: a batched forward pass over a fixed KV slot pool.

This is model-level batching. Every decode step runs ONE forward pass over all active sequences, each
with its own position, length and KV row. Sequences join and leave between steps, so a finished request
frees its slot for the next waiting one without draining the batch.

What it is, precisely:
  * a preallocated slot pool: ``capacity`` rows (plus one staging row) of ``row_tokens`` KV positions per
    layer, allocated once at startup. KV memory is therefore known and fixed; there is no per-step
    allocation, ``torch.cat`` or padding shuffle. It is NOT paged: every active sequence reserves a full
    row, so short sequences waste the unused tail of their row.
  * decode rows kept compact in ``[0, n)``; a finished row is replaced by moving the last live row into
    its slot (only its filled prefix is copied).
  * prefill in chunks of ``prefill_step_size`` tokens on the staging row, interleaved with decode steps so
    a long prompt delays running streams by at most one chunk. The finished row is copied into a decode
    slot.
  * grouped-query attention without expanding K/V: the query-head group is folded into the query
    sequence dimension, so SDPA reads each KV head once.
  * greedy sampling, one device-to-host transfer per step.

What it is not: no paged KV, no prefix cache, no CUDA graphs, no speculative decoding, no quantization.
The forward pass is written for the Qwen2 and Llama decoder layouts only; any other architecture is
refused at construction instead of being assumed to work.
"""
import sys
import threading
import time
from dataclasses import dataclass, field

from .admission import ResourceAdmission
from .backend import Token
from .batch_backend import BatchStep

SUPPORTED_MODEL_TYPES = ("qwen2", "llama")
GIB = 1024**3
# Free memory guaranteed inside the allocator cap after weights and the KV pool: activations, allocator fragmentation and any
# quantization scratch. By construction the planner never grants a pool that leaves less than this. It is a planning margin,
# not a measurement: the stage runner checks the real margin on the GPU before it benchmarks (tensorpilot.memory_gate).
DEFAULT_HEADROOM_BYTES = int(2.5 * GIB)


class UnsupportedModel(ValueError):
    """The loaded checkpoint cannot be served by the batched forward pass."""


@dataclass(frozen=True)
class ModelGeometry:
    layers: int
    q_heads: int
    kv_heads: int
    head_dim: int
    model_type: str

    @property
    def group(self):
        return self.q_heads // self.kv_heads

    def kv_bytes_per_token(self, dtype_bytes):
        """Keys plus values, all layers, one token."""
        return 2 * self.layers * self.kv_heads * self.head_dim * dtype_bytes


def read_geometry(config):
    """Validate that ``config`` is a supported full-attention decoder and return its geometry."""
    model_type = getattr(config, "model_type", None)
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise UnsupportedModel(f"batched CUDA decoding supports {SUPPORTED_MODEL_TYPES}; got {model_type!r}")
    if getattr(config, "use_sliding_window", False):
        raise UnsupportedModel("sliding-window attention is not implemented in the batched forward pass")
    layer_types = getattr(config, "layer_types", None)
    if layer_types is not None and any(kind != "full_attention" for kind in layer_types):
        raise UnsupportedModel("only full-attention layers are implemented in the batched forward pass")
    heads = config.num_attention_heads
    kv_heads = getattr(config, "num_key_value_heads", None) or heads
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // heads
    if heads % kv_heads:
        raise UnsupportedModel("num_attention_heads must be a multiple of num_key_value_heads")
    return ModelGeometry(config.num_hidden_layers, heads, kv_heads, head_dim, model_type)


@dataclass(frozen=True)
class PoolPlan:
    capacity: int
    row_tokens: int
    bytes_per_token: int
    pool_bytes: int
    budget_bytes: int
    weights_bytes: int
    limited_by: str


def plan_kv_pool(*, weights_bytes, bytes_per_token, row_tokens, requested_capacity,
                 total_vram_bytes=None, memory_fraction=0.9, headroom_bytes=DEFAULT_HEADROOM_BYTES,
                 kv_budget_bytes=None):
    """Choose how many sequence slots fit, from measured weight bytes and the allocator cap.

    The pool is sized so weights + pool + headroom stay under ``memory_fraction`` of the device. The
    plan never silently exceeds it: if not even one slot fits, it raises.
    """
    if requested_capacity < 1 or row_tokens < 1 or bytes_per_token < 1:
        raise ValueError("capacity, row tokens and bytes per token must be positive")
    row_bytes = row_tokens * bytes_per_token
    if total_vram_bytes is not None:
        available = int(total_vram_bytes * memory_fraction) - weights_bytes - headroom_bytes
        limit_name = "vram"
    else:
        available, limit_name = None, "explicit"
    if kv_budget_bytes is not None:
        budget = kv_budget_bytes if available is None else min(kv_budget_bytes, available)
        limit_name = "explicit" if available is None or kv_budget_bytes <= available else "vram"
    elif available is not None:
        budget = available
    else:
        raise ValueError("give total_vram_bytes or kv_budget_bytes")
    fit = budget // row_bytes - 1  # one row is the prefill staging row
    if fit < 1:
        raise ValueError(
            f"no KV slot fits: weights {weights_bytes / GIB:.2f} GiB, one row of {row_tokens} tokens needs "
            f"{row_bytes / GIB:.3f} GiB, KV budget is {max(budget, 0) / GIB:.2f} GiB")
    capacity = min(requested_capacity, fit)
    if capacity < requested_capacity:
        reason = limit_name
    else:
        reason = "requested"
    return PoolPlan(capacity, row_tokens, bytes_per_token, (capacity + 1) * row_bytes, budget,
                    weights_bytes, reason)


class IncrementalDetokenizer:
    """Per-sequence text deltas without re-decoding the whole output at every token.

    Only a short window of token ids is decoded. A delta that ends in an incomplete UTF-8 sequence is held
    back until the next token completes it, and ``flush`` emits whatever is pending at the end.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.ids = []
        self.prefix = 0
        self.read = 0

    def add(self, token_id):
        self.ids.append(token_id)
        return self._delta(final=False)

    def flush(self):
        return self._delta(final=True)

    def _delta(self, *, final):
        if self.read >= len(self.ids):
            return ""
        before = self.tokenizer.decode(self.ids[self.prefix:self.read])
        text = self.tokenizer.decode(self.ids[self.prefix:])
        if len(text) <= len(before) or (text.endswith("�") and not final):
            return ""
        delta = text[len(before):]
        self.prefix, self.read = self.read, len(self.ids)
        return delta


@dataclass
class _Sequence:
    uid: int
    prompt_ids: list
    max_tokens: int
    detokenizer: IncrementalDetokenizer
    prefilled: int = 0
    generated: int = 0
    length: int = 0           # KV positions filled
    next_token: int = -1
    slot: int | None = None
    created_s: float = field(default_factory=time.perf_counter)


class CUDABatchAdapter:
    """Adapter for ``BatchEngine``: ``insert`` / ``step`` / ``remove`` / ``close`` on the model thread."""

    MAX_CAPACITY = 256

    def __init__(self, backend, *, capacity=8, prefill_step_size=256, row_tokens=None,
                 kv_budget_bytes=None, headroom_bytes=DEFAULT_HEADROOM_BYTES):
        import torch
        self.torch = torch
        if type(capacity) is not int or not 1 <= capacity <= self.MAX_CAPACITY:
            raise ValueError(f"batch capacity must be in 1..{self.MAX_CAPACITY}")
        if type(prefill_step_size) is not int or prefill_step_size < 1:
            raise ValueError("prefill_step_size must be positive")
        model = backend.model
        self.geometry = read_geometry(model.config)
        module = sys.modules[type(model).__module__]
        self._apply_rope = module.apply_rotary_pos_emb
        self.owner = threading.get_ident()
        self.backend = backend
        self.model = model
        self.device = backend.device
        # The backend's compute dtype, not the first parameter's: a quantized model's first weight is packed uint8.
        self.dtype = getattr(backend, "compute_dtype", None) or next(model.parameters()).dtype
        self.prefill_step_size = prefill_step_size
        context = backend.context_tokens
        wanted = backend.max_prompt_tokens + 512
        self.row_tokens = min(row_tokens or wanted, context) if context else (row_tokens or wanted)
        dtype_bytes = torch.empty((), dtype=self.dtype).element_size()
        self.bytes_per_token = self.geometry.kv_bytes_per_token(dtype_bytes)
        weights_bytes = getattr(backend, "weights_resident_bytes", None) or sum(
            p.numel() * p.element_size() for p in model.parameters())
        total = None
        if self.device.type == "cuda":
            total = torch.cuda.get_device_properties(self.device).total_memory
            # Anything the allocator already holds (quantization scratch included) counts against the pool.
            weights_bytes = max(weights_bytes, torch.cuda.memory_allocated(self.device))
        fraction = backend.provenance.get("cuda_memory_fraction") or 0.9
        self.plan = plan_kv_pool(weights_bytes=weights_bytes, bytes_per_token=self.bytes_per_token,
                                 row_tokens=self.row_tokens, requested_capacity=capacity,
                                 total_vram_bytes=total, memory_fraction=fraction,
                                 headroom_bytes=headroom_bytes, kv_budget_bytes=kv_budget_bytes)
        self.capacity = self.plan.capacity
        self.closed = False
        self.sequences = {}          # uid -> _Sequence (every inserted, unfinished sequence)
        self.prefill_queue = []      # uids waiting for or in prefill, FIFO
        self.decoding = []           # uids in decode slot order; index == pool row
        self.next_uid = 0
        self.last_phase = "decode"
        self.last_insert_profile = None
        self.counters = {"decode_steps": 0, "prefill_chunks": 0, "decode_tokens": 0, "prefill_tokens": 0,
                         "decode_execution_s": 0.0, "prefill_execution_s": 0.0, "max_decode_width": 0,
                         "sequences_finished": 0, "sequences_removed": 0, "slot_moves": 0}
        with torch.inference_mode():
            shape = (self.capacity + 1, self.geometry.kv_heads, self.row_tokens, self.geometry.head_dim)
            # Zeros, not empty: masked columns are multiplied by probability 0, and 0 * NaN would poison
            # the output if a freshly allocated block happened to contain NaN bit patterns.
            self.keys = [torch.zeros(shape, dtype=self.dtype, device=self.device)
                         for _ in range(self.geometry.layers)]
            self.values = [torch.zeros(shape, dtype=self.dtype, device=self.device)
                           for _ in range(self.geometry.layers)]
        self.staging = self.capacity

    # ----- adapter contract -------------------------------------------------------------------------
    def _check_owner(self):
        if threading.get_ident() != self.owner:
            raise RuntimeError("batch adapter must stay on its model-owner thread")
        if self.closed:
            raise RuntimeError("batch adapter is closed")

    def build_admission(self, queue_size):
        """Admission accounting in real bytes: every admitted sequence reserves one full pool row."""
        row_bytes = self.row_tokens * self.bytes_per_token
        return ResourceAdmission((self.capacity + queue_size) * row_bytes,
                                 bytes_per_token=self.bytes_per_token, sequence_overhead_bytes=1)

    def insert(self, prompt, max_tokens):
        self._check_owner()
        if len(self.sequences) >= self.capacity:
            raise RuntimeError("batch capacity exhausted")
        if type(max_tokens) is not int or not 1 <= max_tokens <= 512:
            raise ValueError("invalid output budget")
        started = time.perf_counter()
        ids = self.backend.tokenizer.encode(prompt)
        tokenization_s = time.perf_counter() - started
        if not 1 <= len(ids) <= self.backend.max_prompt_tokens:
            raise ValueError("prompt exceeds token limit or encodes empty")
        if len(ids) + max_tokens > self.row_tokens:
            raise ValueError("prompt plus max_tokens exceeds the KV row length")
        uid = self.next_uid
        self.next_uid += 1
        self.sequences[uid] = _Sequence(uid, ids, max_tokens, IncrementalDetokenizer(self.backend.tokenizer))
        self.prefill_queue.append(uid)
        self.last_insert_profile = {"measurement_source": "model_thread_wall_clock",
                                    "tokenization_s": tokenization_s, "batch_insert_s": 0.0,
                                    "prompt_tokens": len(ids)}
        return uid

    def step(self):
        self._check_owner()
        occupancy = len(self.sequences)
        if not occupancy:
            return []
        torch = self.torch
        if not self.decoding and not self.prefill_queue:
            return []
        with torch.inference_mode():
            run_prefill = bool(self.prefill_queue) and (not self.decoding or self.last_phase == "decode")
            if run_prefill:
                events, execution_s = self._prefill_chunk()
                phase = "prefill"
            else:
                events, execution_s = self._decode_step()
                phase = "decode"
        self.last_phase = phase
        return BatchStep(events, phase, execution_s, occupancy)

    def remove(self, uid):
        self._check_owner()
        sequence = self.sequences.get(uid)
        if sequence is None:
            return False
        if uid in self.prefill_queue:
            self.prefill_queue.remove(uid)
        elif uid in self.decoding:
            with self.torch.inference_mode():
                self._release_slots([self.decoding.index(uid)])
        del self.sequences[uid]
        self.counters["sequences_removed"] += 1
        return True

    def close(self):
        if threading.get_ident() != self.owner:
            raise RuntimeError("batch cleanup must run on its model-owner thread")
        if self.closed:
            return
        self.sequences.clear()
        self.prefill_queue.clear()
        self.decoding.clear()
        self.keys, self.values = [], []
        self.closed = True
        if self.device.type == "cuda":
            self.torch.cuda.empty_cache()

    def snapshot(self):
        """Measured pool state for /health. KV bytes in use count filled positions, not reserved rows."""
        # Called from the event-loop thread while the model thread mutates state: read, never index.
        lengths = [sequence.length for sequence in map(self.sequences.get, list(self.decoding)) if sequence]
        in_use = sum(lengths) * self.bytes_per_token
        return {"kind": "fixed_slot_kv_pool_not_paged",
                "capacity_rows": self.capacity, "row_tokens": self.row_tokens,
                "bytes_per_token": self.bytes_per_token, "pool_bytes_allocated": self.plan.pool_bytes,
                "pool_limited_by": self.plan.limited_by,
                "decode_rows_active": len(self.decoding), "prefill_waiting": len(self.prefill_queue),
                "kv_bytes_filled": in_use,
                "kv_fill_ratio_of_active_rows": (sum(lengths) / (len(lengths) * self.row_tokens)
                                                 if lengths else 0.0),
                "weights_bytes": self.plan.weights_bytes, **self.counters}

    # ----- internals --------------------------------------------------------------------------------
    def _sync(self):
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def _forward(self, input_ids, positions, mask, write, kv_rows, width, last_only):
        """Run all layers. ``write(layer, k, v)`` stores new K/V; attention reads ``kv_rows`` up to ``width``."""
        torch = self.torch
        geometry = self.geometry
        batch, count = input_ids.shape
        backbone = self.model.model
        hidden = backbone.embed_tokens(input_ids)
        cos, sin = backbone.rotary_emb(hidden, positions)
        group = geometry.group
        for index, layer in enumerate(backbone.layers):
            residual = hidden
            x = layer.input_layernorm(hidden)
            attention = layer.self_attn
            q = attention.q_proj(x).view(batch, count, geometry.q_heads, geometry.head_dim).transpose(1, 2)
            k = attention.k_proj(x).view(batch, count, geometry.kv_heads, geometry.head_dim).transpose(1, 2)
            v = attention.v_proj(x).view(batch, count, geometry.kv_heads, geometry.head_dim).transpose(1, 2)
            q, k = self._apply_rope(q, k, cos, sin)
            write(index, k, v)
            keys = self.keys[index][kv_rows, :, :width]
            values = self.values[index][kv_rows, :, :width]
            # Fold the query-head group into the query length so each KV head is read once (no K/V copy).
            folded = q.reshape(batch, geometry.kv_heads, group * count, geometry.head_dim)
            out = torch.nn.functional.scaled_dot_product_attention(
                folded, keys, values, attn_mask=mask.expand(batch, 1, group * count, width) if count == 1
                else mask.repeat(1, 1, group, 1), scale=attention.scaling)
            out = out.reshape(batch, geometry.q_heads, count, geometry.head_dim).transpose(1, 2)
            hidden = residual + attention.o_proj(out.reshape(batch, count, -1))
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
        hidden = backbone.norm(hidden)
        if last_only:
            hidden = hidden[:, -1:, :]
        return self.model.lm_head(hidden)

    def _guard_oom(self, error):
        self.backend.oom_events += 1
        if self.device.type == "cuda":
            self.torch.cuda.empty_cache()
        raise error

    def _prefill_chunk(self):
        torch = self.torch
        sequence = self.sequences[self.prefill_queue[0]]
        start = sequence.prefilled
        chunk = sequence.prompt_ids[start:start + self.prefill_step_size]
        count, width = len(chunk), start + len(chunk)
        self._sync()
        began = time.perf_counter()
        ids = torch.tensor([chunk], dtype=torch.long, device=self.device)
        positions = torch.arange(start, width, dtype=torch.long, device=self.device).unsqueeze(0)
        mask = torch.tril(torch.ones(count, width, dtype=torch.bool, device=self.device),
                          diagonal=start).view(1, 1, count, width)
        stage = self.staging

        def write(layer, k, v):
            self.keys[layer][stage, :, start:width] = k[0]
            self.values[layer][stage, :, start:width] = v[0]

        try:
            logits = self._forward(ids, positions, mask, write, slice(stage, stage + 1), width, last_only=True)
        except self.torch.OutOfMemoryError as error:
            self._guard_oom(error)
        sequence.prefilled = width
        sequence.length = width
        finished_prompt = width == len(sequence.prompt_ids)
        token = int(torch.argmax(logits[0, -1]).item()) if finished_prompt else None
        self._sync()
        execution_s = time.perf_counter() - began
        self.counters["prefill_chunks"] += 1
        self.counters["prefill_tokens"] += count
        self.counters["prefill_execution_s"] += execution_s
        events = []
        if finished_prompt:
            self.prefill_queue.pop(0)
            events.append((sequence.uid, self._emit(sequence, token)))
            if sequence.uid in self.sequences:  # not finished by its first token
                self._join_decode(sequence)
        return events, execution_s

    def _join_decode(self, sequence):
        slot = len(self.decoding)
        for layer in range(self.geometry.layers):
            self.keys[layer][slot, :, :sequence.length] = self.keys[layer][self.staging, :, :sequence.length]
            self.values[layer][slot, :, :sequence.length] = self.values[layer][self.staging, :, :sequence.length]
        sequence.slot = slot
        self.decoding.append(sequence.uid)

    def _decode_step(self):
        torch = self.torch
        rows = len(self.decoding)
        sequences = [self.sequences[uid] for uid in self.decoding]
        lengths = [sequence.length for sequence in sequences]
        width = max(lengths) + 1
        self._sync()
        began = time.perf_counter()
        ids = torch.tensor([[sequence.next_token] for sequence in sequences], dtype=torch.long, device=self.device)
        length_tensor = torch.tensor(lengths, dtype=torch.long, device=self.device)
        positions = length_tensor.unsqueeze(1)
        columns = torch.arange(width, dtype=torch.long, device=self.device)
        mask = (columns.unsqueeze(0) <= length_tensor.unsqueeze(1)).view(rows, 1, 1, width)
        row_index = torch.arange(rows, dtype=torch.long, device=self.device)

        def write(layer, k, v):
            self.keys[layer][row_index, :, length_tensor] = k[:, :, 0]
            self.values[layer][row_index, :, length_tensor] = v[:, :, 0]

        try:
            logits = self._forward(ids, positions, mask, write, slice(0, rows), width, last_only=True)
        except self.torch.OutOfMemoryError as error:
            self._guard_oom(error)
        sampled = torch.argmax(logits[:, -1], dim=-1).tolist()  # the step's only host synchronization
        self._sync()
        execution_s = time.perf_counter() - began
        events, finished_slots = [], []
        for slot, (sequence, token) in enumerate(zip(sequences, sampled)):
            sequence.length += 1
            event = self._emit(sequence, token)
            events.append((sequence.uid, event))
            if event.finish_reason is not None:
                finished_slots.append(slot)
        if finished_slots:
            self._release_slots(finished_slots)
        self.counters["decode_steps"] += 1
        self.counters["decode_tokens"] += rows
        self.counters["decode_execution_s"] += execution_s
        self.counters["max_decode_width"] = max(self.counters["max_decode_width"], rows)
        return events, execution_s

    def _emit(self, sequence, token):
        """Account one sampled token and build its event, mirroring the serial backend's finish rules."""
        eos = token in self.backend.eos_token_ids
        sequence.generated += 1
        last = eos or sequence.generated >= sequence.max_tokens
        text = ""
        if not eos:
            text = sequence.detokenizer.add(token)
        if last:
            text += sequence.detokenizer.flush()
        sequence.next_token = token
        finish = ("stop" if eos else "length") if last else None
        if last:
            self.counters["sequences_finished"] += 1
            # Finished sequences leave the adapter before their final event is returned.
            self.sequences.pop(sequence.uid, None)
        return Token(text, token, not eos, len(sequence.prompt_ids), finish)

    def _release_slots(self, slots):
        """Free decode slots and compact: surviving tail rows move into the holes."""
        before = len(self.decoding)
        gone = set(slots)
        new_count = before - len(gone)
        holes = [slot for slot in sorted(gone) if slot < new_count]
        movers = [slot for slot in range(new_count, before) if slot not in gone]
        assert len(holes) == len(movers)
        for hole, mover in zip(holes, movers):
            uid = self.decoding[mover]
            sequence = self.sequences.get(uid)
            filled = sequence.length if sequence is not None else 0
            if filled:
                for layer in range(self.geometry.layers):
                    self.keys[layer][hole, :, :filled] = self.keys[layer][mover, :, :filled]
                    self.values[layer][hole, :, :filled] = self.values[layer][mover, :, :filled]
            self.counters["slot_moves"] += 1
            self.decoding[hole] = uid
            if sequence is not None:
                sequence.slot = hole
        del self.decoding[new_count:]


class CUDABatchAdapterFactory:
    """Callable handed to ``BatchEngine``; carries the options the engine does not know about."""

    MAX_CAPACITY = CUDABatchAdapter.MAX_CAPACITY

    def __init__(self, *, row_tokens=None, kv_budget_bytes=None, headroom_bytes=DEFAULT_HEADROOM_BYTES):
        self.options = {"row_tokens": row_tokens, "kv_budget_bytes": kv_budget_bytes,
                        "headroom_bytes": headroom_bytes}

    def __call__(self, backend, *, capacity, prefill_step_size):
        return CUDABatchAdapter(backend, capacity=capacity, prefill_step_size=prefill_step_size,
                                **self.options)
