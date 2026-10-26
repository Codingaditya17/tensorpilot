"""CUDA continuous batching: correctness against the serial backend, run on CPU with tiny random models.

These prove the batched forward pass, slot pool and scheduling logic. They say nothing about GPU speed,
CUDA kernels or fp16/bf16 numerics; those need real hardware.
"""
import random
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from tiny_model import make_tiny_model_dir, make_tiny_qwen2_model, save_tiny_qwen2_model  # noqa: E402
from tensorpilot import quantization as quant  # noqa: E402
from tensorpilot.cuda_backend import CUDABackend  # noqa: E402
from tensorpilot.cuda_batch import (CUDABatchAdapter, IncrementalDetokenizer, UnsupportedModel,  # noqa: E402
                                    plan_kv_pool, read_geometry)

MiB = 1024**2
WORDS = [f"w{i}" for i in range(60)] + ["hello", "world", "the", "a"]


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    return make_tiny_model_dir(tmp_path_factory.mktemp("batch-model"))


@pytest.fixture(scope="module")
def qwen2_dir(tmp_path_factory):
    return save_tiny_qwen2_model(tmp_path_factory.mktemp("batch-qwen2"))


def make_backend(model_dir, architecture, qwen2_dir=None):
    """architecture: llama | qwen2, optionally suffixed -nf4 for bitsandbytes NF4 weights (CPU, fp32 compute)."""
    base, _, mode = architecture.partition("-")
    mode = mode or "none"
    backend = CUDABackend(str(model_dir), device="cpu", allow_cpu=True, dtype="float32", quantization=mode)
    if base == "qwen2":
        from transformers import AutoModelForCausalLM
        if mode == "none":
            backend.model = make_tiny_qwen2_model()
        else:
            backend.model = AutoModelForCausalLM.from_pretrained(
                qwen2_dir, dtype=torch.float32, device_map={"": "cpu"},
                quantization_config=quant.make_config(mode, torch.float32)).eval()
        backend.weights_resident_bytes = quant.resident_weight_bytes(backend.model)
    return backend


def make_adapter(backend, *, capacity=8, prefill_step_size=16, **options):
    options.setdefault("kv_budget_bytes", 64 * MiB)
    return CUDABatchAdapter(backend, capacity=capacity, prefill_step_size=prefill_step_size, **options)


def prompt_of(length, seed):
    rng = random.Random(seed)
    return " ".join(rng.choice(WORDS) for _ in range(length))


def serial(backend, prompt, max_tokens):
    tokens = list(backend.stream(prompt, max_tokens))
    return {"ids": [t.token_id for t in tokens if t.output], "finish": tokens[-1].finish_reason,
            "text": "".join(t.text for t in tokens)}


def drive(adapter, plan, *, hook=None, limit=5000):
    """plan: [(arrival_step, prompt, max_tokens)]. Returns per-request results keyed by plan index."""
    pending = sorted(enumerate(plan), key=lambda item: item[1][0])
    results, index_of, step_no = {}, {}, 0
    while pending or adapter.sequences:
        if not adapter.sequences and pending and pending[0][1][0] > step_no:
            step_no = pending[0][1][0]
        while pending and pending[0][1][0] <= step_no:
            index, (_, prompt, max_tokens) = pending.pop(0)
            index_of[adapter.insert(prompt, max_tokens)] = index
            results[index] = {"ids": [], "finish": None, "text": "", "phases": []}
        step = adapter.step()
        for uid, token in step.events:
            entry = results[index_of[uid]]
            entry["text"] += token.text
            if token.output:
                entry["ids"].append(token.token_id)
            if token.finish_reason is not None:
                entry["finish"] = token.finish_reason
        if hook:
            hook(step_no, step)
        step_no += 1
        assert step_no < limit, "batch never drained"
    return results


def assert_matches_serial(backend, plan, results):
    produced = {token for result in results.values() for token in result["ids"]}
    assert len(produced) >= 3, "degenerate output would make this comparison vacuous"
    for index, (_, prompt, max_tokens) in enumerate(plan):
        expected = serial(backend, prompt, max_tokens)
        got = results[index]
        assert got["ids"] == expected["ids"], f"request {index}"
        assert got["finish"] == expected["finish"], f"request {index}"
        assert got["text"] == expected["text"], f"request {index}"


@pytest.fixture(params=["llama", "qwen2", "llama-nf4", "qwen2-nf4"])
def backend(request, model_dir, qwen2_dir):
    return make_backend(model_dir, request.param, qwen2_dir)


def test_batched_greedy_equals_serial_for_varied_prompt_and_output_lengths(backend):
    plan = [(0, prompt_of(length, seed), max_tokens)
            for seed, (length, max_tokens) in enumerate([(1, 6), (3, 1), (7, 12), (12, 4), (20, 9), (2, 15),
                                                         (30, 3), (5, 8)])]
    results = drive(make_adapter(backend, capacity=8), plan)
    assert_matches_serial(backend, plan, results)


def test_logits_match_the_transformers_forward_pass_at_every_step(backend):
    """Stronger than token equality: every row's logits, at every step, match a full Transformers recompute."""
    adapter = make_adapter(backend, capacity=4, prefill_step_size=5)
    captured = []
    original = adapter._forward

    def recording(*args, **kwargs):
        logits = original(*args, **kwargs).detach().clone()
        captured.append(logits)
        return logits

    adapter._forward = recording
    plan = [(prompt_of(n, n), 6) for n in (4, 9, 15, 2)]
    prompts = [backend.tokenizer.encode(prompt) for prompt, _ in plan]
    index_of = {adapter.insert(prompt, max_tokens): i for i, (prompt, max_tokens) in enumerate(plan)}
    generated = {i: [] for i in range(len(plan))}
    checks = 0
    while adapter.sequences:
        prefilling = bool(adapter.prefill_queue) and (not adapter.decoding or adapter.last_phase == "decode")
        if prefilling:
            uid = adapter.prefill_queue[0]
            index = index_of[uid]
            width = min(adapter.sequences[uid].prefilled + adapter.prefill_step_size, len(prompts[index]))
            contexts = [prompts[index][:width]]
        else:
            contexts = [prompts[index_of[uid]] + generated[index_of[uid]] for uid in adapter.decoding]
        captured.clear()
        step = adapter.step()
        assert len(captured) == 1 and captured[0].shape[0] == len(contexts)
        for row, context in enumerate(contexts):
            with torch.inference_mode():
                reference = backend.model(torch.tensor([context])).logits[0, -1]
            torch.testing.assert_close(captured[0][row, -1], reference, atol=1e-4, rtol=1e-4)
            checks += 1
        for uid, token in step.events:
            if token.output:
                generated[index_of[uid]].append(token.token_id)
    assert checks >= 20


def test_eos_stops_a_sequence_without_counting_as_output_and_frees_the_batch_for_the_rest(backend):
    """A random tiny model never emits the real EOS id, so declare a token it does produce to be EOS."""
    prompts = [prompt_of(n, n + 40) for n in (3, 8, 5, 11)]
    probe = serial(backend, prompts[0], 12)["ids"]
    eos = probe[2]  # the first prompt now stops on its third generated token
    backend.eos_token_ids = frozenset({eos})
    plan = [(0, prompt, 12) for prompt in prompts]
    seen = {}

    def watch(step_no, step):
        for uid, token in step.events:
            if token.token_id == eos:
                assert token.output is False and token.finish_reason == "stop"
                seen[uid] = seen.get(uid, 0) + 1

    results = drive(make_adapter(backend, capacity=4), plan, hook=watch)
    assert_matches_serial(backend, plan, results)
    assert results[0]["finish"] == "stop" and len(results[0]["ids"]) == 2 and eos not in results[0]["ids"]
    assert seen and all(count == 1 for count in seen.values()), "EOS must be delivered exactly once per sequence"


def test_sequences_join_a_running_batch_at_different_positions(backend):
    plan = [(0, prompt_of(6, 1), 14), (0, prompt_of(11, 2), 14), (3, prompt_of(4, 3), 10),
            (5, prompt_of(25, 4), 8), (9, prompt_of(2, 5), 12)]
    results = drive(make_adapter(backend, capacity=8, prefill_step_size=8), plan)
    assert_matches_serial(backend, plan, results)


def test_slots_are_reused_after_sequences_finish_without_leaking_old_kv(backend):
    rng = random.Random(7)
    plan = [(rng.randrange(0, 40), prompt_of(rng.randrange(1, 28), i), rng.randrange(1, 14)) for i in range(24)]
    adapter = make_adapter(backend, capacity=4, prefill_step_size=8)
    keys_before = [id(t) for t in adapter.keys]
    inserted = []

    def respect_capacity(step_no, step):
        inserted.append(len(adapter.sequences))
        assert len(adapter.sequences) <= adapter.capacity

    # Admission control belongs to the engine; model it by delaying arrivals while the pool is full.
    results = {}
    pending = sorted(enumerate(plan), key=lambda item: item[1][0])
    index_of, step_no = {}, 0
    while pending or adapter.sequences:
        while pending and pending[0][1][0] <= step_no and len(adapter.sequences) < adapter.capacity:
            index, (_, prompt, mt) = pending.pop(0)
            index_of[adapter.insert(prompt, mt)] = index
            results[index] = {"ids": [], "finish": None, "text": ""}
        if adapter.sequences:
            for uid, token in adapter.step().events:
                entry = results[index_of[uid]]
                entry["text"] += token.text
                if token.output:
                    entry["ids"].append(token.token_id)
                if token.finish_reason:
                    entry["finish"] = token.finish_reason
        step_no += 1
        assert step_no < 5000
    assert_matches_serial(backend, plan, results)
    assert not adapter.decoding and not adapter.sequences and not adapter.prefill_queue
    assert [id(t) for t in adapter.keys] == keys_before, "the KV pool must never be reallocated"
    assert adapter.counters["slot_moves"] > 0, "the test must exercise slot compaction"


def test_removing_a_running_sequence_does_not_disturb_the_others_or_leak_into_its_slot(backend):
    adapter = make_adapter(backend, capacity=4, prefill_step_size=8)
    plan = [(prompt_of(8, 1), 10), (prompt_of(5, 2), 10), (prompt_of(12, 3), 10), (prompt_of(3, 4), 10)]
    uids = [adapter.insert(prompt, mt) for prompt, mt in plan]
    got = {uid: [] for uid in uids}

    def pump(steps):
        for _ in range(steps):
            for uid, token in adapter.step().events:
                if token.output:
                    got[uid].append(token.token_id)

    pump(9)  # all four prefilled, a few decode steps in
    victim = uids[1]
    assert adapter.remove(victim) is True
    assert adapter.remove(victim) is False
    late = adapter.insert(prompt_of(6, 9), 8)  # takes the freed capacity and slot
    got[late] = []
    while adapter.sequences:
        pump(1)
    for uid, (prompt, mt) in zip(uids, plan):
        if uid != victim:
            assert got[uid] == serial(backend, prompt, mt)["ids"]
    assert got[late] == serial(backend, prompt_of(6, 9), 8)["ids"]
    assert len(got[victim]) < 10


def test_chunked_prefill_matches_serial_and_interleaves_with_decode(backend):
    adapter = make_adapter(backend, capacity=4, prefill_step_size=5)
    short = adapter.insert(prompt_of(3, 1), 30)
    phases = []
    got = {short: []}
    for _ in range(3):
        step = adapter.step()
        for uid, token in step.events:
            got[uid].append(token.token_id)
    long_prompt = prompt_of(47, 2)
    long = adapter.insert(long_prompt, 6)
    got[long] = []
    while adapter.sequences:
        step = adapter.step()
        phases.append(step.phase)
        for uid, token in step.events:
            got[uid].append(token.token_id)
    first = phases.index("prefill")
    window = phases[first:first + 2 * (47 // 5)]
    assert "decode" in window, "decode must not be starved while a long prompt prefills"
    assert phases.count("prefill") >= 47 // 5
    assert got[long] == serial(backend, long_prompt, 6)["ids"]
    assert got[short] == serial(backend, prompt_of(3, 1), 30)["ids"][:len(got[short])]


def test_prefill_alternates_with_decode_instead_of_blocking_it(backend):
    adapter = make_adapter(backend, capacity=3, prefill_step_size=4)
    adapter.insert(prompt_of(2, 1), 40)
    adapter.step()  # prefill the short one
    adapter.insert(prompt_of(40, 2), 3)
    seen = []
    for _ in range(8):
        seen.append(adapter.step().phase)
    assert seen[:6] == ["decode", "prefill", "decode", "prefill", "decode", "prefill"]


def test_prompts_that_do_not_fit_and_capacity_overflow_are_refused(backend):
    adapter = make_adapter(backend, capacity=2, row_tokens=32)
    with pytest.raises(ValueError, match="KV row"):
        adapter.insert(prompt_of(30, 1), 8)
    with pytest.raises(ValueError, match="invalid output budget"):
        adapter.insert("hello", 0)
    with pytest.raises(ValueError, match="encodes empty"):
        adapter.insert("", 4)
    adapter.insert("hello world", 2)
    adapter.insert("the a", 2)
    with pytest.raises(RuntimeError, match="capacity exhausted"):
        adapter.insert("hello", 2)


def test_first_token_finish_never_occupies_a_decode_slot(backend):
    adapter = make_adapter(backend, capacity=2)
    adapter.insert(prompt_of(5, 1), 1)
    step = adapter.step()
    assert [t.finish_reason for _, t in step.events] == ["length"]
    assert not adapter.decoding and not adapter.sequences


def test_the_adapter_is_confined_to_its_owner_thread(backend):
    import threading
    adapter = make_adapter(backend, capacity=2)
    failures = []

    def other():
        for call in (lambda: adapter.insert("hello", 2), adapter.step, adapter.close):
            try:
                call()
            except RuntimeError as error:
                failures.append(str(error))

    thread = threading.Thread(target=other)
    thread.start()
    thread.join()
    assert len(failures) == 3 and all("owner thread" in message for message in failures)


def test_close_releases_the_pool_and_blocks_further_use(backend):
    adapter = make_adapter(backend, capacity=2)
    adapter.insert("hello world", 3)
    adapter.close()
    adapter.close()
    assert adapter.keys == [] and not adapter.sequences
    with pytest.raises(RuntimeError, match="closed"):
        adapter.step()


def test_snapshot_reports_measured_pool_state(backend):
    adapter = make_adapter(backend, capacity=4)
    for n in (4, 6):
        adapter.insert(prompt_of(n, n), 5)
    for _ in range(4):
        adapter.step()
    snapshot = adapter.snapshot()
    assert snapshot["kind"] == "fixed_slot_kv_pool_not_paged"
    assert snapshot["capacity_rows"] == 4 and snapshot["decode_rows_active"] == 2
    assert snapshot["kv_bytes_filled"] > 0 and snapshot["decode_tokens"] >= 2
    assert snapshot["pool_bytes_allocated"] == 5 * adapter.row_tokens * adapter.bytes_per_token


def test_admission_is_sized_from_real_kv_bytes_not_the_flat_default(backend):
    adapter = make_adapter(backend, capacity=3)
    admission = adapter.build_admission(queue_size=5)
    assert admission.bytes_per_token == adapter.bytes_per_token
    assert admission.budget_bytes == (3 + 5) * adapter.row_tokens * adapter.bytes_per_token


# ----- model support -------------------------------------------------------------------------------

def config(**overrides):
    values = dict(model_type="qwen2", num_hidden_layers=28, num_attention_heads=28, num_key_value_heads=4,
                  hidden_size=3584, use_sliding_window=False, layer_types=["full_attention"] * 28)
    values.update(overrides)
    return SimpleNamespace(**values)


def test_geometry_matches_the_published_qwen25_7b_layout():
    geometry = read_geometry(config())
    assert (geometry.layers, geometry.q_heads, geometry.kv_heads, geometry.head_dim) == (28, 28, 4, 128)
    assert geometry.group == 7
    assert geometry.kv_bytes_per_token(2) == 57344  # 2 * 28 * 4 * 128 * 2 bytes


@pytest.mark.parametrize("override", [{"model_type": "gemma2"}, {"model_type": "qwen3"}, {"model_type": "mixtral"},
                                      {"use_sliding_window": True},
                                      {"layer_types": ["full_attention"] * 27 + ["sliding_attention"]},
                                      {"num_key_value_heads": 5}])
def test_unsupported_architectures_are_refused_not_assumed(override):
    with pytest.raises(UnsupportedModel):
        read_geometry(config(**override))


def test_the_adapter_refuses_an_unsupported_loaded_model(backend):
    backend.model.config.model_type = "gemma2"
    with pytest.raises(UnsupportedModel):
        make_adapter(backend)


# ----- VRAM planning -------------------------------------------------------------------------------

def test_pool_plan_for_7b_on_a_4090_fits_and_is_limited_by_vram():
    plan = plan_kv_pool(weights_bytes=int(15.23e9), bytes_per_token=57344, row_tokens=1664,
                        requested_capacity=128, total_vram_bytes=25252724736, memory_fraction=0.9)
    assert 1 <= plan.capacity < 128 and plan.limited_by == "vram"
    assert plan.weights_bytes + plan.pool_bytes + 1.5 * 1024**3 <= 25252724736 * 0.9


def test_pool_plan_grants_the_request_when_it_fits():
    plan = plan_kv_pool(weights_bytes=int(1e9), bytes_per_token=12288, row_tokens=1536,
                        requested_capacity=64, total_vram_bytes=25252724736, memory_fraction=0.9)
    assert plan.capacity == 64 and plan.limited_by == "requested"


def test_pool_plan_refuses_a_model_that_does_not_leave_room_for_one_slot():
    with pytest.raises(ValueError, match="no KV slot fits"):
        plan_kv_pool(weights_bytes=int(28e9), bytes_per_token=98304, row_tokens=1664,
                     requested_capacity=8, total_vram_bytes=25252724736, memory_fraction=0.9)


def test_pool_plan_keeps_one_row_for_prefill_staging():
    plan = plan_kv_pool(weights_bytes=0, bytes_per_token=1000, row_tokens=10, requested_capacity=99,
                        kv_budget_bytes=50_000)
    assert plan.capacity == 4 and plan.pool_bytes == 5 * 10_000  # 5 rows: 4 decode + 1 staging


# ----- detokenization ------------------------------------------------------------------------------

def test_incremental_text_equals_full_decode_for_the_word_tokenizer(backend):
    ids = backend.tokenizer.encode(prompt_of(25, 3))
    detok = IncrementalDetokenizer(backend.tokenizer)
    text = "".join(detok.add(token) for token in ids) + detok.flush()
    assert text == backend.tokenizer.decode(ids)


def qwen_tokenizer():
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", local_files_only=True,
                                             revision="7ae557604adf67be50417f59c2c2f167def9a775")
    except Exception:
        return None


def test_incremental_text_survives_multibyte_characters_split_across_tokens():
    tokenizer = qwen_tokenizer()
    if tokenizer is None:
        pytest.skip("Qwen tokenizer is not in the local Hugging Face cache")
    sample = "Hello 世界! 🚀 naïve café 日本語のテキスト emoji: 👩‍💻 end"
    ids = tokenizer.encode(sample)
    detok = IncrementalDetokenizer(tokenizer)
    deltas = [detok.add(token) for token in ids]
    deltas.append(detok.flush())
    assert "".join(deltas) == tokenizer.decode(ids) == sample
    assert not any("�" in delta for delta in deltas[:-1]), "an incomplete UTF-8 sequence leaked into a delta"
    rng = random.Random(3)
    noise = [rng.randrange(1000, 150000) for _ in range(200)]
    detok = IncrementalDetokenizer(tokenizer)
    streamed = "".join(detok.add(token) for token in noise) + detok.flush()
    assert streamed == tokenizer.decode(noise)
