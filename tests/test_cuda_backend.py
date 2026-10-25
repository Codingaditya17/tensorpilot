"""CPU-only tests for the CUDA backend interface. They do not measure GPU performance."""
import asyncio

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from tiny_model import EOS_ID, make_tiny_model_dir  # noqa: E402
from tensorpilot.cuda_backend import CUDABackend  # noqa: E402
from tensorpilot.engine import Engine  # noqa: E402

PROMPT = "hello world the"


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    return make_tiny_model_dir(tmp_path_factory.mktemp("tiny-model"))


def make_backend(model_dir, **options):
    options = {"device": "cpu", "allow_cpu": True, "dtype": "float32", **options}
    return CUDABackend(str(model_dir), **options)


def output_ids(tokens):
    return [token.token_id for token in tokens if token.output]


def test_greedy_stream_matches_transformers_generate(model_dir):
    backend = make_backend(model_dir)
    tokens = list(backend.stream(PROMPT, 6))
    ids = backend.tokenizer.encode(PROMPT)
    with torch.inference_mode():
        reference = backend.model.generate(torch.tensor([ids]), attention_mask=torch.ones(1, len(ids), dtype=torch.long),
                                           max_new_tokens=6, do_sample=False, pad_token_id=EOS_ID)
    expected = []
    for token_id in reference[0].tolist()[len(ids):]:
        expected.append(token_id)
        if token_id in backend.eos_token_ids:
            break
    assert [token.token_id for token in tokens] == expected


def test_length_limit_counts_every_output_token(model_dir):
    backend = make_backend(model_dir)
    tokens = list(backend.stream(PROMPT, 3))
    assert all(token.prompt_tokens == len(backend.tokenizer.encode(PROMPT)) for token in tokens)
    assert [token.finish_reason for token in tokens] == [None, None, "length"]
    assert all(token.output for token in tokens)


def test_eos_is_yielded_but_never_counted_as_output(model_dir):
    backend = make_backend(model_dir)
    baseline = list(backend.stream(PROMPT, 4))
    # Force the third generated token to be end-of-sequence to test accounting deterministically.
    backend.eos_token_ids = frozenset({baseline[2].token_id})
    tokens = list(backend.stream(PROMPT, 4))
    assert [token.output for token in tokens] == [True, True, False]
    assert tokens[-1].finish_reason == "stop"
    assert tokens[-1].text == ""
    assert output_ids(tokens) == [baseline[0].token_id, baseline[1].token_id]


def test_text_deltas_reassemble_to_the_decoded_output(model_dir):
    backend = make_backend(model_dir)
    tokens = list(backend.stream(PROMPT, 8))
    generated = output_ids(tokens)
    assert "".join(token.text for token in tokens) == backend.tokenizer.decode(generated)


def test_prompt_limits_are_enforced_before_inference(model_dir):
    backend = make_backend(model_dir, max_prompt_tokens=2)
    with pytest.raises(ValueError, match="prompt exceeds"):
        next(backend.stream(PROMPT, 1))
    with pytest.raises(ValueError, match="prompt exceeds"):
        next(backend.stream("", 1))
    assert backend.oom_events == 0


def test_closing_a_partial_stream_leaves_backend_reusable(model_dir):
    backend = make_backend(model_dir)
    stream = backend.stream(PROMPT, 16)
    next(stream)
    next(stream)
    stream.close()
    assert len(list(backend.stream(PROMPT, 2))) == 2


def test_out_of_memory_is_bounded_counted_and_recoverable(model_dir):
    backend = make_backend(model_dir)
    real_model = backend.model

    class FailingOnSecondCall:
        calls = 0

        def __call__(self, **kwargs):
            FailingOnSecondCall.calls += 1
            if FailingOnSecondCall.calls == 2:
                raise torch.OutOfMemoryError("injected out of memory")
            return real_model(**kwargs)

    backend.model = FailingOnSecondCall()
    with pytest.raises(torch.OutOfMemoryError):
        list(backend.stream(PROMPT, 5))
    assert backend.oom_events == 1
    assert backend.memory()["cuda_oom_events"] == 1
    backend.model = real_model
    assert len(list(backend.stream(PROMPT, 2))) == 2


def test_cpu_memory_fields_are_unmeasured_not_zero(model_dir):
    backend = make_backend(model_dir)
    sample = backend.memory()
    assert sample["cuda_allocated_bytes"] is None
    assert sample["cuda_peak_reserved_bytes"] is None
    assert sample["cuda_oom_events"] == 0


def test_provenance_identifies_runtime_without_claiming_a_gpu(model_dir):
    backend = make_backend(model_dir)
    assert backend.backend_name == "cuda"
    assert backend.provenance["backend"] == "cuda"
    assert backend.provenance["device"] == "cpu"
    assert backend.provenance["gpu"] is None
    assert backend.provenance["torch_version"] == torch.__version__
    assert backend.provenance["artifact_sha256"]


def test_device_and_configuration_rules_fail_before_loading(model_dir):
    with pytest.raises(ValueError, match="CPU is accepted only"):
        CUDABackend(str(model_dir), device="cpu")
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="cannot see a CUDA device"):
            CUDABackend(str(model_dir), device="cuda")
    with pytest.raises(ValueError, match="dtype"):
        make_backend(model_dir, dtype="int8")
    with pytest.raises(ValueError, match="memory_fraction"):
        make_backend(model_dir, memory_fraction=0)
    with pytest.raises(ValueError, match="max_prompt_tokens"):
        make_backend(model_dir, max_prompt_tokens=0)


async def _terminal(job):
    while True:
        event = await asyncio.wait_for(job.events.get(), 30)
        if event["type"] in ("done", "error"):
            return event


def test_engine_cancels_real_decode_releases_reservation_and_recovers(model_dir):
    async def scenario():
        engine = Engine(lambda: make_backend(model_dir), queue_size=2)
        await engine.start(warmup=True)
        try:
            job = engine.submit(PROMPT, 64, 30)
            first = await asyncio.wait_for(job.events.get(), 30)
            assert first["type"] == "token"
            engine.cancel(job)
            terminal = await _terminal(job)
            assert terminal["type"] == "error" and terminal["code"] == "cancelled"
            assert engine.admission.reserved_bytes == 0
            recovered = await _terminal(engine.submit(PROMPT, 3, 30))
            assert recovered["type"] == "done"
            assert recovered["generated_tokens"] <= 3
            assert recovered["estimated_kv_bytes"] > 0
            assert "cuda_peak_reserved_bytes" in recovered
        finally:
            await engine.close()
    asyncio.run(scenario())


def test_prompt_plus_output_beyond_context_is_rejected_before_inference(model_dir):
    backend = make_backend(model_dir)
    assert backend.context_tokens == 256
    with pytest.raises(ValueError, match="context window"):
        next(backend.stream(PROMPT, 256))
    assert backend.oom_events == 0
    assert len(list(backend.stream(PROMPT, 4))) == 4


def test_closing_a_stream_releases_cache_references(model_dir):
    import gc
    import weakref

    backend = make_backend(model_dir)
    refs = []
    real_forward = backend.model.forward

    def recording(**kwargs):
        output = real_forward(**kwargs)
        refs.append(weakref.ref(output.past_key_values))
        return output

    backend.model.forward = recording
    stream = backend.stream(PROMPT, 8)
    next(stream)
    next(stream)
    stream.close()
    gc.collect()
    assert refs and all(ref() is None for ref in refs)
    del backend.model.forward
