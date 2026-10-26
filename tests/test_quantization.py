"""Explicit NF4 quantization: stack requirements, accounting, the backend load path and the planner.

The load path runs for real on CPU through bitsandbytes' CPU backend. That proves Transformers, bitsandbytes and
TensorPilot cooperate and that nothing falls back silently. It does NOT prove the CUDA kernels, GPU memory use or
GPU speed; those need a GPU.
"""
from importlib.metadata import PackageNotFoundError
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
bnb = pytest.importorskip("bitsandbytes")

from tiny_model import make_tiny_model_dir  # noqa: E402
from tensorpilot import quantization as quant  # noqa: E402
from tensorpilot.cuda_backend import CUDABackend  # noqa: E402
from tensorpilot.model_plan import assess  # noqa: E402

GIB = 1024**3
RTX_4090 = 25252724736


def qwen_config(**overrides):
    values = dict(model_type="qwen2", hidden_size=5120, intermediate_size=13824, num_hidden_layers=48,
                  num_attention_heads=40, num_key_value_heads=8, vocab_size=152064, tie_word_embeddings=False,
                  max_position_embeddings=32768, use_sliding_window=False, layer_types=["full_attention"] * 48)
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    return make_tiny_model_dir(tmp_path_factory.mktemp("quant-model"))


def load(model_dir, **options):
    options = {"device": "cpu", "allow_cpu": True, "dtype": "float32", "quantization": "nf4", **options}
    return CUDABackend(str(model_dir), **options)


# ----- the stack is required, never faked ----------------------------------------------------------

def test_modes_are_explicit_and_unknown_modes_are_rejected():
    assert quant.MODES == ("none", "nf4")
    assert quant.validate_mode("none") == "none"
    for bad in ("int4", "fp4", "NF4", "", None):
        with pytest.raises(ValueError, match="quantization must be one of"):
            quant.validate_mode(bad)


def test_none_needs_no_quantization_stack(monkeypatch):
    monkeypatch.setattr(quant, "version", lambda name: (_ for _ in ()).throw(PackageNotFoundError(name)))
    quant.require_stack("none")


@pytest.mark.parametrize("missing", ["bitsandbytes", "accelerate"])
def test_a_missing_package_fails_with_an_actionable_message(monkeypatch, missing):
    real = quant.version

    def fake(name):
        if name == missing:
            raise PackageNotFoundError(name)
        return real(name)

    monkeypatch.setattr(quant, "version", fake)
    with pytest.raises(quant.QuantizationUnavailable, match=f"needs {missing}.*not installed.*fall back"):
        quant.require_stack("nf4")


def test_an_old_package_is_refused(monkeypatch):
    real = quant.version
    monkeypatch.setattr(quant, "version", lambda name: "0.40.2" if name == "bitsandbytes" else real(name))
    with pytest.raises(quant.QuantizationUnavailable, match="bitsandbytes>=0.46.1; found 0.40.2"):
        quant.require_stack("nf4")


def test_a_host_where_bitsandbytes_cannot_run_is_refused(monkeypatch):
    import transformers.integrations as integrations

    def refuse(raise_exception=False):
        raise RuntimeError("no usable backend")

    monkeypatch.setattr(integrations, "validate_bnb_backend_availability", refuse)
    with pytest.raises(quant.QuantizationUnavailable, match="cannot run on this host: RuntimeError"):
        quant.require_stack("nf4")


def test_the_quantization_recipe_is_fixed_for_comparable_evidence():
    config = quant.make_config("nf4", torch.bfloat16)
    assert config.load_in_4bit and config.bnb_4bit_quant_type == "nf4"
    assert config.bnb_4bit_use_double_quant is True and config.bnb_4bit_compute_dtype == torch.bfloat16
    assert quant.make_config("none", torch.bfloat16) is None


# ----- size estimation ----------------------------------------------------------------------------

def test_the_estimate_matches_a_real_nf4_load_of_qwen25_05b_within_one_percent():
    from transformers import AutoConfig, AutoModelForCausalLM
    from tensorpilot.artifacts import resolve_model
    try:
        path, _ = resolve_model("Qwen/Qwen2.5-0.5B-Instruct", revision="7ae557604adf67be50417f59c2c2f167def9a775",
                                offline=True)
    except Exception:
        pytest.skip("Qwen2.5-0.5B is not in the local Hugging Face cache")
    config = AutoConfig.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16, device_map={"": "cpu"}, quantization_config=quant.make_config("nf4", torch.bfloat16))
    measured = quant.resident_weight_bytes(model)
    assert abs(quant.estimate_resident_bytes(config, "nf4") - measured) / measured < 0.01
    assert measured > sum(p.numel() * p.element_size() for p in model.parameters()), \
        "scale tables must be counted; parameters() alone undercounts a quantized model"


@pytest.mark.parametrize("config, published_bytes", [
    (qwen_config(), 29.54e9),                                                       # Qwen2.5-14B-Instruct
    (qwen_config(hidden_size=3584, intermediate_size=18944, num_hidden_layers=28, num_attention_heads=28,
                 num_key_value_heads=4), 15.23e9)])                                  # Qwen2.5-7B-Instruct
def test_the_bf16_estimate_matches_the_published_checkpoint_sizes(config, published_bytes):
    estimate = quant.estimate_resident_bytes(config, "none")
    assert abs(estimate - published_bytes) / published_bytes < 0.005


def test_nf4_14b_is_far_smaller_than_bf16_but_not_a_quarter():
    bf16, nf4 = (quant.estimate_resident_bytes(qwen_config(), mode) for mode in ("none", "nf4"))
    assert nf4 < 0.4 * bf16
    assert nf4 > 0.25 * bf16, "embeddings and lm_head stay in 16 bits, so the saving is less than 4x"


# ----- the backend load path ----------------------------------------------------------------------

def test_the_backend_quantizes_and_records_what_it_did(model_dir):
    backend = load(model_dir)
    info = backend.provenance["quantization"]
    assert info["format"] == "nf4" and info["double_quantization"] is True and info["block_size"] == 64
    assert info["quantized_linear_layers"] == 2 * 7  # q, k, v, o, gate, up, down in two layers
    assert info["weights_resident_bytes"] == backend.weights_resident_bytes == quant.resident_weight_bytes(backend.model)
    assert info["library_versions"]["bitsandbytes"] == bnb.__version__
    assert backend.quantization == "nf4" and backend.compute_dtype == torch.float32
    kinds = {name: type(module).__name__ for name, module in backend.model.named_modules()
             if isinstance(module, torch.nn.Linear)}
    assert kinds["lm_head"] == "Linear", "the output head stays in the compute dtype"
    assert all(kind == "Linear4bit" for name, kind in kinds.items() if name != "lm_head")


def test_unquantized_loading_is_unchanged_and_says_so(model_dir):
    backend = load(model_dir, quantization="none")
    assert backend.provenance["quantization"] is None and backend.quantization == "none"
    assert backend.weights_resident_bytes == sum(p.numel() * p.element_size() for p in backend.model.parameters())
    assert not any(isinstance(m, bnb.nn.Linear4bit) for m in backend.model.modules())


def test_a_quantized_backend_streams_greedily_and_deterministically(model_dir):
    backend = load(model_dir)
    first = [t.token_id for t in backend.stream("hello world the a", 8) if t.output]
    second = [t.token_id for t in backend.stream("hello world the a", 8) if t.output]
    assert first == second and len(first) == 8


def test_quantization_changes_the_model_so_its_weights_are_smaller(model_dir):
    plain, quantized = load(model_dir, quantization="none"), load(model_dir)
    assert quantized.weights_resident_bytes < plain.weights_resident_bytes


def test_gpu_nf4_refuses_float32_compute_without_the_cpu_test_escape_hatch(model_dir):
    with pytest.raises(ValueError, match="float16 or bfloat16"):
        CUDABackend(str(model_dir), device="cuda", dtype="float32", quantization="nf4")


def test_requested_quantization_never_falls_back_to_bf16_when_the_stack_is_missing(model_dir, monkeypatch):
    from transformers import AutoModelForCausalLM
    real = quant.version

    def fake(name):
        if name == "bitsandbytes":
            raise PackageNotFoundError(name)
        return real(name)

    monkeypatch.setattr(quant, "version", fake)
    loaded = []
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", lambda *a, **k: loaded.append(1))
    with pytest.raises(quant.QuantizationUnavailable):
        load(model_dir)
    assert loaded == [], "a model must not be loaded in any precision when the requested mode is unavailable"


def test_an_unknown_mode_is_rejected_by_the_backend(model_dir):
    with pytest.raises(ValueError, match="quantization must be one of"):
        load(model_dir, quantization="int4")


# ----- planning: the memory claims, including where 14B does not fit ------------------------------

def plan(**options):
    return assess(qwen_config(), repo="Qwen/Qwen2.5-14B-Instruct", revision="r" * 40, license="apache-2.0",
                  weights_bytes=int(29.54e9), gpu_total_bytes=RTX_4090, **options)


def test_14b_does_not_fit_in_bf16_and_fits_in_nf4_on_a_24gb_card():
    assert plan()["verdict"] == "does_not_fit"
    nf4 = plan(quantization="nf4")
    assert nf4["verdict"] == "fits" and nf4["quantization"] == "nf4"
    assert 8.5 < nf4["weights_gib"] < 10.5 and nf4["download_gib"] == pytest.approx(27.51, abs=0.05)
    assert nf4["kv_bytes_per_token"] == 196608 and nf4["dequant_scratch_gib"] > 0.3


def test_14b_nf4_grants_fewer_slots_than_requested_and_shrinks_with_context():
    nf4 = plan(quantization="nf4", requested_capacity=128)
    assert nf4["rows_granted"] < nf4["rows_requested"] and nf4["pool_limited_by"] == "vram"
    by_context = {int(k): v["max_concurrent_sequences"] for k, v in nf4["max_concurrent_sequences_by_context"].items()}
    assert by_context[512] > by_context[1024] > by_context[2048] > by_context[4096] > by_context[8192] >= 1
    assert not any(v["holds_128_concurrent"] for v in nf4["max_concurrent_sequences_by_context"].values()), \
        "14B NF4 cannot hold 128 concurrent sequences at any tested context on a 24 GB card"


def test_the_32k_context_row_alone_exceeds_what_14b_nf4_can_hold_beside_its_weights():
    nf4 = plan(quantization="nf4", prompt_tokens=32000, output_tokens=512)
    assert nf4["verdict"] == "does_not_fit" and "no KV slot fits" in nf4["reason"]


def test_quantization_does_not_change_the_kv_cache():
    bf16 = assess(qwen_config(), repo="m", revision="r" * 40, license="x", weights_bytes=int(15e9),
                  gpu_total_bytes=RTX_4090, quantization="none")
    nf4 = assess(qwen_config(), repo="m", revision="r" * 40, license="x", weights_bytes=int(15e9),
                 gpu_total_bytes=RTX_4090, quantization="nf4")
    assert bf16["kv_bytes_per_token"] == nf4["kv_bytes_per_token"] == 196608


def test_preflight_pins_match_the_hashed_cuda_lock_so_they_cannot_drift():
    import re
    from pathlib import Path
    from tensorpilot.gpu_preflight import EXPECTED_PINS, EXPECTED_QUANT_PINS
    lock = Path(__file__).resolve().parents[1] / "requirements-cuda.txt"
    locked = dict(re.findall(r"^([A-Za-z0-9_.\-]+)==([^\s\\]+)", lock.read_text(), re.M))
    for name, expected in {**EXPECTED_PINS, **EXPECTED_QUANT_PINS}.items():
        assert locked.get(name) == expected, f"{name}: lock has {locked.get(name)}, preflight expects {expected}"
    assert quant.BITSANDBYTES_MIN <= EXPECTED_QUANT_PINS["bitsandbytes"]
