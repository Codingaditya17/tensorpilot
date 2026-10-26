"""GPU preflight decisions with injected probes. No GPU, Hub access or network is used."""
import asyncio
from types import SimpleNamespace

from tensorpilot.backend import Token
from tensorpilot.gpu_preflight import GIB, driver_check, pins_check, remote_revision_check, run_preflight

GOOD_REVISION = "a" * 40
PINS = {"torch": "2.9.1", "transformers": "5.20.1"}


def measured(driver="570.26"):
    return {"status": "measured", "source": "nvidia-smi",
            "devices": [{"index": 0, "name": "NVIDIA L4", "driver_version": driver}]}


TORCH_OK = {"importable": True, "torch_version": "2.9.1+cu128", "cuda_runtime": "12.8", "cuda_available": True,
            "devices": [{"index": 0, "name": "NVIDIA L4", "compute_capability": "8.9",
                         "free_bytes": 22 * GIB, "total_bytes": 23 * GIB}]}
HUB = SimpleNamespace(sha=GOOD_REVISION,
                      siblings=[SimpleNamespace(rfilename="config.json", size=1),
                                SimpleNamespace(rfilename="model.safetensors", size=1 * GIB)])


class FakeBackend:
    provenance = {"artifact_sha256": "f" * 64, "model_revision": GOOD_REVISION, "gpu": {"name": "NVIDIA L4"}}

    def stream(self, prompt, max_tokens):
        for index in range(max_tokens):
            yield Token("x", index, True, 3, "length" if index == max_tokens - 1 else None)

    def memory(self):
        return {"cuda_peak_reserved_bytes": 1024}


async def _async(value):
    return value


def preflight(**overrides):
    options = {"model": "org/model", "revision": GOOD_REVISION, "min_vram_gib": 8,
               "prompt": "hello", "smoke_tokens": 4, "load_backend": FakeBackend,
               "sampler": lambda: _async(measured()), "torch_probe": lambda: TORCH_OK,
               "fetch_info": lambda model, revision: HUB, "installed_pins": PINS}
    options.update(overrides)
    return asyncio.run(run_preflight(**options))


def by_name(report):
    return {item["name"]: item for item in report["checks"]}


def test_ready_when_every_required_check_passes(tmp_path):
    report = preflight(cache_dir=str(tmp_path))
    assert report["ready"] is True
    assert report["warnings"] == []
    checks = by_name(report)
    assert checks["remote_revision"]["weight_bytes"] == 1 * GIB
    assert checks["model_smoke"]["output_tokens"] == 4
    assert checks["model_smoke"]["artifact_sha256"] == "f" * 64


def test_driver_policy_follows_the_cuda_12_x_compatibility_table():
    assert driver_check(measured("500.10"))["status"] == "fail"
    assert driver_check(measured("525.60.13"))["status"] == "warn"
    assert driver_check(measured("535.104.05"))["status"] == "warn"
    assert driver_check(measured("570.25"))["status"] == "warn"
    assert driver_check(measured("570.26"))["status"] == "pass"
    assert driver_check(measured("580.10"))["status"] == "pass"
    assert driver_check({"devices": []})["status"] == "fail"
    assert driver_check(measured("unknown"))["status"] == "fail"


def test_a_warning_driver_is_not_ready_only_because_of_the_warning(tmp_path):
    report = preflight(cache_dir=str(tmp_path), sampler=lambda: _async(measured("535.104.05")))
    assert report["ready"] is True
    assert report["warnings"] == ["driver"]


def test_a_driver_below_525_blocks_the_gate(tmp_path):
    report = preflight(cache_dir=str(tmp_path), sampler=lambda: _async(measured("510.47")))
    assert report["ready"] is False
    assert by_name(report)["driver"]["status"] == "fail"


def test_a_machine_without_nvidia_is_not_ready_and_says_why(tmp_path):
    report = preflight(cache_dir=str(tmp_path),
                       sampler=lambda: _async({"status": "unavailable", "reason": "nvidia-smi not found",
                                               "devices": []}),
                       torch_probe=lambda: {"importable": True, "cuda_available": False, "torch_version": "2.9.1",
                                            "cuda_runtime": None, "devices": []})
    assert report["ready"] is False
    checks = by_name(report)
    assert checks["nvidia_smi_visible"]["detail"] == "nvidia-smi not found"
    assert checks["torch_cuda"]["status"] == "fail"
    assert checks["vram"]["status"] == "fail"


def test_model_fit_uses_weights_plus_headroom_not_only_the_floor(tmp_path):
    small = {**TORCH_OK, "devices": [{"index": 0, "name": "small", "compute_capability": "7.5",
                                      "free_bytes": 2 * GIB, "total_bytes": 4 * GIB}]}
    report = preflight(cache_dir=str(tmp_path), torch_probe=lambda: small, min_vram_gib=1)
    assert by_name(report)["vram"]["status"] == "fail"


def test_unpinned_revision_is_rejected_even_when_everything_else_passes(tmp_path):
    assert by_name(preflight(revision=None, cache_dir=str(tmp_path)))["pinned_revision"]["status"] == "fail"
    assert by_name(preflight(revision="main", cache_dir=str(tmp_path)))["pinned_revision"]["status"] == "fail"


def test_upstream_sha_mismatch_and_unreachable_hub_are_failures():
    other = SimpleNamespace(sha="b" * 40, siblings=HUB.siblings)
    result, weights = remote_revision_check("org/model", GOOD_REVISION, fetch_info=lambda m, r: other)
    assert result["status"] == "fail" and weights is None

    def unreachable(model, revision):
        raise OSError("no route")
    result, weights = remote_revision_check("org/model", GOOD_REVISION, fetch_info=unreachable)
    assert result["status"] == "fail" and "OSError" in result["detail"]

    empty = SimpleNamespace(sha=GOOD_REVISION, siblings=[])
    assert remote_revision_check("org/model", GOOD_REVISION, fetch_info=lambda m, r: empty)[0]["status"] == "fail"


def test_skipping_remote_verification_cannot_be_ready(tmp_path):
    report = preflight(cache_dir=str(tmp_path), skip_remote=True)
    assert report["ready"] is False
    assert by_name(report)["remote_revision"]["status"] == "fail"


def test_storage_must_hold_the_weights_twice_plus_margin(tmp_path):
    from tensorpilot.gpu_preflight import storage_check
    assert storage_check(str(tmp_path), 1024)["status"] == "pass"
    assert storage_check(str(tmp_path), 10**15)["status"] == "fail"


def test_pins_detect_drift_from_the_lock():
    assert pins_check(PINS)["status"] == "pass"
    drifted = pins_check({"torch": "2.8.0", "transformers": "5.20.1"})
    assert drifted["status"] == "fail"
    assert "torch" in drifted["detail"] or "torch" in str(drifted)


def test_smoke_failure_reports_error_type_without_message(tmp_path):
    class Broken(FakeBackend):
        def stream(self, prompt, max_tokens):
            raise RuntimeError("secret detail that should not be echoed")
            yield  # pragma: no cover

    smoke = by_name(preflight(load_backend=Broken, cache_dir=str(tmp_path)))["model_smoke"]
    assert smoke["status"] == "fail"
    assert smoke["detail"] == "smoke inference failed: RuntimeError"
    assert "secret" not in str(smoke)


def test_traceback_file_is_private_redacted_and_keeps_the_report_message_free(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_supersecrettokenvalue123")
    monkeypatch.setenv("TENSORPILOT_API_KEY", "worker-key-value-1234")

    class Broken(FakeBackend):
        def stream(self, prompt, max_tokens):
            raise ValueError("bad shape; token hf_supersecrettokenvalue123 key worker-key-value-1234 "
                             "header Bearer abcdefghijklmnopqrstuvwx and a bearer token phrase")
            yield  # pragma: no cover

    target = tmp_path / "preflight.traceback.txt"
    smoke = by_name(preflight(load_backend=Broken, cache_dir=str(tmp_path), traceback_file=str(target)))["model_smoke"]
    assert smoke["detail"] == "smoke inference failed: ValueError"
    assert smoke["stage"] == "generate" and smoke["traceback_file"] == str(target)
    assert "bad shape" not in str(smoke)
    text = target.read_text()
    assert "ValueError: bad shape" in text and "stream" in text and "during generate" in text
    for leaked in ("hf_supersecrettokenvalue123", "worker-key-value-1234", "abcdefghijklmnopqrstuvwx"):
        assert leaked not in text
    assert "bearer token phrase" in text   # ordinary prose is not mistaken for a credential
    assert oct(target.stat().st_mode & 0o777) == "0o600"


def test_traceback_reports_the_load_stage_and_survives_an_unwritable_path(tmp_path):
    def broken_load():
        raise ValueError("cannot load")

    target = tmp_path / "t.txt"
    smoke = by_name(preflight(load_backend=broken_load, cache_dir=str(tmp_path), traceback_file=str(target)))["model_smoke"]
    assert smoke["stage"] == "load" and "during load" in target.read_text()
    bad = by_name(preflight(load_backend=broken_load, cache_dir=str(tmp_path),
                            traceback_file=str(tmp_path / "missing" / "t.txt")))["model_smoke"]
    assert smoke["status"] == bad["status"] == "fail"
    assert bad["traceback_file_error"] == "FileNotFoundError" and "traceback_file" not in bad


def test_no_traceback_file_is_written_unless_requested(tmp_path):
    class Broken(FakeBackend):
        def stream(self, prompt, max_tokens):
            raise ValueError("x")
            yield  # pragma: no cover

    preflight(load_backend=Broken, cache_dir=str(tmp_path))
    assert list(tmp_path.glob("*traceback*")) == []


def test_skipping_the_smoke_inference_cannot_make_the_gate_ready(tmp_path):
    assert preflight(cache_dir=str(tmp_path), skip_load=True)["ready"] is False


# --- quantized loads: the download is bf16 but only a fraction stays on the GPU --------------------------------

QUANT_PINS = {**PINS, "bitsandbytes": "0.50.2", "accelerate": "1.15.0"}
HUB_14B = SimpleNamespace(sha=GOOD_REVISION, siblings=[SimpleNamespace(rfilename="model.safetensors", size=int(29.54e9))])


def qwen14b_config(model, revision):
    return SimpleNamespace(model_type="qwen2", hidden_size=5120, intermediate_size=13824, num_hidden_layers=48,
                           num_attention_heads=40, num_key_value_heads=8, vocab_size=152064,
                           tie_word_embeddings=False)


def preflight_14b(tmp_path, **overrides):
    options = {"fetch_info": lambda model, revision: HUB_14B, "cache_dir": str(tmp_path),
               "installed_pins": QUANT_PINS, "config_loader": qwen14b_config,
               "torch_probe": lambda: {**TORCH_OK, "devices": [{**TORCH_OK["devices"][0],
                                                                   "free_bytes": int(23.1 * GIB),
                                                                   "total_bytes": int(23.5 * GIB)}]}}
    options.update(overrides)
    return preflight(**options)


def test_14b_bf16_is_rejected_on_vram_but_nf4_is_judged_on_its_quantized_size(tmp_path):
    bf16 = by_name(preflight_14b(tmp_path))
    assert bf16["vram"]["status"] == "fail" and "need 29" in bf16["vram"]["detail"]
    nf4 = by_name(preflight_14b(tmp_path, quantization="nf4"))
    assert nf4["vram"]["status"] == "pass"
    assert "estimated resident weights" in nf4["vram"]["detail"] and "download is 27.5 GiB" in nf4["vram"]["detail"]
    assert 8.5 * GIB < nf4["vram"]["resident_bytes_estimate"] < 10.5 * GIB
    assert nf4["quantization_stack"]["status"] == "pass"


def test_quantized_runs_pin_bitsandbytes_and_accelerate_but_plain_runs_do_not(tmp_path, monkeypatch):
    import importlib.metadata
    real = importlib.metadata.version

    def hermetic(name):  # this test must not depend on what the developer's environment has installed
        if name in ("bitsandbytes", "accelerate"):
            raise importlib.metadata.PackageNotFoundError(name)
        return real(name)

    monkeypatch.setattr(importlib.metadata, "version", hermetic)
    missing = pins_check(PINS, "nf4")
    assert missing["status"] == "fail" and "bitsandbytes" in missing["detail"]
    assert pins_check(PINS, "none")["status"] == "pass"
    drifted = pins_check({**QUANT_PINS, "bitsandbytes": "0.49.2"}, "nf4")
    assert drifted["status"] == "fail" and "0.49.2" in drifted["detail"]
    assert pins_check(QUANT_PINS, "nf4")["status"] == "pass"


def test_an_unusable_quantization_stack_fails_the_gate_instead_of_degrading(tmp_path, monkeypatch):
    from tensorpilot import quantization

    def refuse(mode):
        raise quantization.QuantizationUnavailable("bitsandbytes is not installed")

    monkeypatch.setattr(quantization, "require_stack", refuse)
    report = preflight_14b(tmp_path, quantization="nf4")
    checks = by_name(report)
    assert checks["quantization_stack"]["status"] == "fail" and "not installed" in checks["quantization_stack"]["detail"]
    assert report["ready"] is False


def test_failing_to_size_a_quantized_model_fails_the_gate_rather_than_guessing(tmp_path):
    def broken(model, revision):
        raise OSError("hub unreachable")

    checks = by_name(preflight_14b(tmp_path, quantization="nf4", config_loader=broken))
    assert checks["quantized_size"]["status"] == "fail" and "OSError" in checks["quantized_size"]["detail"]


def test_storage_is_still_judged_on_the_download_not_the_quantized_size(tmp_path):
    from tensorpilot.gpu_preflight import STORAGE_FACTOR, STORAGE_MARGIN_BYTES
    checks = by_name(preflight_14b(tmp_path, quantization="nf4"))
    assert checks["storage"]["required_bytes"] == int(29.54e9) * STORAGE_FACTOR + STORAGE_MARGIN_BYTES
