"""GPU preflight decisions with injected probes. No GPU is required to run these."""
import asyncio

from tensorpilot.backend import Token
from tensorpilot.gpu_preflight import GIB, run_preflight

GOOD_REVISION = "a" * 40
MEASURED = {"status": "measured", "source": "nvidia-smi",
            "devices": [{"index": 0, "name": "NVIDIA L4"}]}
TORCH_OK = {"importable": True, "torch_version": "2.9.1", "cuda_runtime": "12.8", "cuda_available": True,
            "devices": [{"index": 0, "name": "NVIDIA L4", "compute_capability": "8.9",
                         "free_bytes": 22 * GIB, "total_bytes": 23 * GIB}]}


class FakeBackend:
    provenance = {"artifact_sha256": "f" * 64, "model_revision": GOOD_REVISION,
                  "gpu": {"name": "NVIDIA L4"}}

    def stream(self, prompt, max_tokens):
        for index in range(max_tokens):
            yield Token("x", index, True, 3, "length" if index == max_tokens - 1 else None)

    def memory(self):
        return {"cuda_peak_reserved_bytes": 1024}


def preflight(**overrides):
    options = {"model": "org/model", "revision": GOOD_REVISION, "min_vram_gib": 16,
               "prompt": "hello", "smoke_tokens": 4, "load_backend": FakeBackend,
               "sampler": lambda: _async(MEASURED), "torch_probe": lambda: TORCH_OK}
    options.update(overrides)
    return asyncio.run(run_preflight(**options))


async def _async(value):
    return value


def by_name(report):
    return {item["name"]: item for item in report["checks"]}


def test_ready_only_when_every_required_check_passes():
    report = preflight()
    assert report["ready"] is True
    assert all(item["status"] == "pass" for item in report["checks"])
    assert by_name(report)["model_smoke"]["output_tokens"] == 4
    assert by_name(report)["model_smoke"]["artifact_sha256"] == "f" * 64


def test_a_machine_without_nvidia_is_not_ready_and_says_why():
    report = preflight(sampler=lambda: _async({"status": "unavailable", "reason": "nvidia-smi not found",
                                               "devices": []}),
                       torch_probe=lambda: {"importable": True, "cuda_available": False,
                                            "torch_version": "2.9.1", "cuda_runtime": None, "devices": []})
    assert report["ready"] is False
    checks = by_name(report)
    assert checks["nvidia_smi_visible"]["detail"] == "nvidia-smi not found"
    assert checks["torch_cuda"]["status"] == "fail"
    assert checks["vram"]["status"] == "fail"


def test_vram_below_requirement_fails_the_gate():
    small = {**TORCH_OK, "devices": [{"index": 0, "name": "small", "compute_capability": "7.5",
                                      "free_bytes": 6 * GIB, "total_bytes": 8 * GIB}]}
    report = preflight(torch_probe=lambda: small)
    assert by_name(report)["vram"]["status"] == "fail"
    assert report["ready"] is False


def test_unpinned_revision_is_rejected_even_when_everything_else_passes():
    assert by_name(preflight(revision=None))["pinned_revision"]["status"] == "fail"
    assert by_name(preflight(revision="main"))["pinned_revision"]["status"] == "fail"
    assert preflight(revision=None)["ready"] is False


def test_smoke_failure_reports_error_type_without_message():
    class Broken(FakeBackend):
        def stream(self, prompt, max_tokens):
            raise RuntimeError("secret detail that should not be echoed")
            yield  # pragma: no cover

    smoke = by_name(preflight(load_backend=Broken))["model_smoke"]
    assert smoke["status"] == "fail"
    assert smoke["detail"] == "smoke inference failed: RuntimeError"
    assert "secret" not in str(smoke)


def test_skipping_the_smoke_inference_cannot_make_the_gate_ready():
    skipped = asyncio.run(run_preflight(
        model="org/model", revision=GOOD_REVISION, min_vram_gib=16, prompt="hi", smoke_tokens=2,
        load_backend=FakeBackend, sampler=lambda: _async(MEASURED), torch_probe=lambda: TORCH_OK,
        skip_load=True))
    assert skipped["ready"] is False
    assert by_name(skipped)["model_smoke"]["status"] == "fail"
