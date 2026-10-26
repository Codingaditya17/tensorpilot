"""Test-wide fixtures."""
import pytest


@pytest.fixture(autouse=True)
def bitsandbytes_generic_cpu_path(monkeypatch):
    """Pin bitsandbytes' CPU code path so NF4 tests do not depend on the host CPU.

    On CPUs with AVX512-BF16, ``Linear4bit`` converts its weights to a packed CPU layout that cannot represent a layer
    narrower than the 64-element NF4 block, and the tiny test models are narrower (32 input features). GitHub's runners
    differ in CPU, so the same job passed on one and failed on another. Real models are far wider than a block, and the CUDA
    path never takes this branch, so nothing about a real run is hidden by pinning the generic path here.
    """
    try:
        import bitsandbytes.nn.modules as modules
    except ImportError:
        return
    monkeypatch.setattr(modules, "has_avx512bf16", lambda: False, raising=False)
