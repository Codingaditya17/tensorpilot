"""Both runtimes expose the serving contract that the engine and HTTP layer depend on."""
from tensorpilot.backend import BACKENDS, InferenceBackend, MLXBackend
from tensorpilot.cuda_backend import CUDABackend


def assert_contract(cls):
    for name in ("stream", "memory"):
        assert callable(getattr(cls, name)), name
    assert cls.backend_name in BACKENDS


def test_cuda_backend_satisfies_the_contract():
    assert_contract(CUDABackend)
    assert CUDABackend.backend_name == "cuda"


def test_mlx_backend_satisfies_the_contract_without_regression():
    # Class-level check only: importing MLXBackend does not import MLX.
    assert_contract(MLXBackend)
    assert MLXBackend.backend_name == "mlx"


def test_protocol_declares_the_members_the_engine_reads():
    assert {"backend_name", "provenance", "max_prompt_tokens"} <= set(InferenceBackend.__annotations__)
