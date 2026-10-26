"""Command-line guards for the CUDA backend. These run without torch or a GPU."""
import sys

import pytest

from tensorpilot import server
from tensorpilot.bench import DEFAULT_MODEL


def parse_error(monkeypatch, capsys, *args, env=None):
    monkeypatch.setattr(sys, "argv", ["tensorpilot-server", *args])
    for name in ("TENSORPILOT_BACKEND", "TENSORPILOT_MODEL", "TENSORPILOT_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    for name, value in (env or {}).items():
        monkeypatch.setenv(name, value)
    with pytest.raises(SystemExit) as failure:
        server.main()
    assert failure.value.code == 2
    return capsys.readouterr().err


def test_cuda_refuses_the_mlx_default_model(monkeypatch, capsys):
    assert "needs --model" in parse_error(monkeypatch, capsys, "--backend", "cuda")


def test_cuda_refuses_adaptive_execution_but_accepts_batch(monkeypatch, capsys):
    message = parse_error(monkeypatch, capsys, "--backend", "cuda", "--model", "org/model",
                          "--execution", "adaptive")
    assert "MLX-only" in message
    # Batch is accepted by argument validation; a bad width is the failure here, not the mode.
    message = parse_error(monkeypatch, capsys, "--backend", "cuda", "--model", "org/model",
                          "--execution", "batch", "--batch-size", "257")
    assert "1..256" in message and "cuda" in message


def test_batch_width_limits_differ_by_backend(monkeypatch, capsys):
    assert "1..8" in parse_error(monkeypatch, capsys, "--execution", "batch", "--batch-size", "9")
    assert "kv-budget-gib" in parse_error(monkeypatch, capsys, "--backend", "cuda", "--model", "org/model",
                                          "--execution", "batch", "--kv-budget-gib", "-1")


def test_cuda_refuses_mlx_memory_profiling_and_bad_fraction(monkeypatch, capsys):
    assert "MLX-only" in parse_error(monkeypatch, capsys, "--backend", "cuda", "--model", "org/model",
                                     "--profile-memory")
    assert "cuda-memory-fraction" in parse_error(monkeypatch, capsys, "--backend", "cuda",
                                                 "--model", "org/model", "--cuda-memory-fraction", "1.5")


def test_environment_selects_backend_and_model(monkeypatch, capsys):
    message = parse_error(monkeypatch, capsys, env={"TENSORPILOT_BACKEND": "cuda"})
    assert "needs --model" in message


def test_prompt_cap_is_bounded(monkeypatch, capsys):
    assert "max-prompt-tokens" in parse_error(monkeypatch, capsys, "--max-prompt-tokens", "0")


def test_non_loopback_bind_requires_an_api_key(monkeypatch, capsys):
    message = parse_error(monkeypatch, capsys, "--backend", "cuda", "--model", "org/model",
                          "--host", "0.0.0.0")
    assert "TENSORPILOT_API_KEY" in message


def test_default_model_remains_the_mlx_default():
    assert DEFAULT_MODEL.startswith("mlx-community/")
