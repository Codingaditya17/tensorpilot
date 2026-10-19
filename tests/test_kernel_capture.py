from pathlib import Path
from types import SimpleNamespace

import pytest

from tensorpilot.kernel_capture import run_capture


class Metal:
    def __init__(self):
        self.started = self.stopped = False

    def start_capture(self, path):
        self.started = True
        Path(path).mkdir()

    def stop_capture(self):
        self.stopped = True


class Backend:
    def __init__(self):
        self.mx = SimpleNamespace(metal=Metal(), synchronize=lambda: None)
        self.provenance = {"artifact_sha256": "test"}

    def stream(self, prompt, max_tokens):
        return iter([SimpleNamespace(prompt_tokens=3, output=True, finish_reason="length")
                     for _ in range(max_tokens)])

    def memory(self):
        return {"mlx_active_bytes": 10, "mlx_peak_bytes_process": 20}


def test_bounded_capture_records_provenance_and_stops(tmp_path):
    backend = Backend()
    result = run_capture(backend, tmp_path / "trace.gputrace", ["hello"], 2)
    assert backend.mx.metal.started and backend.mx.metal.stopped
    assert result["requests"][0]["output_tokens"] == 2
    assert result["model"]["artifact_sha256"] == "test"
    assert result["measurement_source"].startswith("MLX Metal capture")


def test_capture_requires_gputrace_and_refuses_overwrite(tmp_path):
    with pytest.raises(ValueError):
        run_capture(Backend(), tmp_path / "trace.bin", ["hello"], 1)
    path = tmp_path / "trace.gputrace"
    path.mkdir()
    with pytest.raises(FileExistsError):
        run_capture(Backend(), path, ["hello"], 1)
