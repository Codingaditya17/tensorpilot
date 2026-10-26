import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
SCRIPT = Path(__file__).resolve().parents[1] / "deploy" / "runpod" / "diagnose_smoke.py"


@pytest.mark.skipif(torch.cuda.is_available(), reason="needs a machine without CUDA to force a failure")
def test_diagnostic_captures_a_redacted_traceback_per_dtype(tmp_path):
    report = tmp_path / "diag.txt"
    env = {**os.environ, "HF_TOKEN": "hf_neverprintthisvalue99", "TENSORPILOT_API_KEY": "worker-key-value-1234"}
    run = subprocess.run([sys.executable, "-I", str(SCRIPT), "--model", "org/model", "--output", str(report)],
                         capture_output=True, text=True, env=env, cwd=tmp_path, timeout=120)
    assert run.returncode == 1
    text = report.read_text()
    for dtype in ("float16", "bfloat16", "float32"):
        assert f"=== dtype {dtype}" in text
    assert text.count("FAILED: RuntimeError") == 3 and "Traceback" in text
    assert "summary: float16=FAIL RuntimeError" in text
    for leaked in ("hf_neverprintthisvalue99", "worker-key-value-1234"):
        assert leaked not in text and leaked not in run.stdout and leaked not in run.stderr
    assert oct(report.stat().st_mode & 0o777) == "0o600"
