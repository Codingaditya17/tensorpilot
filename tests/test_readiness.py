"""The container healthcheck must fail closed when the worker is not ready."""
import socket
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_healthcheck_fails_when_nothing_listens(tmp_path):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    result = subprocess.run([sys.executable, "-m", "tensorpilot.readiness"], cwd=REPO,
                            env={"TENSORPILOT_PORT": str(port), "PATH": "/usr/bin:/bin"},
                            capture_output=True, timeout=30)
    assert result.returncode == 1
