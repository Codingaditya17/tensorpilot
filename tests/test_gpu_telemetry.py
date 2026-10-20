"""Telemetry parsing and sampling. Fixtures stand in for nvidia-smi; nothing here touches a GPU."""
import asyncio
import stat

import pytest

from tensorpilot.gpu_telemetry import GPUTelemetry, parse_nvidia_smi, sample_gpus

ROW = ("0, NVIDIA L4, 550.54.15, 8.9, 23034, 812, 37, 52, 41.20, [N/A]")
ROW_WITH_ECC = ("1, NVIDIA A100-SXM4-80GB, 550.54.15, 8.0, 81920, 2048, 99, 70, 380.5, 0")


def write_fake_smi(path, *, body, exit_code=0, sleep_s=0):
    script = f"#!/bin/sh\nsleep {sleep_s}\ncat <<'EOF_SMI'\n{body}\nEOF_SMI\nexit {exit_code}\n"
    path.write_text(script)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


def test_parser_maps_fields_and_keeps_unsupported_values_null():
    devices = parse_nvidia_smi(ROW + "\n" + ROW_WITH_ECC + "\n")
    assert devices[0] == {
        "index": 0, "name": "NVIDIA L4", "driver_version": "550.54.15", "compute_capability": "8.9",
        "memory_total_mib": 23034.0, "memory_used_mib": 812.0, "utilization_gpu_percent": 37.0,
        "temperature_c": 52.0, "power_draw_w": 41.2, "ecc_uncorrected_volatile": None}
    assert devices[1]["ecc_uncorrected_volatile"] == 0
    assert devices[1]["index"] == 1


def test_parser_rejects_unexpected_shape():
    with pytest.raises(ValueError, match="shape"):
        parse_nvidia_smi("0, only three, fields")


def test_missing_binary_is_reported_as_unavailable_not_as_zero():
    result = asyncio.run(sample_gpus(executable="tensorpilot-no-such-nvidia-smi"))
    assert result["status"] == "unavailable"
    assert result["devices"] == []


def test_fake_binary_produces_a_measured_sample(tmp_path):
    binary = write_fake_smi(tmp_path / "nvidia-smi", body=ROW)
    result = asyncio.run(sample_gpus(executable=str(binary)))
    assert result["status"] == "measured"
    assert result["source"] == "nvidia-smi"
    assert result["devices"][0]["name"] == "NVIDIA L4"


def test_nonzero_exit_and_timeout_are_errors(tmp_path):
    failing = write_fake_smi(tmp_path / "failing", body="", exit_code=9)
    assert asyncio.run(sample_gpus(executable=str(failing)))["status"] == "error"
    slow = write_fake_smi(tmp_path / "slow", body=ROW, sleep_s=5)
    result = asyncio.run(sample_gpus(executable=str(slow), timeout_s=.2))
    assert result["status"] == "error" and "timed out" in result["reason"]


def test_sampler_faults_do_not_stop_later_samples():
    calls = []

    async def flaky():
        calls.append(None)
        if len(calls) == 1:
            raise RuntimeError("driver hiccup")
        return {"status": "measured", "devices": []}

    async def scenario():
        telemetry = GPUTelemetry(interval_s=.01, sampler=flaky)
        await telemetry.start()
        for _ in range(200):
            await asyncio.sleep(.005)
            if telemetry.latest["status"] == "measured":
                break
        await telemetry.close()
        return telemetry

    telemetry = asyncio.run(scenario())
    assert telemetry.latest["status"] == "measured"
    assert telemetry.snapshot()["sample_age_s"] is not None
    assert len(calls) >= 2


def test_telemetry_rejects_nonpositive_interval():
    with pytest.raises(ValueError):
        GPUTelemetry(0)
