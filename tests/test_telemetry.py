"""Telemetry collector against a stand-in nvidia-smi and a fake worker /health endpoint."""
import asyncio
import json
import stat
import sys

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from tensorpilot import telemetry


def fake_smi(tmp_path, rows):
    """An executable that prints the next scripted row on each call and repeats the last one after that."""
    counter = tmp_path / "calls"
    counter.write_text("0")
    script = tmp_path / "nvidia-smi"
    script.write_text(f"#!{sys.executable}\nimport json, pathlib\nrows = {rows!r}\n"
                      f"path = pathlib.Path({str(counter)!r})\nn = int(path.read_text())\npath.write_text(str(n + 1))\n"
                      "print(rows[min(n, len(rows) - 1)])\n")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return str(script)


ROWS = ["0, 40, 20, 3000, 24564, 120.5, 50, 2520, 10501, 0x0000000000000001",
        "0, 90, 60, 20000, 24564, 400.0, 70, 2100, 10501, 0x0000000000000004",
        "0, 100, 70, 21000, 24564, 420.0, 76, 1900, 10501, 0x0000000000000024"]


def health_app(calls, *, key=None):
    app = web.Application()

    async def health(request):
        if key and request.headers.get("Authorization") != f"Bearer {key}":
            return web.json_response({"error": "no"}, status=401)
        calls.append(1)
        n = len(calls)
        return web.json_response({
            "ready": True, "queue_depth": n % 4, "active_sequences": 8 + n, "execution_concurrency": 32,
            "kv_pool": {"decode_rows_active": 8 + n, "prefill_waiting": 0, "kv_bytes_filled": 1000 * n,
                        "kv_fill_ratio_of_active_rows": 0.25, "pool_bytes_allocated": 10**9,
                        "decode_steps": 10 * n, "decode_tokens": 100 * n, "decode_execution_s": 0.5 * n,
                        "prefill_chunks": n, "prefill_tokens": 50 * n, "prefill_execution_s": 0.1 * n},
            "last_measured_memory": {"cuda_allocated_bytes": 5 * 10**9, "cuda_reserved_bytes": 6 * 10**9 + n,
                                     "cuda_peak_reserved_bytes": 7 * 10**9, "cuda_oom_events": 0 if n < 3 else 2},
            "admission": {"reservations": n % 3}})

    app.router.add_get("/health", health)
    return app


def test_the_csv_parser_keeps_unknown_fields_as_null_not_zero():
    device = telemetry.parse_gpu_csv("0, [N/A], 5, 100, 200, [Not Supported], 40, 2000, 9000, 0x0")[0]
    assert device["util_gpu_pct"] is None and device["power_w"] is None and device["util_mem_pct"] == 5.0
    with pytest.raises(ValueError, match="shape"):
        telemetry.parse_gpu_csv("0, 1, 2")


def test_throttle_reasons_are_decoded_and_idle_is_not_a_throttle():
    assert telemetry.decode_throttle("0x0000000000000001") == []
    assert telemetry.decode_throttle("0x4") == ["sw_power_cap"]
    assert telemetry.decode_throttle("0x0000000000000024") == ["sw_power_cap", "sw_thermal_slowdown"]
    assert telemetry.decode_throttle("garbage") is None and telemetry.decode_throttle(None) is None


def test_collector_samples_gpu_and_server_continuously_and_summarizes_a_window(tmp_path):
    async def scenario():
        calls = []
        async with TestServer(health_app(calls, key="k")) as server:
            collector = telemetry.Collector(tmp_path / "t", url=str(server.make_url("")), api_key="k", interval_s=0.05,
                                            executable=fake_smi(tmp_path, ROWS))
            await collector.start()
            collector.mark("window_start")
            await asyncio.sleep(0.6)
            collector.mark("window_end")
            overhead = await collector.stop()
        return collector, overhead

    collector, overhead = asyncio.run(scenario())
    assert len(collector.gpu_samples) >= 5 and len(collector.server_samples) >= 5
    start = next(e for e in collector.events if e["event"] == "window_start")["t"]
    end = next(e for e in collector.events if e["event"] == "window_end")["t"]
    window = collector.window(start, end)
    gpu = window["gpu"]["0"]
    assert gpu["utilization_gpu_pct"]["max"] == 100.0 and gpu["utilization_gpu_pct"]["min"] >= 40.0
    assert gpu["power_w"]["max"] == 420.0 and gpu["temperature_c"]["max"] == 76.0
    assert gpu["throttle_reasons_seen"] == ["sw_power_cap", "sw_thermal_slowdown"]
    # decode counters advance 100 tokens per 0.5 s of decode time: 200 tokens per decode-second.
    assert window["active_decode_tokens_per_s"] == pytest.approx(200.0)
    assert window["cuda_oom_events_delta"] == 2, "an OOM during the window must be visible"
    assert window["active_sequences"]["max"] > window["active_sequences"]["min"]
    assert overhead["gpu_polls"] >= 5 and overhead["server_polls"] >= 5 and overhead["server_poll_errors"] == 0
    assert overhead["gpu_mean_call_s"] > 0 and "telemetry-off run" in overhead["note"]


def test_every_record_is_timestamped_and_files_are_valid_jsonl(tmp_path):
    async def scenario():
        async with TestServer(health_app([])) as server:
            async with telemetry.Collector(tmp_path / "t", url=str(server.make_url("")), interval_s=0.05,
                                           executable=fake_smi(tmp_path, ROWS)) as collector:
                collector.mark("phase", name="warmup")
                await asyncio.sleep(0.25)

    asyncio.run(scenario())
    for name in ("gpu", "server", "events"):
        lines = [json.loads(line) for line in (tmp_path / "t" / f"{name}.jsonl").read_text().splitlines()]
        assert lines and all("t" in r and "utc" in r for r in lines)
        times = [r["t"] for r in lines]
        assert times == sorted(times)
    events = [json.loads(line)["event"] for line in (tmp_path / "t" / "events.jsonl").read_text().splitlines()]
    assert events[0] == "collector_started" and "phase" in events and events[-1] == "collector_stopping"


def test_a_host_without_nvidia_smi_still_collects_server_telemetry(tmp_path):
    async def scenario():
        async with TestServer(health_app([])) as server:
            collector = telemetry.Collector(tmp_path / "t", url=str(server.make_url("")), interval_s=0.05,
                                            executable="definitely-not-nvidia-smi")
            await collector.start()
            await asyncio.sleep(0.25)
            await collector.stop()
        return collector

    collector = asyncio.run(scenario())
    assert collector.gpu_status.startswith("unavailable") and not collector.gpu_samples
    assert len(collector.server_samples) >= 2
    assert collector.window(0, 10)["gpu"] == {}


def test_an_unreachable_or_unauthorized_server_is_recorded_as_errors_not_a_crash(tmp_path):
    async def scenario():
        async with TestServer(health_app([], key="right")) as server:
            collector = telemetry.Collector(tmp_path / "t", url=str(server.make_url("")), api_key="wrong",
                                            interval_s=0.05, gpu=False)
            await collector.start()
            await asyncio.sleep(0.25)
            return collector, await collector.stop()

    collector, overhead = asyncio.run(scenario())
    assert overhead["server_poll_errors"] >= 2 and not collector.server_samples
    assert collector.gpu_status == "not_requested"
    errors = [json.loads(line) for line in (tmp_path / "t" / "server.jsonl").read_text().splitlines()]
    assert all("error" in record for record in errors)


def test_stop_cancels_every_poller_and_closes_the_files(tmp_path):
    async def scenario():
        async with TestServer(health_app([])) as server:
            collector = telemetry.Collector(tmp_path / "t", url=str(server.make_url("")), interval_s=0.05,
                                            executable=fake_smi(tmp_path, ROWS))
            await collector.start()
            await asyncio.sleep(0.15)
            await collector.stop()
            return collector

    collector = asyncio.run(scenario())
    assert all(task.done() for task in collector._tasks)
    assert all(handle.closed for handle in collector._files.values())


def test_a_nonpositive_interval_is_refused():
    with pytest.raises(ValueError, match="interval_s"):
        telemetry.Collector("x", interval_s=0)
