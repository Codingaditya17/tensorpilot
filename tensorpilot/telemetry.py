"""Timestamped GPU and server telemetry for an experiment window, written as raw JSONL plus a summary.

The baseline GPU run had only single ``nvidia-smi`` snapshots. This collector samples on a timer for the whole
experiment so utilization, memory, power, clocks and the server's own state (queue depth, batch width, KV pool,
CUDA allocator, OOM counter) can be lined up against request timing.

Overhead is measured, not assumed: every poll records its own latency, and the summary reports the sampler's
call counts. The only way to know the effect on throughput is to run the same workload with and without it.

Files (one set per experiment): ``gpu.jsonl``, ``server.jsonl``, ``events.jsonl``. Every record carries a
monotonic time ``t`` (seconds since the collector started) and a UTC wall time.
"""
import asyncio
import json
import shutil
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import aiohttp

GPU_FIELDS = (
    ("index", "index", int), ("utilization.gpu", "util_gpu_pct", float), ("utilization.memory", "util_mem_pct", float),
    ("memory.used", "mem_used_mib", float), ("memory.total", "mem_total_mib", float), ("power.draw", "power_w", float),
    ("temperature.gpu", "temp_c", float), ("clocks.sm", "clock_sm_mhz", float), ("clocks.mem", "clock_mem_mhz", float),
    ("clocks_throttle_reasons.active", "throttle_mask", str),
)
GPU_QUERY = ",".join(name for name, _, _ in GPU_FIELDS)
UNKNOWN = frozenset(("[N/A]", "[Not Supported]", "N/A", ""))
THROTTLE_BITS = ((0x2, "applications_clocks_setting"), (0x4, "sw_power_cap"), (0x8, "hw_slowdown"),
                 (0x10, "sync_boost"), (0x20, "sw_thermal_slowdown"), (0x40, "hw_thermal_slowdown"),
                 (0x80, "hw_power_brake_slowdown"), (0x100, "display_clock_setting"))


def decode_throttle(mask):
    """Names of the active clock-throttle reasons. The idle bit (0x1) is not a throttle and is ignored."""
    try:
        value = int(mask, 16)
    except (TypeError, ValueError):
        return None
    return [name for bit, name in THROTTLE_BITS if value & bit]


def parse_gpu_csv(text):
    devices = []
    for line in text.strip().splitlines():
        values = [piece.strip() for piece in line.split(",")]
        if len(values) != len(GPU_FIELDS):
            raise ValueError("unexpected nvidia-smi output shape")
        device = {}
        for (_, key, convert), value in zip(GPU_FIELDS, values):
            device[key] = None if value in UNKNOWN else (value if convert is str else convert(value))
        devices.append(device)
    return devices


def _percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * fraction
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def _stats(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {"mean": statistics.fmean(values), "p95": _percentile(values, .95), "max": max(values),
            "min": min(values), "samples": len(values)}


def server_view(health):
    """The fields of /health worth sampling. Anything the server did not report stays None."""
    pool = health.get("kv_pool") or {}
    memory = health.get("last_measured_memory") or {}
    admission = health.get("admission") or {}
    return {"ready": health.get("ready"), "queue_depth": health.get("queue_depth"),
            "active_sequences": health.get("active_sequences"),
            "execution_concurrency": health.get("execution_concurrency"),
            "kv_rows_active": pool.get("decode_rows_active"), "kv_prefill_waiting": pool.get("prefill_waiting"),
            "kv_bytes_filled": pool.get("kv_bytes_filled"), "kv_fill_ratio": pool.get("kv_fill_ratio_of_active_rows"),
            "kv_pool_bytes": pool.get("pool_bytes_allocated"),
            "decode_steps": pool.get("decode_steps"), "decode_tokens": pool.get("decode_tokens"),
            "decode_execution_s": pool.get("decode_execution_s"), "prefill_chunks": pool.get("prefill_chunks"),
            "prefill_tokens": pool.get("prefill_tokens"), "prefill_execution_s": pool.get("prefill_execution_s"),
            "cuda_allocated_bytes": memory.get("cuda_allocated_bytes"),
            "cuda_reserved_bytes": memory.get("cuda_reserved_bytes"),
            "cuda_peak_reserved_bytes": memory.get("cuda_peak_reserved_bytes"),
            "cuda_oom_events": memory.get("cuda_oom_events"),
            "admission_reservations": admission.get("reservations"),
            "unhealthy_reason": health.get("unhealthy_reason")}


class Collector:
    def __init__(self, directory, *, url=None, api_key=None, interval_s=1.0, gpu=True, executable="nvidia-smi",
                 session_factory=aiohttp.ClientSession):
        if not interval_s > 0:
            raise ValueError("interval_s must be positive")
        self.directory = Path(directory)
        self.url = url.rstrip("/") if url else None
        self.headers = {"Authorization": "Bearer " + api_key} if api_key else {}
        self.interval_s = interval_s
        self.gpu = gpu
        self.executable = executable
        self.session_factory = session_factory
        self.gpu_samples, self.server_samples, self.events = [], [], []
        self.gpu_errors = self.server_errors = 0
        self.gpu_call_s, self.server_call_s = [], []
        self.gpu_status = "not_requested" if not gpu else "pending"
        self._tasks = []
        self._origin = None
        self._files = {}

    # ----- lifecycle --------------------------------------------------------------------------------
    async def start(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        self._files = {name: (self.directory / f"{name}.jsonl").open("w", encoding="utf-8")
                       for name in ("gpu", "server", "events")}
        self._origin = time.monotonic()
        if self.gpu:
            if shutil.which(self.executable) is None:
                self.gpu_status = "unavailable: nvidia-smi not found"
            else:
                self.gpu_status = "sampling"
                self._tasks.append(asyncio.create_task(self._poll_gpu(), name="tensorpilot-telemetry-gpu"))
        if self.url:
            self._tasks.append(asyncio.create_task(self._poll_server(), name="tensorpilot-telemetry-server"))
        self.mark("collector_started", interval_s=self.interval_s, gpu_status=self.gpu_status)
        return self

    async def stop(self):
        self.mark("collector_stopping")
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        for handle in self._files.values():
            handle.close()
        return self.overhead()

    async def __aenter__(self):
        return await self.start()

    async def __aexit__(self, *exc):
        await self.stop()

    # ----- recording --------------------------------------------------------------------------------
    def now(self):
        return time.monotonic() - self._origin

    def _write(self, stream, record):
        record = {"t": round(self.now(), 4), "utc": datetime.now(timezone.utc).isoformat(), **record}
        self._files[stream].write(json.dumps(record, separators=(",", ":")) + "\n")
        self._files[stream].flush()
        return record

    def mark(self, label, **fields):
        self.events.append(self._write("events", {"event": label, **fields}))
        return self.events[-1]

    async def _poll_gpu(self):
        while True:
            began = time.perf_counter()
            process = None
            try:
                process = await asyncio.create_subprocess_exec(
                    self.executable, f"--query-gpu={GPU_QUERY}", "--format=csv,noheader,nounits",
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                out, _ = await asyncio.wait_for(process.communicate(), 5)
                if process.returncode != 0:
                    raise RuntimeError(f"nvidia-smi exited {process.returncode}")
                took = time.perf_counter() - began
                self.gpu_call_s.append(took)
                for device in parse_gpu_csv(out.decode()):
                    self.gpu_samples.append(self._write("gpu", {**device, "call_s": round(took, 4)}))
            except asyncio.CancelledError:
                raise
            except Exception as error:
                self.gpu_errors += 1
                self._write("gpu", {"error": type(error).__name__})
            finally:
                # A poll cancelled or timed out mid-flight must not leave its child process behind.
                if process is not None and process.returncode is None:
                    process.kill()
                    await asyncio.shield(process.wait())
            await asyncio.sleep(max(0, self.interval_s - (time.perf_counter() - began)))

    async def _poll_server(self):
        async with self.session_factory(headers=self.headers, timeout=aiohttp.ClientTimeout(total=5)) as session:
            while True:
                began = time.perf_counter()
                try:
                    async with session.get(self.url + "/health") as response:
                        response.raise_for_status()
                        health = await response.json()
                    took = time.perf_counter() - began
                    self.server_call_s.append(took)
                    self.server_samples.append(self._write("server", {**server_view(health), "call_s": round(took, 4)}))
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    self.server_errors += 1
                    self._write("server", {"error": type(error).__name__})
                await asyncio.sleep(max(0, self.interval_s - (time.perf_counter() - began)))

    # ----- summaries --------------------------------------------------------------------------------
    def overhead(self):
        """What the sampler itself cost. Its effect on throughput is measured by an A/B run, not inferred here."""
        return {"interval_s": self.interval_s, "gpu_status": self.gpu_status,
                "gpu_polls": len(self.gpu_call_s), "gpu_poll_errors": self.gpu_errors,
                "gpu_mean_call_s": statistics.fmean(self.gpu_call_s) if self.gpu_call_s else None,
                "server_polls": len(self.server_call_s), "server_poll_errors": self.server_errors,
                "server_mean_call_s": statistics.fmean(self.server_call_s) if self.server_call_s else None,
                "note": "each GPU poll launches nvidia-smi once; compare a telemetry-off run to quantify its "
                        "effect on throughput"}

    def window(self, start_t, end_t):
        """Statistics over samples taken in [start_t, end_t] (collector-relative seconds)."""
        gpu = [s for s in self.gpu_samples if start_t <= s["t"] <= end_t]
        server = [s for s in self.server_samples if start_t <= s["t"] <= end_t]
        by_device = {}
        for sample in gpu:
            by_device.setdefault(sample.get("index"), []).append(sample)
        devices = {}
        for index, samples in by_device.items():
            reasons = sorted({reason for s in samples for reason in (decode_throttle(s.get("throttle_mask")) or [])})
            devices[str(index)] = {
                "utilization_gpu_pct": _stats([s["util_gpu_pct"] for s in samples]),
                "utilization_memory_pct": _stats([s["util_mem_pct"] for s in samples]),
                "memory_used_mib": _stats([s["mem_used_mib"] for s in samples]),
                "power_w": _stats([s["power_w"] for s in samples]),
                "temperature_c": _stats([s["temp_c"] for s in samples]),
                "clock_sm_mhz": _stats([s["clock_sm_mhz"] for s in samples]),
                "throttle_reasons_seen": reasons}
        decode_rate = None
        counted = [s for s in server if s.get("decode_tokens") is not None and s.get("decode_execution_s") is not None]
        if len(counted) >= 2:
            tokens = counted[-1]["decode_tokens"] - counted[0]["decode_tokens"]
            seconds = counted[-1]["decode_execution_s"] - counted[0]["decode_execution_s"]
            decode_rate = tokens / seconds if seconds > 0 else None
        oom = [s["cuda_oom_events"] for s in server if s.get("cuda_oom_events") is not None]
        return {"window_s": end_t - start_t, "gpu_samples": len(gpu), "server_samples": len(server),
                "gpu": devices,
                "queue_depth": _stats([s["queue_depth"] for s in server]),
                "active_sequences": _stats([s["active_sequences"] for s in server]),
                "kv_rows_active": _stats([s["kv_rows_active"] for s in server]),
                "kv_bytes_filled": _stats([s["kv_bytes_filled"] for s in server]),
                "cuda_reserved_bytes": _stats([s["cuda_reserved_bytes"] for s in server]),
                "cuda_oom_events_delta": (oom[-1] - oom[0]) if len(oom) >= 2 else None,
                "active_decode_tokens_per_s": decode_rate,
                "active_decode_note": "decode tokens divided by synchronized decode execution time over the window; "
                                      "unlike wall-clock throughput it excludes idle time and prefill"}
