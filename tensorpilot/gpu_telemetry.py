"""Measured NVIDIA device state from nvidia-smi, sampled in the background.

Samples are never simulated. Hosts without nvidia-smi report ``unavailable``, and
fields the driver marks ``[N/A]`` are reported as null rather than zero. nvidia-smi
is used instead of a Python binding so the sampler adds no dependency and works in
the NVIDIA container runtime, where the binary is injected by the host.
"""
import asyncio
from datetime import datetime, timezone
import math
import shutil
import time

FIELDS = (
    ("index", "index", int),
    ("name", "name", str),
    ("driver_version", "driver_version", str),
    ("compute_cap", "compute_capability", str),
    ("memory.total", "memory_total_mib", float),
    ("memory.used", "memory_used_mib", float),
    ("utilization.gpu", "utilization_gpu_percent", float),
    ("temperature.gpu", "temperature_c", float),
    ("power.draw", "power_draw_w", float),
    ("ecc.errors.uncorrected.volatile.total", "ecc_uncorrected_volatile", int),
)
QUERY = ",".join(name for name, _, _ in FIELDS)
UNKNOWN_VALUES = frozenset(("[N/A]", "[Not Supported]", "N/A", ""))


def parse_nvidia_smi(text):
    devices = []
    for line in text.strip().splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) != len(FIELDS):
            raise ValueError("unexpected nvidia-smi output shape")
        device = {}
        for (_, key, convert), value in zip(FIELDS, values):
            device[key] = None if value in UNKNOWN_VALUES else _convert(convert, value)
        devices.append(device)
    return devices


def _convert(convert, value):
    number = convert(value) if convert is not float else float(value)
    if isinstance(number, float) and not math.isfinite(number):
        return None
    return number


async def sample_gpus(*, executable="nvidia-smi", timeout_s=2.0):
    sampled_at = datetime.now(timezone.utc).isoformat()
    binary = shutil.which(executable)
    if binary is None:
        return {"status": "unavailable", "reason": "nvidia-smi not found", "sampled_at_utc": sampled_at,
                "devices": []}
    try:
        process = await asyncio.create_subprocess_exec(
            binary, f"--query-gpu={QUERY}", "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout_s)
        except TimeoutError:
            process.kill()
            await process.wait()
            return {"status": "error", "reason": "nvidia-smi timed out", "sampled_at_utc": sampled_at,
                    "devices": []}
    except OSError as exc:
        return {"status": "error", "reason": f"nvidia-smi could not start: {type(exc).__name__}",
                "sampled_at_utc": sampled_at, "devices": []}
    if process.returncode != 0:
        return {"status": "error", "reason": f"nvidia-smi exited {process.returncode}",
                "sampled_at_utc": sampled_at, "devices": []}
    try:
        devices = parse_nvidia_smi(stdout.decode())
    except (ValueError, UnicodeDecodeError) as exc:
        return {"status": "error", "reason": f"unparseable nvidia-smi output: {type(exc).__name__}",
                "sampled_at_utc": sampled_at, "devices": []}
    return {"status": "measured", "source": "nvidia-smi", "sampled_at_utc": sampled_at, "devices": devices}


class GPUTelemetry:
    """Keep the latest sample; the health endpoint never runs nvidia-smi itself."""

    def __init__(self, interval_s=2.0, *, sampler=sample_gpus, clock=time.monotonic):
        if not math.isfinite(interval_s) or interval_s <= 0:
            raise ValueError("interval_s must be finite and positive")
        self.interval_s = interval_s
        self.sampler = sampler
        self.clock = clock
        self.latest = {"status": "pending", "reason": "no sample taken yet", "devices": []}
        self.sampled_at = None
        self._task = None

    async def start(self):
        self._task = asyncio.create_task(self._run(), name="tensorpilot-gpu-telemetry")

    async def _run(self):
        while True:
            try:
                self.latest = await self.sampler()
            except Exception as exc:  # A sampler fault must not stop later samples or the server.
                self.latest = {"status": "error", "reason": f"sampler failed: {type(exc).__name__}",
                               "devices": []}
            self.sampled_at = self.clock()
            await asyncio.sleep(self.interval_s)

    def snapshot(self):
        age = None if self.sampled_at is None else max(0.0, self.clock() - self.sampled_at)
        return {**self.latest, "interval_s": self.interval_s, "sample_age_s": age}

    async def close(self):
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None
