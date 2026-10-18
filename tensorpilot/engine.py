"""Bounded admission and scheduling; one dedicated thread owns model execution."""
import asyncio
import json
import logging
import math
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .scheduler import Scheduler

LOGGER = logging.getLogger(__name__)
TERMINAL_STATES = frozenset(("completed", "cancelled", "timed_out", "failed"))
PUBLIC_ERRORS = {
    "cancelled": "Request cancelled",
    "deadline_exceeded": "Request deadline exceeded",
    "invalid_prompt": "Prompt cannot be processed",
    "inference_error": "Inference failed",
    "shutting_down": "Server shutting down",
    "backend_unavailable": "Inference worker is unavailable",
    "slow_consumer": "Response buffer is full; request shed to protect other sequences",
}


class Overloaded(Exception):
    pass


class BackendUnavailable(RuntimeError):
    pass


@dataclass
class Job:
    prompt: str
    max_tokens: int
    deadline: float
    admitted: float = field(default_factory=time.perf_counter)
    events: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(16))
    cancelled: bool = False
    cancel_code: str | None = None
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    state: str = "queued"
    schedule_reason: str | None = None
    expiry_task: asyncio.Task | None = field(default=None, repr=False)


def advance(iterator):
    # StopIteration cannot cross an asyncio Future boundary.
    return next(iterator, None)


class Engine:
    execution_mode = "serial"
    max_active_sequences = 1

    def __init__(self, factory, queue_size=8, metrics=None, *, scheduler="fifo", max_queue_wait_s=1.0,
                 stall_timeout_s=30.0):
        if not math.isfinite(stall_timeout_s) or stall_timeout_s <= 0:
            raise ValueError("stall_timeout_s must be finite and positive")
        self.factory = factory
        self.metrics = metrics
        self.queue = Scheduler(queue_size, scheduler, max_queue_wait_s)
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tensorpilot-mlx")
        self.closing = False
        self.unhealthy = False
        self.unhealthy_reason = None
        self.stall_timeout_s = stall_timeout_s
        self.call_started_s = None
        self.active = None
        self.worker = None
        self.watchdog = None
        self.last_memory_sample_s = None
        self._close_task = None

    @property
    def ready(self):
        return not self.closing and not self.unhealthy and self.worker is not None and not self.worker.done()

    @property
    def active_count(self):
        return int(self.active is not None)

    async def call(self, function, *args):
        self.call_started_s = time.perf_counter()
        try:
            return await asyncio.get_running_loop().run_in_executor(self.pool, function, *args)
        finally:
            self.call_started_s = None

    async def start(self, warmup=True):
        try:
            self.backend = await self.call(self.factory)
            if warmup:
                await self.call(lambda: list(self.backend.stream("Hello", 8)))
            self._observe_memory(await self.call(self.backend.memory))
            await self._prepare_execution()
        except BaseException:
            self.pool.shutdown(wait=False)
            raise
        self.worker = asyncio.create_task(self.run(), name="tensorpilot-model-worker")
        self.worker.add_done_callback(self._worker_stopped)
        self.watchdog = asyncio.create_task(self._monitor_progress(), name="tensorpilot-worker-watchdog")

    async def _prepare_execution(self):
        """Execution-specific initialization, before admission/worker startup."""

    def _observe_memory(self, sample):
        self.last_memory_sample_s = time.perf_counter()
        if self.metrics:
            self.metrics.update_memory(sample)

    async def _monitor_progress(self):
        while not self.closing and not self.unhealthy:
            await asyncio.sleep(min(.25, self.stall_timeout_s / 4))
            if self.call_started_s is not None and time.perf_counter() - self.call_started_s >= self.stall_timeout_s:
                if self.metrics:
                    self.metrics.watchdog_trips.inc()
                self._degrade("worker_stalled")

    def _degrade(self, reason):
        if self.unhealthy:
            return
        self.unhealthy = True
        self.unhealthy_reason = reason
        LOGGER.error(json.dumps({"event": "backend_unhealthy", "reason": reason}))
        self._drain("backend_unavailable", "failed")
        if self.active:
            self._finish_error(self.active, "backend_unavailable", "failed", reason)

    def _worker_stopped(self, task):
        if self.closing:
            return
        self._degrade("worker_stopped")
        error_type = "CancelledError" if task.cancelled() else type(task.exception()).__name__
        LOGGER.error(json.dumps({"event": "worker_stopped", "error_type": error_type}))
        self._drain("backend_unavailable", "failed")

    def submit(self, prompt, max_tokens, timeout_s, request_id=None):
        if not self.ready:
            if self.metrics:
                self.metrics.rejected.inc()
            raise Overloaded("inference worker unavailable")
        now = time.perf_counter()
        job = Job(prompt, max_tokens, now + timeout_s, admitted=now)
        if request_id:
            job.request_id = request_id
        try:
            self.queue.put_nowait(job)
        except asyncio.QueueFull as exc:
            if self.metrics:
                self.metrics.rejected.inc()
            raise Overloaded("request queue full") from exc
        job.expiry_task = asyncio.create_task(self._expire_waiting(job), name="tensorpilot-queue-deadline")
        if self.metrics:
            self.metrics.admitted.inc()
        return job

    async def _expire_waiting(self, job):
        await asyncio.sleep(max(0, job.deadline - time.perf_counter()))
        if self.queue.discard(job):
            self._finish_error(job, "deadline_exceeded", "timed_out", "TimeoutError")

    @staticmethod
    def _stop_expiry(job):
        if job.expiry_task is not None:
            if job.expiry_task is not asyncio.current_task():
                job.expiry_task.cancel()
            job.expiry_task = None

    def _record_outcome(self, job, outcome):
        job.state = outcome
        if self.metrics:
            self.metrics.outcomes.labels(outcome=outcome).inc()
        LOGGER.info(json.dumps({"event": "job_finished", "request_id": job.request_id,
                                "outcome": outcome, "elapsed_s": time.perf_counter() - job.admitted}))

    def _finish_error(self, job, code, outcome, error_type=None):
        if job.state in TERMINAL_STATES:
            return
        self._stop_expiry(job)
        while not job.events.empty():
            job.events.get_nowait()
        event = {"type": "error", "error": PUBLIC_ERRORS[code], "code": code}
        if error_type:
            event["error_type"] = error_type
        job.events.put_nowait(event)
        self._record_outcome(job, outcome)

    def cancel(self, job):
        if job.state in TERMINAL_STATES:
            return
        job.cancelled = True
        job.cancel_code = "deadline_exceeded" if time.perf_counter() >= job.deadline else "cancelled"
        if self.queue.discard(job):
            outcome = "timed_out" if job.cancel_code == "deadline_exceeded" else "cancelled"
            self._finish_error(job, job.cancel_code, outcome)

    def _drain(self, code, outcome):
        while not self.queue.empty():
            job = self.queue.get_nowait()
            self.queue.task_done()
            if job is not None:
                self._finish_error(job, code, outcome)

    async def emit(self, job, event):
        while True:
            remaining = job.deadline - time.perf_counter()
            if job.cancelled or remaining <= 0:
                raise TimeoutError("request cancelled or deadline exceeded")
            try:
                await asyncio.wait_for(job.events.put(event), min(.1, remaining))
                return
            except TimeoutError:
                continue

    async def run(self):
        while True:
            job = await self.queue.get()
            if job is None:
                self.queue.task_done()
                return
            self._stop_expiry(job)
            job.state = "running"
            self.active = job
            try:
                await self._execute(job)
            finally:
                self.active = None
                self.queue.task_done()

    async def _execute(self, job):
        iterator = None
        started = time.perf_counter()
        token_times, sampled = [], 0
        profiles = []
        prompt_tokens, finish_reason = None, None
        terminal = None
        code, outcome, error_type = None, None, None
        worker_cancelled = False
        if self.metrics:
            self.metrics.queue_wait.observe(started - job.admitted)
            self.metrics.scheduled.labels(reason=job.schedule_reason).inc()
        try:
            if self.unhealthy or self.closing:
                raise RuntimeError("inference worker unavailable")
            self._check_deadline(job)
            iterator = await self.call(lambda: iter(self.backend.stream(job.prompt, job.max_tokens)))
            while True:
                self._check_deadline(job)
                token = await self.call(advance, iterator)
                self._check_deadline(job)
                now = time.perf_counter()
                if token is None:
                    break
                sampled += 1
                prompt_tokens = token.prompt_tokens
                finish_reason = token.finish_reason
                if token.memory_profile is not None:
                    profiles.append({**token.memory_profile, "elapsed_s": now - job.admitted})
                if token.output:
                    token_times.append(now - job.admitted)
                    if self.metrics:
                        self.metrics.tokens.inc()
                        if len(token_times) == 1:
                            self.metrics.ttft.observe(token_times[0])
                await self.emit(job, {"type": "token", "text": token.text,
                                      "token_id": token.token_id, "output": token.output})
            memory = await self.call(self.backend.memory)
            self._observe_memory(memory)
            terminal = {"type": "done", "queue_wait_s": started - job.admitted,
                        "prompt_tokens": prompt_tokens, "finish_reason": finish_reason,
                        "ttft_s": token_times[0] if token_times else None,
                        "token_times_s": token_times, "generated_tokens": len(token_times),
                        "sampled_tokens_including_eos": sampled, "scheduler": self.queue.policy,
                        "schedule_reason": job.schedule_reason, **memory}
            if profiles:
                terminal["memory_profile"] = profiles
        except asyncio.CancelledError:
            worker_cancelled = True
            self._degrade("worker_stopped")
            code, outcome, error_type = "backend_unavailable", "failed", "CancelledError"
        except Exception as exc:
            code, outcome = self._classify(job, exc)
            error_type = type(exc).__name__
        finally:
            # Finalize backend resources BEFORE publishing successful completion.
            try:
                if iterator is not None:
                    await self.call(iterator.close)
            except Exception as exc:
                self._degrade("backend_cleanup_failed")
                code, outcome, error_type = "backend_unavailable", "failed", type(exc).__name__
                self._drain("backend_unavailable", "failed")
                LOGGER.error(json.dumps({"event": "backend_cleanup_failed", "request_id": job.request_id,
                                         "error_type": error_type}))
        if code is None:
            try:
                terminal["server_elapsed_s"] = time.perf_counter() - job.admitted
                await self.emit(job, terminal)
                self._record_outcome(job, "completed")
            except Exception as exc:
                code, outcome = self._classify(job, exc)
                error_type = type(exc).__name__
        if code is not None:
            self._finish_error(job, code, outcome, error_type)
        if worker_cancelled:
            raise asyncio.CancelledError

    def _check_deadline(self, job):
        if self.unhealthy:
            raise BackendUnavailable("inference worker unavailable")
        if job.cancelled or time.perf_counter() >= job.deadline:
            raise TimeoutError("request cancelled or deadline exceeded")

    @staticmethod
    def _classify(job, exc):
        if job.cancelled:
            code = job.cancel_code or "cancelled"
            return code, "timed_out" if code == "deadline_exceeded" else "cancelled"
        if isinstance(exc, TimeoutError):
            return "deadline_exceeded", "timed_out"
        if isinstance(exc, ValueError):
            return "invalid_prompt", "failed"
        if isinstance(exc, BackendUnavailable):
            return "backend_unavailable", "failed"
        return "inference_error", "failed"

    async def close(self):
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close())
        await asyncio.shield(self._close_task)

    async def _close(self):
        self.closing = True
        if self.watchdog is not None:
            self.watchdog.cancel()
            await asyncio.gather(self.watchdog, return_exceptions=True)
        if self.active:
            self.cancel(self.active)
        self._drain("shutting_down", "cancelled")
        if self.worker is not None and not self.worker.done():
            self.queue.put_nowait(None)
            await self.worker
        self.pool.shutdown(wait=True)
