"""Opt-in continuous batching with bounded per-sequence delivery buffers."""
import asyncio
from dataclasses import dataclass, field
import time

from .batch_backend import MLXBatchAdapter
from .adaptive import AdaptiveBatchController
from .engine import Engine, TERMINAL_STATES


@dataclass
class Running:
    job: object
    started: float
    token_times: list = field(default_factory=list)
    sampled: int = 0
    max_decode_width: int = 0
    insert_profile: dict | None = None
    prefill_execution_s: float = 0.0
    decode_execution_s: float = 0.0
    batch_steps: int = 0


class BatchEngine(Engine):
    execution_mode = "batch"

    def __init__(self, factory, queue_size=8, metrics=None, *, max_batch_size=2,
                 prefill_step_size=256, adapter_factory=MLXBatchAdapter,
                 adaptive_controller=None, **kwargs):
        if type(max_batch_size) is not int or not 1 <= max_batch_size <= 8:
            raise ValueError("max_batch_size must be in 1..8")
        if type(prefill_step_size) is not int or not 1 <= prefill_step_size <= 2048:
            raise ValueError("prefill_step_size must be in 1..2048")
        super().__init__(factory, queue_size, metrics, **kwargs)
        self.max_active_sequences = max_batch_size
        self.prefill_step_size = prefill_step_size
        self.adapter_factory = adapter_factory
        self.running = {}
        self.admitting = None
        self.profile_totals = {"prefill_steps": 0, "decode_steps": 0,
                               "prefill_execution_s": 0.0, "decode_execution_s": 0.0,
                               "occupancy_sum": 0, "steps": 0}
        self.adaptive_controller = adaptive_controller
        if adaptive_controller is not None:
            if not isinstance(adaptive_controller, AdaptiveBatchController):
                raise TypeError("adaptive_controller must be an AdaptiveBatchController")
            if adaptive_controller.max_width != max_batch_size:
                raise ValueError("adaptive controller width must match max_batch_size")
            self.execution_mode = "adaptive"

    @property
    def active_count(self):
        return len(self.running) + int(self.admitting is not None)

    def _refresh_active(self):
        self.active = next((state.job for state in self.running.values()), self.admitting)

    def runtime_profile(self):
        steps = self.profile_totals["steps"]
        return {"measurement_source": "synchronized_model_thread_wall_clock",
                **self.profile_totals,
                "mean_occupancy": self.profile_totals["occupancy_sum"] / steps if steps else None}

    async def _prepare_execution(self):
        if getattr(self.backend, "profile_memory", False):
            raise ValueError("per-token memory profiling is not supported by batch execution yet")
        self.adapter = await self.call(lambda: self.adapter_factory(
            self.backend, capacity=self.max_active_sequences, prefill_step_size=self.prefill_step_size))

    def _degrade(self, reason):
        super()._degrade(reason)
        for state in self.running.values():
            self._finish_error(state.job, "backend_unavailable", "failed", reason)
        if self.admitting is not None:
            self._finish_error(self.admitting, "backend_unavailable", "failed", reason)

    async def _admit(self, job):
        self._stop_expiry(job)
        job.state = "running"
        self.admitting = job
        self._refresh_active()
        started = time.perf_counter()
        handed_off = False
        try:
            self._check_deadline(job)
            if self.metrics:
                self.metrics.queue_wait.observe(started - job.admitted)
                self.metrics.scheduled.labels(reason=job.schedule_reason).inc()
            uid = await self.call(self.adapter.insert, job.prompt, job.max_tokens)
            if self.unhealthy:
                return  # Global adapter cleanup also removes this UID.
            profile = getattr(self.adapter, "last_insert_profile", None)
            self.running[uid] = Running(job, started, insert_profile=dict(profile) if profile else None)
            handed_off = True
        except asyncio.CancelledError:
            self._finish_error(job, "backend_unavailable", "failed", "CancelledError")
            self._degrade("worker_stopped")
            raise
        except (ValueError, TimeoutError) as exc:
            code, outcome = self._classify(job, exc)
            self._finish_error(job, code, outcome, type(exc).__name__)
        except Exception as exc:
            self._finish_error(job, "backend_unavailable", "failed", type(exc).__name__)
            self._degrade("batch_insert_failed")
        finally:
            self.admitting = None
            self._refresh_active()
            if not handed_off:
                self.queue.task_done()

    def _retire(self, uid):
        state = self.running.pop(uid)
        self.queue.task_done()
        self._refresh_active()
        return state

    async def _abort(self, uid, code, outcome):
        state = self.running[uid]
        # Remove backend state before announcing a per-sequence failure. Failure
        # to remove invalidates the entire shared execution context.
        try:
            await self.call(self.adapter.remove, uid)
        except Exception:
            self._degrade("batch_remove_failed")
            return
        self._finish_error(state.job, code, outcome)
        self._retire(uid)

    async def _prune(self):
        for uid, state in list(self.running.items()):
            if self.unhealthy:
                return
            job = state.job
            if job.cancelled or time.perf_counter() >= job.deadline:
                code, outcome = self._classify(job, TimeoutError())
                await self._abort(uid, code, outcome)

    async def _offer(self, uid, event):
        job = self.running[uid].job
        try:
            job.events.put_nowait(event)
            return True
        except asyncio.QueueFull:
            await self._abort(uid, "slow_consumer", "failed")
            return False

    async def _step(self):
        result = await self.call(self.adapter.step)
        events = getattr(result, "events", result)
        phase = getattr(result, "phase", "decode" if events else "prefill")
        execution_s = getattr(result, "synchronized_execution_s", None)
        occupancy = getattr(result, "occupancy", len(self.running))
        if self.unhealthy:
            return
        await self._prune()
        if self.unhealthy:
            return
        if self.metrics:
            self.metrics.batch_steps.inc()
            self.metrics.batch_occupancy.observe(occupancy)
            if execution_s is not None:
                self.metrics.batch_step_seconds.labels(phase=phase).observe(execution_s)
            if events:
                self.metrics.decode_width.observe(len(events))
        self.profile_totals["steps"] += 1
        self.profile_totals["occupancy_sum"] += occupancy
        self.profile_totals[f"{phase}_steps"] += 1
        if execution_s is not None:
            self.profile_totals[f"{phase}_execution_s"] += execution_s
        memory = None
        for state in self.running.values():
            state.batch_steps += 1
            if execution_s is not None:
                if phase == "prefill":
                    state.prefill_execution_s += execution_s
                else:
                    state.decode_execution_s += execution_s
        for uid, token in events:
            if uid not in self.running:
                continue  # Sequence expired/cancelled while the step executed.
            state = self.running[uid]
            job = state.job
            state.sampled += 1
            state.max_decode_width = max(state.max_decode_width, len(events))
            if token.output:
                state.token_times.append(time.perf_counter() - job.admitted)
                if self.metrics:
                    self.metrics.tokens.inc()
                    if len(state.token_times) == 1:
                        self.metrics.ttft.observe(state.token_times[0])
            if not await self._offer(uid, {"type": "token", "text": token.text,
                                          "token_id": token.token_id, "output": token.output}):
                if self.unhealthy:
                    return
                continue
            if token.finish_reason is None:
                continue
            if memory is None:
                memory = await self.call(self.backend.memory)
                self._observe_memory(memory)
            if self.unhealthy:
                return
            # Backend.step finalized the detokenizer and removed this sequence's
            # cache membership before returning its finishing token.
            event = {"type": "done", "queue_wait_s": state.started - job.admitted,
                     "server_elapsed_s": time.perf_counter() - job.admitted,
                     "ttft_s": state.token_times[0] if state.token_times else None,
                     "token_times_s": state.token_times, "generated_tokens": len(state.token_times),
                     "sampled_tokens_including_eos": state.sampled, "prompt_tokens": token.prompt_tokens,
                     "finish_reason": token.finish_reason, "execution_mode": self.execution_mode,
                     "max_decode_batch_width": state.max_decode_width,
                     "inter_token_latency_s": [later - earlier for earlier, later in zip(state.token_times, state.token_times[1:])],
                     "phase_profile": {"measurement_source": "synchronized_model_thread_wall_clock",
                                       "insert": state.insert_profile,
                                       "prefill_execution_s": state.prefill_execution_s,
                                       "decode_execution_s": state.decode_execution_s,
                                       "observed_batch_steps": state.batch_steps},
                     "estimated_kv_bytes": job.resource_estimate.kv_bytes,
                     "estimated_total_tokens_upper_bound": job.resource_estimate.total_tokens_upper_bound,
                     "runtime_profile_snapshot": self.runtime_profile(),
                     "memory_scope": "shared batch process snapshot",
                     "scheduler": self.queue.policy, "schedule_reason": job.schedule_reason, **memory}
            if job.cancelled or time.perf_counter() >= job.deadline:
                code, outcome = self._classify(job, TimeoutError())
                await self._abort(uid, code, outcome)
            elif await self._offer(uid, event):
                self._record_outcome(job, "completed")
                self._retire(uid)

    async def run(self):
        try:
            while not self.closing and not self.unhealthy:
                await self._prune()
                if self.unhealthy:
                    break
                if not self.running:
                    job = await self.queue.get()
                    if job is None:
                        self.queue.task_done()
                        return
                    await self._admit(job)
                target_width = self.max_active_sequences
                if self.adaptive_controller is not None:
                    decision = self.adaptive_controller.choose(
                        queue_depth=self.queue.qsize(), active_sequences=len(self.running),
                        oldest_wait_s=self.queue.oldest_wait_s(),
                        estimated_memory_pressure=self.admission.reserved_bytes / self.admission.budget_bytes)
                    target_width = decision.width
                    if self.metrics:
                        self.metrics.adaptive_width.set(target_width)
                        self.metrics.adaptive_decisions.labels(reason=decision.reason,
                                                               width=str(target_width)).inc()
                while (not self.closing and not self.unhealthy and
                       len(self.running) < target_width and not self.queue.empty()):
                    job = self.queue.get_nowait()
                    if job is None:
                        self.queue.task_done()
                        return
                    await self._admit(job)
                if self.running and not self.closing and not self.unhealthy:
                    await self._step()
        except asyncio.CancelledError:
            self._degrade("worker_stopped")
            raise
        except Exception:
            self._degrade("batch_step_failed")
        finally:
            try:
                await self.call(self.adapter.close)
            except Exception:
                self._degrade("backend_cleanup_failed")
            for uid in list(self.running):
                job = self.running[uid].job
                if job.state not in TERMINAL_STATES:
                    self._finish_error(job, "shutting_down" if self.closing else "backend_unavailable",
                                       "cancelled" if self.closing else "failed")
                self._retire(uid)
            self._drain("shutting_down" if self.closing else "backend_unavailable",
                        "cancelled" if self.closing else "failed")
