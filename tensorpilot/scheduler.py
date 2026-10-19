"""Bounded single-consumer queue. Selection never preempts an active GPU job."""
import asyncio
import math
import time


class Scheduler:
    POLICIES = ("fifo", "shortest-budget")

    def __init__(self, capacity=8, policy="fifo", max_wait_s=1.0, clock=time.perf_counter):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        if policy not in self.POLICIES:
            raise ValueError(f"unknown scheduler policy: {policy}")
        if not math.isfinite(max_wait_s) or max_wait_s <= 0:
            raise ValueError("max_wait_s must be finite and positive")
        self.capacity = capacity
        self.policy = policy
        self.max_wait_s = max_wait_s
        self.clock = clock
        self._pending = []
        self._changed = asyncio.Event()
        self._idle = asyncio.Event()
        self._idle.set()
        self._unfinished = 0

    def qsize(self):
        return len(self._pending)

    def empty(self):
        return not self._pending

    def oldest_wait_s(self):
        jobs = [job for job in self._pending if job is not None]
        return max(0.0, self.clock() - min(job.admitted for job in jobs)) if jobs else 0.0

    def put_nowait(self, job):
        if self.qsize() >= self.capacity:
            raise asyncio.QueueFull
        self._pending.append(job)
        self._unfinished += 1
        self._idle.clear()
        self._changed.set()

    async def put(self, job):
        # Admission must never silently wait and bypass the overload policy.
        self.put_nowait(job)

    def get_nowait(self):
        if not self._pending:
            raise asyncio.QueueEmpty
        oldest = self._pending[0]
        index, reason = 0, "fifo"
        if oldest is not None and self.policy == "shortest-budget":
            if self.clock() - oldest.admitted >= self.max_wait_s:
                reason = "aging"
            else:
                index = min(range(len(self._pending)), key=lambda i: self._pending[i].max_tokens)
                reason = "shortest-budget"
        job = self._pending.pop(index)
        if job is not None:
            job.schedule_reason = reason
        return job

    async def get(self):
        while not self._pending:
            self._changed.clear()
            await self._changed.wait()
        return self.get_nowait()

    def discard(self, job):
        """Remove an exact waiting object and release its admission slot."""
        for index, pending in enumerate(self._pending):
            if pending is job:
                self._pending.pop(index)
                self.task_done()
                return True
        return False

    def task_done(self):
        if self._unfinished <= 0:
            raise ValueError("task_done called too many times")
        self._unfinished -= 1
        if self._unfinished == 0:
            self._idle.set()

    async def join(self):
        await self._idle.wait()
