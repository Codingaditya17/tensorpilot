"""Deterministic scheduling tests use an injected clock, never GPU timings."""
import asyncio
from types import SimpleNamespace

import pytest

from tensorpilot.scheduler import Scheduler


def job(name, budget, admitted=0):
    return SimpleNamespace(name=name, max_tokens=budget, admitted=admitted, schedule_reason=None)


@pytest.mark.parametrize("policy,expected", [("fifo", ["long", "short", "medium"]),
                                            ("shortest-budget", ["short", "medium", "long"])])
def test_policy_selection(policy, expected):
    queue = Scheduler(3, policy, max_wait_s=10, clock=lambda: 0)
    for item in [job("long", 128), job("short", 8), job("medium", 32)]:
        queue.put_nowait(item)
    assert [queue.get_nowait().name for _ in range(3)] == expected
    with pytest.raises(asyncio.QueueEmpty):
        queue.get_nowait()


def test_equal_budgets_preserve_arrival_order():
    queue = Scheduler(3, "shortest-budget", clock=lambda: 0)
    for name in ("a", "b", "c"):
        queue.put_nowait(job(name, 16))
    assert [queue.get_nowait().name for _ in range(3)] == ["a", "b", "c"]


def test_aging_prevents_continuing_short_jobs_from_bypassing_old_job():
    now = [0.0]
    queue = Scheduler(2, "shortest-budget", max_wait_s=1, clock=lambda: now[0])
    old = job("old-long", 256)
    queue.put_nowait(old)
    for index in range(1, 10):
        now[0] = index / 10
        short = job(f"short-{index}", 1, admitted=now[0])
        queue.put_nowait(short)
        assert queue.get_nowait() is short
        queue.task_done()
    now[0] = 1.0
    queue.put_nowait(job("new-short", 1, admitted=1))
    assert queue.get_nowait() is old
    assert old.schedule_reason == "aging"


def test_capacity_and_discard_release_exactly_one_slot():
    async def scenario():
        queue = Scheduler(1)
        first = job("a", 16)
        queue.put_nowait(first)
        with pytest.raises(asyncio.QueueFull):
            queue.put_nowait(job("b", 16))
        assert queue.discard(job("a", 16)) is False  # Identity, not value equality.
        assert queue.discard(first) is True
        assert queue.discard(first) is False
        await asyncio.wait_for(queue.join(), .1)
        second = job("b", 16)
        queue.put_nowait(second)
        assert queue.get_nowait() is second
        queue.task_done()
        await asyncio.wait_for(queue.join(), .1)
        with pytest.raises(ValueError):
            queue.task_done()
    asyncio.run(scenario())


def test_consumer_wakes_after_admission():
    async def scenario():
        queue = Scheduler(1)
        waiting = asyncio.create_task(queue.get())
        await asyncio.sleep(0)
        item = job("wakeup", 8)
        queue.put_nowait(item)
        assert await asyncio.wait_for(waiting, .1) is item
        queue.task_done()
    asyncio.run(scenario())


@pytest.mark.parametrize("kwargs", [{"capacity": 0}, {"capacity": True},
                                   {"policy": "unknown"}, {"max_wait_s": 0},
                                   {"max_wait_s": float("nan")}])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        Scheduler(**kwargs)
