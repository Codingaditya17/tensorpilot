import asyncio
import json

from tensorpilot.supervisor import supervise


class Process:
    def __init__(self, code, pid):
        self.code, self.pid, self.returncode = code, pid, None

    async def wait(self):
        self.returncode = self.code
        return self.code

    def send_signal(self, _signal):
        self.returncode = 0

    def kill(self):
        self.returncode = -9


def test_supervisor_restarts_failed_and_clean_unrequested_exits(tmp_path):
    async def scenario():
        processes = iter((Process(17, 101), Process(0, 102)))

        async def start(*_command, **_options):
            return next(processes)

        async def ready(process, _url, _timeout):
            assert process.returncode is None

        lifecycle = tmp_path / "lifecycle.jsonl"
        result = await supervise(["worker"], "http://unused", lifecycle, tmp_path / "worker.log",
                                 max_restarts=1, backoff_s=0, start_factory=start, readiness=ready)
        assert result == 1
        records = [json.loads(line) for line in lifecycle.read_text().splitlines()]
        assert [record["event"] for record in records].count("worker_ready") == 2
        assert any(record["event"] == "worker_restart_scheduled" for record in records)
        assert records[-1]["event"] == "restart_budget_exhausted"
        assert records[-2]["exit_code"] == 0
    asyncio.run(scenario())


def test_supervisor_stops_after_restart_budget(tmp_path):
    async def scenario():
        async def start(*_command, **_options):
            return Process(1, 101)

        async def ready(_process, _url, _timeout):
            return None

        lifecycle = tmp_path / "lifecycle.jsonl"
        result = await supervise(["worker"], "http://unused", lifecycle, tmp_path / "worker.log",
                                 max_restarts=1, backoff_s=0, start_factory=start, readiness=ready)
        assert result == 1
        assert "restart_budget_exhausted" in lifecycle.read_text()
    asyncio.run(scenario())
