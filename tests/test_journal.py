import asyncio
import json
from types import SimpleNamespace

from aiohttp.test_utils import TestClient, TestServer
import pytest

from tensorpilot.journal import Journal, recover
from tensorpilot.load import run, validate_config
from tensorpilot.server import create_app
from test_serving import FakeBackend


def record(index=0):
    return {"index": index, "ok": True, "class": "short", "client_elapsed_s": .2,
            "dispatch_offset_s": index * .3, "client_ttft_s": .01, "received_output_tokens": 2,
            "server": {"generated_tokens": 2, "queue_wait_s": .001, "token_times_s": [.01, .1]}}


def test_complete_journal_recovery_preserves_raw_records(tmp_path):
    path = tmp_path / "raw.jsonl"
    journal = Journal(path, {"run_id": "test"})
    journal.record(record())
    journal.record(record(1))
    journal.finish("completed", 1)
    result = recover(path)
    assert result["summary"]["successful_requests"] == 2
    assert result["summary"]["completed_output_tokens_per_s"] == 4
    assert result["metadata"]["recovery"]["run_status"] == "completed"
    assert result["requests"][0]["server"]["token_times_s"] == [.01, .1]
    with pytest.raises(FileExistsError):
        Journal(path, {})


@pytest.mark.parametrize("tail", [b'{"type":"request",', b'{"text":"\xf0\x9f'])
def test_recovery_tolerates_only_an_incomplete_final_line(tmp_path, tail):
    path = tmp_path / "interrupted.jsonl"
    journal = Journal(path, {})
    journal.record(record())
    journal.stream.close()  # Represents termination without a footer.
    with path.open("ab") as stream:
        stream.write(tail)
    recovered = recover(path)
    assert recovered["metadata"]["recovery"]["truncated_final_line"]
    assert recovered["metadata"]["recovery"]["run_status"] == "interrupted"
    assert recovered["summary"]["wall_s"] == .2
    assert recovered["summary"]["successful_requests"] == 1


def test_recovery_rejects_middle_corruption_and_duplicate_records(tmp_path):
    path = tmp_path / "bad.jsonl"
    journal = Journal(path, {})
    journal.record(record())
    journal.record(record())
    journal.finish("completed", 1)
    with pytest.raises(ValueError, match="duplicate"):
        recover(path)
    lines = path.read_bytes().splitlines(keepends=True)
    path.write_bytes(lines[0] + b"broken\n" + lines[-1])
    with pytest.raises(ValueError, match="corrupt"):
        recover(path)


def test_soak_caps_records_and_keeps_raw_arrays_only_in_journal(tmp_path):
    async def scenario():
        async with TestClient(TestServer(create_app(FakeBackend, warmup=False))) as http:
            args = SimpleNamespace(url=str(http.make_url("/")), requests=2, concurrency=2,
                                   rate=0, max_tokens=1, timeout=5, output="unused", duration_s=1,
                                   max_requests=5, journal=str(tmp_path / "soak.jsonl"), compact=True)
            result = await run(args)
            assert len(result["requests"]) == 5
            assert result["metadata"]["stop_reason"] == "request_cap_reached"
            assert "client_token_times_s" not in result["requests"][0]
            assert "token_times_s" not in result["requests"][0]["server"]
            raw = recover(args.journal)
            assert len(raw["requests"]) == 5
            assert raw["requests"][0]["server"]["token_times_s"]
            assert raw["summary"] == result["summary"]
    asyncio.run(scenario())


def test_duration_stops_dispatch_and_cancellation_leaves_recoverable_records(tmp_path):
    async def scenario():
        async with TestClient(TestServer(create_app(FakeBackend, warmup=False))) as http:
            args = SimpleNamespace(url=str(http.make_url("/")), requests=2, concurrency=1,
                                   rate=0, max_tokens=1, timeout=5, output="unused", duration_s=.02,
                                   journal=str(tmp_path / "duration.jsonl"))
            result = await run(args)
            assert result["metadata"]["stop_reason"] == "duration_elapsed"
            assert result["summary"]["wall_s"] >= .02
            assert len(result["requests"]) < 10000
            args.duration_s = 5
            args.journal = str(tmp_path / "cancelled.jsonl")
            task = asyncio.create_task(run(args))
            # Wait for a completed record rather than relying on a fixed sleep.
            for _ in range(100):
                if (tmp_path / "cancelled.jsonl").exists():
                    if b'"type":"request"' in (tmp_path / "cancelled.jsonl").read_bytes():
                        break
                await asyncio.sleep(.005)
            else:
                pytest.fail("no journal record was written")
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            recovered = recover(args.journal)
            assert recovered["metadata"]["recovery"]["run_status"] == "interrupted"
            assert recovered["summary"]["successful_requests"] >= 1
    asyncio.run(scenario())


@pytest.mark.parametrize("field,value", [("duration_s", 0), ("duration_s", float("inf")),
                                        ("max_requests", 100001), ("concurrency", 1000)])
def test_unsafe_workload_configuration_is_rejected(field, value):
    args = SimpleNamespace(requests=10, concurrency=1, rate=0, max_tokens=32, timeout=5)
    setattr(args, field, value)
    with pytest.raises(ValueError):
        validate_config(args)


def test_footer_count_mismatch_is_not_reported_as_valid(tmp_path):
    path = tmp_path / "count.jsonl"
    entries = [{"type": "metadata", "metadata": {}}, {"type": "request", "record": record()},
               {"type": "end", "status": "completed", "wall_s": 1, "record_count": 2}]
    path.write_text("".join(json.dumps(entry) + "\n" for entry in entries))
    with pytest.raises(ValueError, match="count"):
        recover(path)
