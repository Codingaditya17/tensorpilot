"""The OpenAI-compatible client (used for the vLLM comparison) against a fake SSE server."""
import argparse
import asyncio
import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from tensorpilot import load
from tensorpilot.server import create_app
from tensorpilot.workloads import validate_workload
from test_serving import FakeBackend


def args(**overrides):
    values = dict(url="", requests=1, concurrency=1, rate=0, max_tokens=8, timeout=10, workload="uniform",
                  workload_file=None, seed=7, duration_s=None, max_requests=1000, journal=None, journal_fsync=False,
                  compact=False, overwrite=True, arrival="uniform", burst_size=8, slo=None, api="openai",
                  model_name="Qwen/Qwen2.5-7B-Instruct")
    values.update(overrides)
    return argparse.Namespace(**values)


def sse(payload):
    return (f"data: {payload if isinstance(payload, str) else json.dumps(payload)}\n\n").encode()


class FakeVLLM:
    """Streams like vLLM's /v1/completions: text chunks, a finish chunk, a usage chunk, then [DONE]."""

    def __init__(self, *, mode="normal", chunks=6, delay=0.0, empty_every=0, models=("Qwen/Qwen2.5-7B-Instruct",)):
        self.mode, self.chunks, self.delay, self.empty_every, self.models = mode, chunks, delay, empty_every, models
        self.bodies, self.disconnects = [], 0
        self.app = web.Application()
        self.app.router.add_get("/v1/models", self.list_models)
        self.app.router.add_post("/v1/completions", self.complete)

    async def list_models(self, request):
        return web.json_response({"data": [{"id": m, "max_model_len": 4096} for m in self.models]})

    async def complete(self, request):
        body = await request.json()
        self.bodies.append(body)
        if self.mode == "bad_request":
            return web.json_response({"error": {"type": "BadRequestError", "code": 400, "message": "nope"}}, status=400)
        if self.mode == "busy":
            return web.json_response({"error": {"type": "server_busy", "message": "full"}}, status=503)
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        produced = 0
        try:
            for index in range(min(self.chunks, body["max_tokens"])):
                await asyncio.sleep(self.delay)
                text = "" if self.empty_every and index % self.empty_every == 1 else f"w{index} "
                produced += 1
                await response.write(sse({"choices": [{"text": text, "finish_reason": None}]}))
                if self.mode == "stream_error" and index == 2:
                    await response.write(sse({"error": {"message": "engine died"}}))
                    break
            else:
                if self.mode == "truncated":
                    return response
                await response.write(sse({"choices": [{"text": "", "finish_reason": "length"}]}))
                await response.write(sse({"choices": [], "usage": {"prompt_tokens": 9, "completion_tokens": produced,
                                                                    "total_tokens": 9 + produced}}))
                await response.write(sse("[DONE]"))
            await response.write_eof()
        except ConnectionError:
            self.disconnects += 1
        return response


async def drive(fake, plan, tmp_path, **overrides):
    async with TestServer(fake.app) as server:
        path = tmp_path / "plan.json"
        path.write_text(json.dumps(validate_workload(plan)))
        return await load.run(args(url=str(server.make_url("")), requests=len(plan), workload_file=str(path),
                                   **overrides))


def item(**extra):
    return {"prompt": " the of and", "max_tokens": 6, "class": "c", **extra}


def test_requests_pin_greedy_sampling_so_a_model_repo_generation_config_cannot_leak_in(tmp_path):
    fake = FakeVLLM()
    result = asyncio.run(drive(fake, [item(), item(max_tokens=4)], tmp_path))
    assert result["summary"]["successful_requests"] == 2
    body = fake.bodies[0]
    assert body == {"model": "Qwen/Qwen2.5-7B-Instruct", "prompt": " the of and", "max_tokens": 6, "stream": True,
                    "stream_options": {"include_usage": True}, "temperature": 0, "top_p": 1.0, "top_k": -1,
                    "repetition_penalty": 1.0}
    assert "ignore_eos" not in body, "generation must stop at EOS exactly as the TensorPilot path does"
    assert fake.bodies[1]["max_tokens"] == 4


def test_token_arrivals_come_from_text_chunks_and_the_count_from_the_servers_usage(tmp_path):
    fake = FakeVLLM(chunks=6, delay=0.01, empty_every=2)  # chunks 1, 3, 5 carry no text
    record = asyncio.run(drive(fake, [item()], tmp_path))["requests"][0]
    assert record["ok"] and record["outcome"] == "completed"
    assert record["received_output_tokens"] == 3 and record["server"]["generated_tokens"] == 6
    assert record["client_ttft_s"] >= 0.01 and len(record["client_inter_token_s"]) == 2
    assert record["server"]["api"] == "openai" and record["server"]["finish_reason"] == "length"


def test_the_summary_works_without_server_side_queue_wait_and_computes_goodput(tmp_path):
    result = asyncio.run(drive(FakeVLLM(delay=0.005), [item() for _ in range(5)], tmp_path, concurrency=2,
                               slo={"ttft_s": 5, "e2e_s": 5}))
    summary = result["summary"]
    assert summary["successful_requests"] == 5 and summary["p95_queue_wait_s"] is None
    assert summary["slo"]["good_requests"] == 5 and summary["completed_output_tokens"] == 30
    assert result["metadata"]["server"]["service"] == "openai-compatible"
    assert result["metadata"]["server"]["models"][0]["max_model_len"] == 4096


def test_http_errors_overload_and_broken_streams_get_distinct_outcomes(tmp_path):
    def outcome(mode):
        return asyncio.run(drive(FakeVLLM(mode=mode), [item()], tmp_path))["requests"][0]

    bad = outcome("bad_request")
    assert bad["outcome"] == "failed" and bad["error_code"] == "400" and bad["http_status"] == 400
    busy = outcome("busy")
    assert busy["outcome"] == "rejected" and busy["error_code"] == "overloaded"
    cut = outcome("truncated")
    assert cut["outcome"] == "failed" and cut["error_code"] == "incomplete_stream"
    broken = outcome("stream_error")
    assert broken["outcome"] == "failed" and broken["error_code"] == "inference_error"


def test_client_cancellation_closes_the_connection_and_is_not_an_error(tmp_path):
    fake = FakeVLLM(chunks=200, delay=0.01)
    result = asyncio.run(drive(fake, [item(max_tokens=200, cancel_after_tokens=3)], tmp_path))
    record = result["requests"][0]
    assert record["outcome"] == "client_cancelled" and record["received_output_tokens"] == 3
    # Every request was cancelled by design, so there is nothing to count as an error: the rate is undefined.
    assert result["summary"]["error_rate"] is None and result["summary"]["client_cancelled_requests"] == 1
    assert result["summary"]["failed_requests"] == 0


def test_a_client_deadline_is_a_timeout(tmp_path):
    record = asyncio.run(drive(FakeVLLM(chunks=500, delay=0.02), [item(max_tokens=500, timeout_s=0.1)],
                               tmp_path))["requests"][0]
    assert record["outcome"] == "timed_out"


def test_a_server_with_no_models_is_not_ready(tmp_path):
    with pytest.raises(RuntimeError, match="not ready"):
        asyncio.run(drive(FakeVLLM(models=()), [item()], tmp_path))


def test_the_api_selection_is_validated():
    with pytest.raises(ValueError, match="model-name"):
        load.validate_config(args(model_name=None))
    with pytest.raises(ValueError, match="api must be"):
        load.validate_config(args(api="grpc"))
    load.validate_config(args(api="tensorpilot", model_name=None))


def test_both_clients_produce_records_the_summary_and_comparison_tools_can_read(tmp_path):
    """Schema parity: a comparison across engines is only meaningful if both report the same fields."""
    async def both():
        plan_path = tmp_path / "plan.json"
        plan_path.write_text(json.dumps(validate_workload([item()])))
        async with TestServer(FakeVLLM().app) as vllm:
            theirs = await load.run(args(url=str(vllm.make_url("")), workload_file=str(plan_path)))
        async with TestServer(create_app(FakeBackend, queue_size=4, warmup=False)) as ours:
            mine = await load.run(args(url=str(ours.make_url("")), workload_file=str(plan_path), api="tensorpilot"))
        return mine["requests"][0], theirs["requests"][0], mine["summary"], theirs["summary"]

    mine, theirs, mine_summary, theirs_summary = asyncio.run(both())
    needed = {"index", "ok", "outcome", "class", "max_tokens", "client_ttft_s", "client_elapsed_s",
              "client_token_times_s", "client_inter_token_s", "client_tpot_s", "received_output_tokens",
              "dispatch_offset_s", "client_last_token_s", "server"}
    assert needed <= set(mine) and needed <= set(theirs)
    assert set(mine_summary) == set(theirs_summary)
    assert {"generated_tokens"} <= set(mine["server"]) and {"generated_tokens"} <= set(theirs["server"])
