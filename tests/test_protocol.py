import asyncio
import json

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
import pytest

from tensorpilot.client import Client, TensorPilotError
from tensorpilot.protocol import StreamProtocolError, StreamValidator, decode_event


def token(**kwargs):
    return {"type": "token", "text": "x", "token_id": 1, "output": True, "request_id": "test", **kwargs}


def test_eos_does_not_increase_output_count_and_terminal_matches():
    validator = StreamValidator("test", 1)
    validator.feed(token())
    validator.feed(token(output=False, text=""))
    validator.feed({"type": "done", "request_id": "test", "generated_tokens": 1})
    validator.eof()
    assert validator.output_tokens == 1
    with pytest.raises(StreamProtocolError):
        validator.feed(token())


@pytest.mark.parametrize("event", [[], token(request_id="other"), token(token_id=True),
                                  token(output=1), token(text=None), {"type": "unknown"}])
def test_invalid_events_are_rejected(event):
    with pytest.raises(StreamProtocolError, match="."):
        StreamValidator("test", 1).feed(event)


def test_budget_and_count_mismatches_never_become_success():
    validator = StreamValidator(None, 1)
    validator.feed(token())
    with pytest.raises(StreamProtocolError, match="budget"):
        validator.feed(token())
    validator = StreamValidator(None, 2)
    validator.feed(token())
    with pytest.raises(StreamProtocolError) as failure:
        validator.feed({"type": "done", "generated_tokens": 2})
    assert failure.value.code == "token_count_mismatch"
    with pytest.raises(StreamProtocolError):
        decode_event(b'{broken')


def test_nonfinite_timings_are_rejected():
    with pytest.raises(StreamProtocolError, match="timing"):
        StreamValidator(None, 1).feed({"type": "done", "generated_tokens": 0, "ttft_s": float("nan")})


def test_sdk_detects_request_id_mismatch():
    async def scenario():
        app = web.Application()

        async def mismatched(request):
            return web.Response(text=json.dumps(token(request_id="wrong")) + "\n",
                                headers={"X-Request-ID": "expected"})

        app.router.add_post("/v1/generate", mismatched)
        async with TestClient(TestServer(app)) as http:
            async with Client(str(http.make_url("/"))) as client:
                with pytest.raises(TensorPilotError) as failure:
                    _ = [event async for event in client.stream("hi", max_tokens=1)]
                assert failure.value.code == "invalid_stream"
                assert failure.value.request_id == "expected"
    asyncio.run(scenario())
