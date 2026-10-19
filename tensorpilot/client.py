"""Async integration client for TensorPilot's versioned streaming API."""
import math

import aiohttp

from .protocol import StreamProtocolError, StreamValidator, decode_event


class TensorPilotError(Exception):
    def __init__(self, code, message, *, request_id=None, status=None):
        super().__init__(message)
        self.code = code
        self.request_id = request_id
        self.status = status


class Client:
    """Use as an async context manager; never retries partially generated work."""
    def __init__(self, url="http://127.0.0.1:8080", *, api_key=None):
        self.url = url.rstrip("/")
        self.headers = {"Authorization": "Bearer " + api_key} if api_key else {}
        self.session = None

    async def __aenter__(self):
        if self.session is not None:
            raise RuntimeError("Client context is already open")
        self.session = aiohttp.ClientSession(headers=self.headers)
        return self

    async def __aexit__(self, *exc):
        await self.session.close()
        self.session = None

    async def stream(self, prompt, *, max_tokens=64, timeout_s=60):
        if self.session is None:
            raise RuntimeError("Use Client inside 'async with'")
        if type(timeout_s) not in (int, float) or not math.isfinite(timeout_s) or not 0 < timeout_s <= 120:
            raise TensorPilotError("invalid_request", "timeout_s must be finite and in (0, 120]")
        if type(max_tokens) is not int or not 1 <= max_tokens <= 512:
            raise TensorPilotError("invalid_request", "max_tokens must be an integer in 1..512")
        request_id = None
        try:
            async with self.session.post(
                self.url + "/v1/generate",
                json={"prompt": prompt, "max_tokens": max_tokens, "timeout_s": timeout_s},
                timeout=aiohttp.ClientTimeout(total=timeout_s + 5),
            ) as response:
                request_id = response.headers.get("X-Request-ID")
                if response.status != 200:
                    try:
                        body = await response.json()
                        code, message = body["error"]["code"], body["error"]["message"]
                    except (ValueError, KeyError, TypeError, aiohttp.ContentTypeError):
                        code, message = "http_error", f"HTTP {response.status}"
                    raise TensorPilotError(code, message, request_id=request_id, status=response.status)
                validator = StreamValidator(request_id, max_tokens)
                async for line in response.content:
                    event = decode_event(line)
                    kind = validator.feed(event)
                    if kind == "error":
                        raise TensorPilotError(event.get("code", "inference_error"), event.get("error", "Inference failed"),
                                               request_id=request_id, status=200)
                    yield event
                    if kind == "done":
                        return
                validator.eof()
        except StreamProtocolError as exc:
            raise TensorPilotError(exc.code, str(exc), request_id=request_id, status=200) from exc
        except (aiohttp.ClientError, TimeoutError) as exc:
            raise TensorPilotError("transport_error", "Connection failed or timed out", request_id=request_id) from exc
