"""Shared streaming contract checks for application clients and measurements."""
import json
import math


class StreamProtocolError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def decode_event(line):
    try:
        return json.loads(line)
    except (ValueError, UnicodeDecodeError) as exc:
        raise StreamProtocolError("invalid_stream", "Invalid JSON stream event") from exc


class StreamValidator:
    def __init__(self, request_id, max_tokens):
        self.request_id = request_id
        self.max_tokens = max_tokens
        self.output_tokens = 0
        self.finished = False

    def feed(self, event):
        def invalid(message):
            raise StreamProtocolError("invalid_stream", message)

        if self.finished:
            invalid("Event received after terminal event")
        if not isinstance(event, dict):
            invalid("Stream event must be an object")
        if self.request_id is not None and event.get("request_id") != self.request_id:
            invalid("Response request IDs disagree")
        kind = event.get("type")
        if kind == "token":
            if not isinstance(event.get("text"), str) or type(event.get("token_id")) is not int or event["token_id"] < 0:
                invalid("Invalid token payload")
            if type(event.get("output")) is not bool:
                invalid("Token output flag must be boolean")
            self.output_tokens += int(event["output"])
            if self.output_tokens > self.max_tokens:
                invalid("Response exceeds requested output budget")
        elif kind == "done":
            count = event.get("generated_tokens")
            if type(count) is not int or count < 0:
                invalid("Invalid completion token count")
            if count != self.output_tokens:
                raise StreamProtocolError("token_count_mismatch", "Client/server output token counts disagree")
            for field in ("queue_wait_s", "server_elapsed_s", "ttft_s"):
                value = event.get(field)
                if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
                    invalid("Invalid completion timing")
            self.finished = True
        elif kind == "error":
            if not isinstance(event.get("code"), str) or not isinstance(event.get("error"), str):
                invalid("Invalid error payload")
            self.finished = True
        else:
            invalid("Unknown stream event type")
        return kind

    def eof(self):
        if not self.finished:
            raise StreamProtocolError("incomplete_stream", "Stream ended without a terminal event")
