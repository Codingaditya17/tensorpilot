"""A stand-in OpenAI-compatible server for stage-runner tests. It insists on the bearer key from VLLM_API_KEY, as vLLM does."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from aiohttp import web  # noqa: E402
from test_load_openai import FakeVLLM  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    key = os.environ.get("VLLM_API_KEY")
    fake = FakeVLLM(chunks=6, delay=0.002)

    @web.middleware
    async def authenticate(request, handler):
        if key and request.headers.get("Authorization") != f"Bearer {key}":
            return web.json_response({"error": {"message": "unauthorized"}}, status=401)
        return await handler(request)

    fake.app.middlewares.append(authenticate)
    web.run_app(fake.app, host="127.0.0.1", port=args.port, print=None, access_log=None)


if __name__ == "__main__":
    main()
