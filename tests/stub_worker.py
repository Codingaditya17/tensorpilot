"""A real worker process for fault tests: the production server app around a fast fake backend.

Run as ``python stub_worker.py --port N``. Authentication comes from TENSORPILOT_API_KEY, as on a Pod.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from aiohttp import web  # noqa: E402
from tensorpilot.backend import Token  # noqa: E402
from tensorpilot.server import create_app  # noqa: E402


class QuickBackend:
    provenance = {"backend": "stub"}
    max_prompt_tokens = 2048

    def stream(self, prompt, max_tokens):
        for index in range(max_tokens):
            time.sleep(0.002)
            yield Token("x", index, True)

    def memory(self):
        return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    app = create_app(QuickBackend, queue_size=64, max_tokens=64, warmup=False,
                     api_key=os.environ.get("TENSORPILOT_API_KEY"))
    web.run_app(app, host="127.0.0.1", port=args.port, print=None, access_log=None)


if __name__ == "__main__":
    main()
