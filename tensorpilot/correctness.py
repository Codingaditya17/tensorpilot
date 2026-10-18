"""Record or verify deterministic token-ID references against a running service."""
import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import aiohttp

from .client import Client
from .reporting import write_json

CASES = (
    {"name": "short_fact", "prompt": "State that two plus two equals four.", "max_tokens": 12},
    {"name": "serving", "prompt": "Define bounded admission in one sentence.", "max_tokens": 20},
    {"name": "unicode", "prompt": "Repeat these symbols once: café λ.", "max_tokens": 16},
)
PINNED_FIELDS = ("artifact_sha256", "model_revision", "sampling", "prompt_format")


async def model_info(session, url):
    async with session.get(url.rstrip("/") + "/v1/model",
                           timeout=aiohttp.ClientTimeout(total=10)) as response:
        response.raise_for_status()
        return await response.json()


async def capture(client, case):
    token_ids, text, done = [], [], None
    async for event in client.stream(case["prompt"], max_tokens=case["max_tokens"]):
        if event["type"] == "token":
            token_ids.append(event["token_id"])
            if event["output"]:
                text.append(event["text"])
        elif event["type"] == "done":
            done = event
    if done is None:
        raise RuntimeError("generation returned no terminal event")
    return {**case, "token_ids_including_eos": token_ids, "output_text": "".join(text),
            "generated_tokens": done["generated_tokens"], "finish_reason": done["finish_reason"]}


async def record(url, cases=CASES):
    key = os.environ.get("TENSORPILOT_API_KEY")
    headers = {"Authorization": "Bearer " + key} if key else {}
    async with aiohttp.ClientSession(headers=headers) as session:
        server = await model_info(session, url)
    records = []
    async with Client(url, api_key=key) as client:
        for case in cases:
            first, second = await capture(client, case), await capture(client, case)
            if first["token_ids_including_eos"] != second["token_ids_including_eos"]:
                raise RuntimeError(f"nondeterministic token IDs for {case['name']}")
            records.append(first)
    return {"schema_version": 1, "measurement_kind": "deterministic_token_id_reference",
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "server": {field: server.get(field) for field in PINNED_FIELDS}, "cases": records}


async def verify(url, reference):
    key = os.environ.get("TENSORPILOT_API_KEY")
    headers = {"Authorization": "Bearer " + key} if key else {}
    async with aiohttp.ClientSession(headers=headers) as session:
        server = await model_info(session, url)
    mismatched_metadata = [field for field in PINNED_FIELDS
                           if server.get(field) != reference["server"].get(field)]
    if mismatched_metadata:
        raise ValueError("server does not match pinned reference: " + ", ".join(mismatched_metadata))
    results = []
    async with Client(url, api_key=key) as client:
        for expected in reference["cases"]:
            actual = await capture(client, {key: expected[key] for key in ("name", "prompt", "max_tokens")})
            exact = actual["token_ids_including_eos"] == expected["token_ids_including_eos"]
            results.append({"name": expected["name"], "exact_token_ids": exact,
                            "expected_count": len(expected["token_ids_including_eos"]),
                            "actual_count": len(actual["token_ids_including_eos"])})
    return {"passed": all(result["exact_token_ids"] for result in results),
            "execution_mode": server.get("execution_mode"), "results": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference")
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    try:
        if args.record:
            result = asyncio.run(record(args.url))
            write_json(args.reference, result)
        else:
            reference = json.loads(Path(args.reference).read_text())
            result = asyncio.run(verify(args.url, reference))
            if args.output:
                write_json(args.output, result)
    except (ValueError, OSError, RuntimeError, aiohttp.ClientError) as exc:
        parser.exit(1, f"Correctness check failed: {exc}\n")
    print(json.dumps(result, indent=2))
    if not args.record and not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
