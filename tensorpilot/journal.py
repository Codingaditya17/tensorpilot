"""Append-only request journals and explicit partial-run recovery."""
import argparse
import json
import math
import os
from pathlib import Path

from .reporting import write_json


class Journal:
    def __init__(self, path, metadata, *, overwrite=False, sync=False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("w" if overwrite else "x", encoding="utf-8")
        self.sync = sync
        self.count = 0
        try:
            self._write({"type": "metadata", "schema_version": 1, "metadata": metadata})
        except BaseException:
            self.stream.close()
            raise

    def _write(self, value):
        self.stream.write(json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":")) + "\n")
        self.stream.flush()
        if self.sync:
            os.fsync(self.stream.fileno())

    def record(self, record):
        self._write({"type": "request", "record": record})
        self.count += 1

    def finish(self, status, wall_s):
        try:
            self._write({"type": "end", "status": status, "wall_s": wall_s, "record_count": self.count})
            os.fsync(self.stream.fileno())
        finally:
            self.stream.close()


def recover(path):
    # Import lazily: load uses Journal for writing, recovery uses its summaries.
    from .load import class_summaries, summarize
    metadata, footer = None, None
    records, indices = [], set()
    truncated = False
    with Path(path).open("rb") as stream:
        line = stream.readline()
        while line:
            following = stream.readline()
            try:
                entry = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                if following or line.endswith(b"\n") or footer is not None:
                    raise ValueError("corrupt journal record before the end of file") from exc
                truncated = True
                break
            if not isinstance(entry, dict):
                raise ValueError("journal entries must be objects")
            kind = entry.get("type")
            if kind == "metadata" and metadata is None and not records:
                metadata = entry["metadata"]
            elif kind == "request" and metadata is not None and footer is None:
                record = entry["record"]
                index = record["index"]
                if type(index) is not int or index < 0 or index in indices:
                    raise ValueError("duplicate or invalid request index in journal")
                records.append(record)
                if len(records) > 100000:
                    raise ValueError("recovery request count exceeds the 100000-record safety limit")
                indices.add(index)
            elif kind == "end" and metadata is not None and footer is None:
                footer = entry
            else:
                raise ValueError("invalid journal entry order")
            line = following
    if metadata is None or not records:
        raise ValueError("journal contains no recoverable request records")
    if footer is not None and footer["record_count"] != len(records):
        raise ValueError("journal footer count does not match request records")
    if footer is not None and footer["status"] not in ("completed", "interrupted", "failed"):
        raise ValueError("invalid journal completion status")
    last_completion = max(record["dispatch_offset_s"] + record["client_elapsed_s"] for record in records)
    wall_s = footer["wall_s"] if footer else last_completion
    if not isinstance(wall_s, (int, float)) or not math.isfinite(wall_s) or wall_s <= 0:
        raise ValueError("journal has no valid observation window")
    recovery = {"journal_path": str(path), "truncated_final_line": truncated,
                "run_status": footer["status"] if footer else "interrupted",
                "observation_window": "full recorded run" if footer else "through last recorded completion",
                "inflight_requests_at_interruption": "unknown; never counted as successful"}
    return {"schema_version": 3, "metadata": {**metadata, "recovery": recovery},
            "summary": summarize(records, wall_s), "by_class": class_summaries(records, wall_s),
            "requests": sorted(records, key=lambda record: record["index"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("journal")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        report = recover(args.journal)
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(1, f"Recovery failed: {exc}\n")
    write_json(args.output, report)
    print(json.dumps(report["summary"], indent=2))
    print(f"Recovered {len(report['requests'])} records: {args.output}")


if __name__ == "__main__":
    main()
