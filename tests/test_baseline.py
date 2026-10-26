"""The v0.9.1-4 RTX 4090 baseline is immutable evidence: its checksum and its summary are enforced."""
import hashlib
import json
import tarfile
from pathlib import Path

BASE = Path(__file__).resolve().parents[1] / "benchmarks" / "baselines" / "v0.9.1-4-rtx4090"
PUBLISHED_SHA256 = "3424f781ff38eea57bbfb11f448399b61f9cc5c08f7ad8befb171aa04cc30bd3"


def test_the_archive_is_byte_for_byte_the_one_exported_from_the_pod():
    assert hashlib.sha256((BASE / "evidence.tar.gz").read_bytes()).hexdigest() == PUBLISHED_SHA256
    assert (BASE / "evidence.tar.gz.sha256").read_text().split()[0] == PUBLISHED_SHA256


def test_the_summary_agrees_with_the_raw_files_it_was_generated_from():
    summary = json.loads((BASE / "summary.json").read_text())
    assert summary["evidence_sha256"] == PUBLISHED_SHA256
    with tarfile.open(BASE / "evidence.tar.gz") as tar:
        reports = {}
        for member in tar:
            if "/bench/" in member.name and member.name.endswith(".json") and not member.name.endswith(
                    ("manifest.json", ".workload.json")):
                data = json.load(tar.extractfile(member))
                reports[data["experiment"]["id"]] = data["summary"]
    assert set(reports) == set(summary["experiments"])
    for name, recorded in summary["experiments"].items():
        raw = reports[name]
        assert recorded["successful"] == raw["successful_requests"] and recorded["failed"] == raw["failed_requests"]
        assert recorded["output_tokens_per_s_wall"] == round(raw["completed_output_tokens_per_s"], 2)
    assert sum(e["successful"] for e in summary["experiments"].values()) == 20
    assert summary["gateway_smoke"] == {"successful": 5, "failed": 0}


def test_the_baseline_states_what_it_does_not_cover():
    summary = json.loads((BASE / "summary.json").read_text())
    assert "batching" in summary["not_covered"] and "overload and rejection" in summary["not_covered"]
    assert summary["status"].startswith("IMMUTABLE")
