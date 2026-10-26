"""The v1.0.0-rc1 quick-session baseline is immutable evidence: checksums, summary and the findings the docs rely on are enforced."""
import hashlib
import json
import tarfile
from pathlib import Path

import pytest

from tensorpilot import acceptance, report_v10

BASE = Path(__file__).resolve().parents[1] / "benchmarks" / "baselines" / "v1.0.0-rc1-quick-rtx4090"
PUBLISHED = {"serial": "2e357d4e9960483815247f925987b9d07443e6b3a712bed27a7b9bce9dc5e28d",
             "batch": "35d0035739e43021fe81971262ad42e0e3e205c866b7bbe753c6183a1d08d3ff"}


def test_the_archives_are_byte_for_byte_the_ones_exported_from_the_pod():
    for name, digest in PUBLISHED.items():
        assert hashlib.sha256((BASE / f"{name}.tar.gz").read_bytes()).hexdigest() == digest
        assert (BASE / f"{name}.tar.gz.sha256").read_text().split()[0] == digest


def test_the_archives_contain_no_credentials():
    for name in PUBLISHED:
        assert acceptance.scan_archive(BASE / f"{name}.tar.gz") == []


@pytest.fixture(scope="module")
def extracted(tmp_path_factory):
    root = tmp_path_factory.mktemp("quick")
    for name in PUBLISHED:
        with tarfile.open(BASE / f"{name}.tar.gz") as tar:
            tar.extractall(root, filter="data")
    return report_v10.load_session(root)


def test_the_summary_is_what_the_archives_say(extracted):
    summary = json.loads((BASE / "summary.json").read_text())
    assert summary["archives"] == PUBLISHED
    for key, recorded in summary["experiments"].items():
        kind, name = key.split("/")
        stage = next(ev for stage_name, ev in extracted.items() if stage_name.endswith(kind))
        raw = stage["reports"][name][0]["summary"]
        assert recorded["successful"] == raw["successful_requests"]
        assert recorded["output_tokens_per_s_wall"] == round(raw["completed_output_tokens_per_s"], 2)
    assert summary["invariance"] == {"comparable": 32, "exact_matches": 31, "earliest_divergence": 7,
                                     "diverged_request_index": 21}


def test_the_findings_quoted_in_the_docs_follow_from_the_evidence(extracted):
    """The document says: ~20 ms per decode step at every width, GPU utilization under 25%, prefill 22 ms per chunk."""
    rows = {r["experiment"]: r for r in report_v10.profile_rows(extracted["0.5b-quick-batch"])}
    steps = [rows[name]["ms_per_decode_step"] for name in ("quick-c1", "quick-c8", "quick-c32", "quick-c128")]
    assert max(steps) - min(steps) < 1.5 and 19 < min(steps) and max(steps) < 21.5
    widths = [rows[name]["tokens_per_step"] for name in ("quick-c1", "quick-c8", "quick-c32", "quick-c128")]
    assert widths == sorted(widths) and widths[-1] > 20 * widths[0]
    assert all(row["gpu_util_mean"] < 25 for row in rows.values())
    assert rows["quick-c128"]["prefill_share"] > 0.35 and 21 < rows["quick-c128"]["ms_per_prefill_chunk"] < 24


def test_the_quick_gate_verdict_and_speedup_are_as_recorded():
    verdict = json.loads((BASE / "quick-verdict.json").read_text())
    assert verdict["passed"] is True
    assert "5.42x" in next(c["detail"] for c in verdict["checks"] if c["name"] == "batched_c8_beats_serial_c8")


def test_the_baseline_states_what_it_does_not_cover():
    summary = json.loads((BASE / "summary.json").read_text())
    assert "logits for the diverged request" in summary["not_covered"] and summary["status"].startswith("IMMUTABLE")
