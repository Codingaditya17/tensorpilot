"""The runtime memory gate: it reads measured numbers and refuses when the real margin or the pool falls short."""
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from tensorpilot import memory_gate

GIB = 1024**3
TOTAL = 25252724736


def health(*, peak_gib=18.0, reserved_gib=None, rows=48, oom=0, fraction=0.9, total=TOTAL, pool=True):
    memory = {"cuda_peak_reserved_bytes": int(peak_gib * GIB), "cuda_oom_events": oom}
    if reserved_gib is not None:
        memory = {"cuda_reserved_bytes": int(reserved_gib * GIB), "cuda_oom_events": oom}
    value = {"last_measured_memory": memory, "gpu": {"total_memory_bytes": total}, "cuda_memory_fraction": fraction}
    if pool:
        value["kv_pool"] = {"capacity_rows": rows}
    return value


def names(result, passed):
    return {c["name"] for c in result["checks"] if c["passed"] is passed}


def test_a_comfortable_margin_and_a_full_pool_pass():
    result = memory_gate.evaluate(health(peak_gib=18.0), min_margin_gib=1.5, expect_rows=48)
    assert result["ok"] and names(result, True) == {"measured_margin_inside_the_allocator_cap", "pool_holds_the_requested_slots"}
    check = result["checks"][0]
    assert check["cap_gib"] == pytest.approx(21.17, abs=0.01) and check["margin_gib"] == pytest.approx(3.17, abs=0.01)


def test_a_thin_measured_margin_fails_even_if_the_plan_promised_more():
    result = memory_gate.evaluate(health(peak_gib=20.0), min_margin_gib=1.5, expect_rows=48)  # 1.17 GiB free
    assert not result["ok"] and "measured_margin_inside_the_allocator_cap" in names(result, False)
    assert "1.17 GiB free" in result["checks"][0]["detail"]


def test_the_peak_is_used_so_a_loading_transient_counts():
    result = memory_gate.evaluate(health(peak_gib=20.5, reserved_gib=15.0), min_margin_gib=1.5)
    assert result["ok"], "with only the current reservation reported the gate falls back to it"
    both = health(peak_gib=20.5)
    both["last_measured_memory"]["cuda_reserved_bytes"] = int(15.0 * GIB)
    assert not memory_gate.evaluate(both, min_margin_gib=1.5)["ok"], "the higher peak wins over the lower current value"


def test_a_pool_smaller_than_requested_or_missing_fails():
    assert "pool_holds_the_requested_slots" in names(memory_gate.evaluate(health(rows=20), min_margin_gib=1.5, expect_rows=48), False)
    result = memory_gate.evaluate(health(pool=False), min_margin_gib=1.5, expect_rows=48)
    assert not result["ok"] and "no KV pool reported" in result["checks"][1]["detail"]
    assert memory_gate.evaluate(health(pool=False), min_margin_gib=1.5)["ok"], "a serial stage requests no pool"


def test_an_oom_event_fails_the_gate():
    result = memory_gate.evaluate(health(oom=1), min_margin_gib=1.5)
    assert not result["ok"] and "no_out_of_memory_events" in names(result, False)


@pytest.mark.parametrize("missing", ["total", "fraction", "peak"])
def test_missing_information_fails_closed_because_the_margin_cannot_be_verified(missing):
    value = health()
    if missing == "total":
        value["gpu"] = {}
    elif missing == "fraction":
        value["cuda_memory_fraction"] = None
    else:
        value["last_measured_memory"] = {}
    result = memory_gate.evaluate(value, min_margin_gib=1.5)
    assert not result["ok"] and result["checks"][0]["name"] == "memory_information_present"


def serve(document):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps(document).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def test_the_command_line_reads_health_writes_the_result_and_sets_the_exit_code(tmp_path, monkeypatch, capsys):
    good, bad = serve(health(peak_gib=18)), serve(health(peak_gib=20.4))
    out = tmp_path / "gate.json"
    try:
        monkeypatch.setattr(sys, "argv", ["gate", "--url", f"http://127.0.0.1:{good.server_port}", "--expect-rows", "48",
                                          "--output", str(out)])
        assert memory_gate.main() == 0 and json.loads(out.read_text())["ok"] is True
        monkeypatch.setattr(sys, "argv", ["gate", "--url", f"http://127.0.0.1:{bad.server_port}"])
        assert memory_gate.main() == 1 and "[FAIL] measured_margin" in capsys.readouterr().out
    finally:
        good.shutdown()
        bad.shutdown()


def test_a_non_http_url_is_refused_before_anything_is_opened(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["gate", "--url", "file:///etc/passwd"])
    with pytest.raises(SystemExit) as exit_info:
        memory_gate.main()
    assert exit_info.value.code == 2 and "http or https" in capsys.readouterr().err
