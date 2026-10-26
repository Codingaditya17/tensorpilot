import json

from tensorpilot import smoke_check

WORKER_401 = json.dumps({"error": {"code": "unauthorized", "message": "A valid bearer token is required"},
                         "request_id": "r1"})


def report(records):
    ok = sum(1 for record in records if record["ok"])
    return {"summary": {"successful_requests": ok, "failed_requests": len(records) - ok}, "requests": records}


def failure(index, *, status=401, code="unauthorized", error=WORKER_401):
    return {"index": index, "ok": False, "http_status": status, "error_code": code, "error": error,
            "request_id": f"id{index}", "client_elapsed_s": .005}


def test_the_pod_failure_is_grouped_and_explained(tmp_path, capsys):
    path = tmp_path / "smoke.json"
    path.write_text(json.dumps(report([failure(i) for i in range(5)])))
    worker_log = tmp_path / "worker.log"
    worker_log.write_text("\n".join(f'{{"event": "http_request", "status": 401, "n": {i}}}' for i in range(9)))
    assert smoke_check.main([str(path), "--expect", "5", "--log", str(worker_log)]) == 1
    err = capsys.readouterr().err
    assert "0 of 5 streams validated" in err
    assert "5 x http_status=401 error_code=unauthorized" in err
    assert "first failure: index=0 request_id=id0" in err and "A valid bearer token is required" in err
    assert "likely cause: a service behind the gateway rejected the request with 401" in err
    assert "last 5 line(s) of worker.log" in err and '"n": 8' in err and '"n": 3' not in err


def test_gateway_fault_code_points_at_the_data_plane_credential(tmp_path, capsys):
    path = tmp_path / "smoke.json"
    path.write_text(json.dumps(report([failure(0, status=502, code="worker_unauthorized",
                                               error="Worker rejected the data plane's credential")])))
    assert smoke_check.main([str(path), "--expect", "1"]) == 1
    assert "set TENSORPILOT_WORKER_API_KEY on the data plane" in capsys.readouterr().err


def test_client_token_rejection_is_told_apart_from_a_worker_rejection(tmp_path, capsys):
    message = json.dumps({"error": {"code": "unauthorized", "message": "A valid data-plane token is required"}})
    path = tmp_path / "smoke.json"
    path.write_text(json.dumps(report([failure(0, error=message)])))
    smoke_check.main([str(path), "--expect", "1"])
    assert "must use TENSORPILOT_DATA_API_KEY" in capsys.readouterr().err


def test_credentials_are_redacted_from_messages_and_logs(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("TENSORPILOT_API_KEY", "worker-key-value-1234")
    path = tmp_path / "smoke.json"
    path.write_text(json.dumps(report([failure(0, status=500, code="boom",
                                               error="bad token worker-key-value-1234 hf_abcdefghijklmnopqrst")])))
    log = tmp_path / "data.log"
    log.write_text("Authorization: Bearer worker-key-value-1234\n")
    smoke_check.main([str(path), "--expect", "1", "--log", str(log)])
    err = capsys.readouterr().err
    assert "worker-key-value-1234" not in err and "hf_abcdefghijklmnopqrst" not in err and "[REDACTED]" in err


def test_success_requires_every_expected_stream(tmp_path, capsys):
    ok = {"index": 0, "ok": True}
    good, short, mixed = tmp_path / "g.json", tmp_path / "s.json", tmp_path / "m.json"
    good.write_text(json.dumps(report([ok] * 5)))
    short.write_text(json.dumps(report([ok] * 4)))
    mixed.write_text(json.dumps(report([ok] * 4 + [failure(4, status=503, code="overloaded", error="busy")])))
    assert smoke_check.main([str(good), "--expect", "5"]) == 0
    assert "5 of 5 streams validated" in capsys.readouterr().out
    assert smoke_check.main([str(short), "--expect", "5"]) == 1
    assert smoke_check.main([str(mixed), "--expect", "5"]) == 1


def test_a_missing_or_corrupt_report_fails_with_an_explanation(tmp_path, capsys):
    assert smoke_check.main([str(tmp_path / "absent.json"), "--expect", "5"]) == 1
    assert "no readable report" in capsys.readouterr().err
    (tmp_path / "bad.json").write_text("{")
    assert smoke_check.main([str(tmp_path / "bad.json"), "--expect", "5"]) == 1
