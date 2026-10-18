from tensorpilot.bench import summarize


def test_summary():
    result = summarize([
        {"ok": True, "elapsed_s": 1, "generated_tokens": 10},
        {"ok": True, "elapsed_s": 2, "generated_tokens": 20},
        {"ok": False, "elapsed_s": 1, "generated_tokens": 0},
    ])
    assert result["successful_requests"] == 2
    assert result["failed_requests"] == 1
    assert result["total_generated_tokens"] == 30
    assert result["aggregate_output_tokens_per_s"] == 7.5
    assert result["median_request_latency_s"] == 1.5
