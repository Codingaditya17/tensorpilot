import pytest

from tensorpilot.routing import WorkerRegistry


def health(*, ready=True, active=0, queued=0, reserved=0, budget=1000,
           model="model-a", artifact="sha-a", concurrency=2, queue_capacity=2):
    return {"ready": ready, "active_sequences": active, "queue_depth": queued,
            "execution_concurrency": concurrency, "queue_capacity": queue_capacity,
            "admission": {"estimated_kv_reserved_bytes": reserved,
                          "estimated_kv_budget_bytes": budget},
            "model": model, "artifact_sha256": artifact, "service_version": "0.7.0",
            "instance_id": "instance", "execution_mode": "adaptive"}


def registry(clock=lambda: 0):
    result = WorkerRegistry(stale_after_s=5, clock=clock)
    for worker_id, port in (("a", 8101), ("b", 8102)):
        result.register(worker_id, f"http://127.0.0.1:{port}")
        result.update(worker_id, health())
    return result


def test_capacity_aware_balances_in_flight_routes_and_releases_exactly():
    workers = registry()
    first = workers.select(100)
    second = workers.select(100)
    assert (first.worker_id, second.worker_id) == ("a", "b")
    assert first.reason == "lowest-load-then-memory"
    workers.release("a")
    workers.release("b")
    assert all(worker.in_flight_routes == 0 for worker in workers.workers.values())
    with pytest.raises(RuntimeError):
        workers.release("a")


def test_capacity_aware_tie_breaker_is_fair_for_sequential_requests():
    workers = registry()
    chosen = []
    for index in range(6):
        placement = workers.select(100, request_id=f"request-{index}")
        chosen.append(placement.worker_id)
        workers.release(placement.worker_id)
    assert chosen == ["a", "b", "a", "b", "a", "b"]
    assert len(workers.snapshot()["placement_history"]) == 6


def test_drain_stale_model_memory_and_capacity_are_hard_filters():
    now = [0.0]
    workers = registry(lambda: now[0])
    workers.set_draining("a", True)
    assert workers.select(100).worker_id == "b"
    workers.release("b")
    workers.set_draining("a", False)
    workers.update("a", health(model="other"))
    workers.update("b", health(reserved=950))
    assert workers.select(100, model="model-a", request_id="rejected") is None
    rejection = workers.snapshot()["placement_history"][-1]
    assert rejection["excluded"]["model_mismatch"] == 1
    assert rejection["excluded"]["memory_pressure"] == 1
    workers.update("a", health(artifact="other"))
    workers.update("b", health())
    assert workers.select(1, artifact_sha256="missing") is None
    assert workers.snapshot()["placement_history"][-1]["excluded"]["artifact_mismatch"] == 2
    workers.update("b", health())
    now[0] = 6
    assert workers.select(100) is None
    assert all(item["stale"] for item in workers.snapshot()["workers"])


def test_round_robin_and_busy_removal():
    workers = WorkerRegistry(policy="round-robin", clock=lambda: 0)
    for worker_id, port in (("a", 8101), ("b", 8102)):
        workers.register(worker_id, f"http://127.0.0.1:{port}")
        workers.update(worker_id, health())
    placements = [workers.select(1) for _ in range(2)]
    assert [placement.worker_id for placement in placements] == ["a", "b"]
    with pytest.raises(ValueError, match="in-flight"):
        workers.remove("a")
    for placement in placements:
        workers.release(placement.worker_id)
    assert workers.remove("a").worker_id == "a"


@pytest.mark.parametrize("worker_id,url", [
    ("", "http://127.0.0.1:1"), ("bad id", "http://127.0.0.1:1"),
    ("a", "https://127.0.0.1:1"), ("a", "http://example.com:1"),
    ("a", "http://user:pass@127.0.0.1:1"), ("a", "http://127.0.0.1/path"),
])
def test_worker_identity_validation(worker_id, url):
    with pytest.raises(ValueError):
        WorkerRegistry.validate_identity(worker_id, url, {"127.0.0.1"})


def test_invalid_health_does_not_silently_coerce_values():
    workers = WorkerRegistry(clock=lambda: 0)
    workers.register("a", "http://127.0.0.1:1")
    with pytest.raises(ValueError):
        workers.update("a", health(active=True))


def test_instance_change_and_drain_are_in_bounded_lifecycle_history():
    workers = registry()
    changed = health()
    changed["instance_id"] = "replacement"
    workers.update("a", changed)
    workers.set_draining("a", True)
    events = [event["event"] for event in workers.snapshot()["lifecycle_history"]]
    assert "worker_instance_changed" in events
    assert events[-1] == "worker_drain_changed"


def test_power_of_two_uses_compatible_index_and_preserves_filters():
    workers = WorkerRegistry(policy="power-of-two", clock=lambda: 0)
    for index in range(100):
        worker_id = f"worker-{index}"
        workers.register(worker_id, f"http://127.0.0.1:{9000 + index}")
        workers.update(worker_id, health(model="wanted" if index < 50 else "other"))
    selected = []
    for _ in range(100):
        placement = workers.select(1, model="wanted", artifact_sha256="sha-a")
        selected.append(placement.worker_id)
        workers.release(placement.worker_id)
    assert len(set(selected)) == 50
    assert all(int(worker_id.split("-")[1]) < 50 for worker_id in selected)
    for index in range(50):
        workers.set_draining(f"worker-{index}", True)
    assert workers.select(1, model="wanted", artifact_sha256="sha-a") is None
    rejection = workers.snapshot()["placement_history"][-1]
    assert rejection["excluded"]["draining"] == 50


def test_worker_pages_are_ordered_and_detect_membership_changes():
    workers = WorkerRegistry(clock=lambda: 0)
    for worker_id in ("c", "a", "b"):
        workers.register(worker_id, f"http://127.0.0.1:{9000 + len(workers.workers)}")
        workers.update(worker_id, health())
    first = workers.page(limit=2)
    assert [worker["worker_id"] for worker in first["workers"]] == ["a", "b"]
    assert first["next_cursor"]
    second = workers.page(limit=2, cursor=first["next_cursor"])
    assert [worker["worker_id"] for worker in second["workers"]] == ["c"]
    workers.register("d", "http://127.0.0.1:9004")
    with pytest.raises(RuntimeError, match="stale"):
        workers.page(limit=2, cursor=first["next_cursor"])
    with pytest.raises(ValueError, match="cursor"):
        workers.page(cursor="broken")


def test_summary_is_bounded_and_excludes_worker_and_audit_records():
    workers = registry()
    summary = workers.summary()
    assert summary["worker_counts"]["registered"] == 2
    assert summary["worker_counts"]["available"] == 2
    assert summary["uniform_model"] == "model-a"
    assert summary["uniform_artifact_sha256"] == "sha-a"
    assert "workers" not in summary
    assert "placement_history" not in summary
