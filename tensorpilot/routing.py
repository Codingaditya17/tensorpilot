"""Worker registry and deterministic cluster placement policies."""
from dataclasses import asdict, dataclass, field
from collections import defaultdict, deque
import base64
import json
import math
import re
import time
from urllib.parse import urlparse


WORKER_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


@dataclass
class Worker:
    worker_id: str
    url: str
    registered_at: float
    last_seen: float
    ready: bool = False
    draining: bool = False
    active_sequences: int = 0
    queue_depth: int = 0
    execution_concurrency: int = 1
    queue_capacity: int = 8
    estimated_kv_reserved_bytes: int = 0
    estimated_kv_budget_bytes: int = 0
    model: str | None = None
    artifact_sha256: str | None = None
    service_version: str | None = None
    instance_id: str | None = None
    failures: int = 0
    in_flight_routes: int = 0
    last_error: str | None = None
    assigned_total: int = 0
    metadata: dict = field(default_factory=dict)

    def snapshot(self, now, stale_after_s):
        value = asdict(self)
        value["heartbeat_age_s"] = max(0.0, now - self.last_seen)
        value["stale"] = value["heartbeat_age_s"] > stale_after_s
        value["available_slots"] = self.available_slots
        value["estimated_memory_pressure"] = self.memory_pressure
        return value

    @property
    def available_slots(self):
        total = self.execution_concurrency + self.queue_capacity
        # Routed requests become the worker's active/queued requests. Heartbeats
        # observe that same work later, so summing both would double-charge it.
        used = max(self.active_sequences + self.queue_depth, self.in_flight_routes)
        return max(0, total - used)

    @property
    def accounted_load(self):
        return max(self.active_sequences + self.queue_depth, self.in_flight_routes)

    @property
    def memory_pressure(self):
        if self.estimated_kv_budget_bytes <= 0:
            return 1.0
        return self.estimated_kv_reserved_bytes / self.estimated_kv_budget_bytes


@dataclass(frozen=True)
class Placement:
    worker_id: str
    url: str
    reason: str
    policy: str
    load_ratio: float
    estimated_memory_pressure: float

    def as_dict(self):
        return asdict(self)


class WorkerRegistry:
    POLICIES = ("capacity-aware", "round-robin", "power-of-two")

    def __init__(self, *, stale_after_s=5.0, policy="capacity-aware", clock=time.monotonic):
        if not math.isfinite(stale_after_s) or stale_after_s <= 0:
            raise ValueError("stale_after_s must be finite and positive")
        if policy not in self.POLICIES:
            raise ValueError("unknown routing policy")
        self.stale_after_s = stale_after_s
        self.policy = policy
        self.clock = clock
        self.workers = {}
        self.round_robin_cursor = 0
        self.placements = 0
        self.history = deque(maxlen=100)
        self.lifecycle = deque(maxlen=100)
        self.compatible = defaultdict(list)
        self.compatibility_key = {}
        self.cursors = defaultdict(int)
        self.membership_revision = 0

    @staticmethod
    def validate_identity(worker_id, url, allowed_hosts):
        if not isinstance(worker_id, str) or not WORKER_ID.fullmatch(worker_id):
            raise ValueError("worker_id must match [A-Za-z0-9_.-] and contain 1..64 characters")
        parsed = urlparse(url)
        if parsed.scheme != "http" or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("worker URL must be plain HTTP without credentials")
        if parsed.hostname not in allowed_hosts:
            raise ValueError("worker host is not allowed")
        if parsed.path not in ("", "/") or parsed.query or parsed.fragment:
            raise ValueError("worker URL must not contain a path, query or fragment")
        if parsed.port is None:
            raise ValueError("worker URL requires an explicit port")
        return url.rstrip("/")

    def register(self, worker_id, url):
        now = self.clock()
        existing = self.workers.get(worker_id)
        if existing and existing.url != url:
            raise ValueError("worker_id is already registered with a different URL")
        if existing:
            return existing
        worker = Worker(worker_id, url, now, now)
        self.workers[worker_id] = worker
        self.membership_revision += 1
        self.lifecycle.append({"at_monotonic_s": now, "event": "worker_registered",
                               "worker_id": worker_id, "url": url})
        return worker

    def update(self, worker_id, health):
        worker = self.workers[worker_id]
        previous_instance = worker.instance_id
        previous_compatibility = self.compatibility_key.get(worker_id)
        admission = health.get("admission") or {}
        worker.ready = health.get("ready") is True
        worker.active_sequences = _nonnegative_int(health.get("active_sequences", 0), "active_sequences")
        worker.queue_depth = _nonnegative_int(health.get("queue_depth", 0), "queue_depth")
        worker.execution_concurrency = _positive_int(health.get("execution_concurrency", 1),
                                                     "execution_concurrency")
        worker.queue_capacity = _positive_int(health.get("queue_capacity", 8), "queue_capacity")
        worker.estimated_kv_reserved_bytes = _nonnegative_int(
            admission.get("estimated_kv_reserved_bytes", 0), "estimated_kv_reserved_bytes")
        worker.estimated_kv_budget_bytes = _positive_int(
            admission.get("estimated_kv_budget_bytes", 1), "estimated_kv_budget_bytes")
        worker.model = _optional_string(health.get("model"), "model")
        worker.artifact_sha256 = _optional_string(health.get("artifact_sha256"), "artifact_sha256")
        worker.service_version = _optional_string(health.get("service_version"), "service_version")
        worker.instance_id = _optional_string(health.get("instance_id"), "instance_id")
        worker.last_seen = self.clock()
        worker.failures = 0
        worker.last_error = None
        worker.metadata = {key: health.get(key) for key in
                           ("execution_mode", "mlx_device", "model_revision") if key in health}
        compatibility = (worker.model, worker.artifact_sha256)
        if compatibility != previous_compatibility:
            self.compatibility_key[worker_id] = compatibility
            if previous_compatibility is None:
                self.compatible[(None, None)].append(worker_id)
            self.compatible[(worker.model, None)].append(worker_id)
            self.compatible[(None, worker.artifact_sha256)].append(worker_id)
            self.compatible[compatibility].append(worker_id)
        if previous_instance is not None and previous_instance != worker.instance_id:
            self.lifecycle.append({"at_monotonic_s": worker.last_seen, "event": "worker_instance_changed",
                                   "worker_id": worker_id, "previous_instance_id": previous_instance,
                                   "instance_id": worker.instance_id})
        return worker

    def failure(self, worker_id, error_type):
        worker = self.workers[worker_id]
        worker.failures += 1
        worker.ready = False
        worker.last_error = error_type
        self.lifecycle.append({"at_monotonic_s": self.clock(), "event": "worker_probe_failed",
                               "worker_id": worker_id, "error_type": error_type,
                               "consecutive_failures": worker.failures})

    def set_draining(self, worker_id, draining):
        if type(draining) is not bool:
            raise ValueError("draining must be a boolean")
        self.workers[worker_id].draining = draining
        self.lifecycle.append({"at_monotonic_s": self.clock(), "event": "worker_drain_changed",
                               "worker_id": worker_id, "draining": draining})
        return self.workers[worker_id]

    def remove(self, worker_id):
        worker = self.workers[worker_id]
        if worker.in_flight_routes:
            raise ValueError("worker has in-flight routed requests")
        removed = self.workers.pop(worker_id)
        self.membership_revision += 1
        self.compatibility_key.pop(worker_id, None)
        self.lifecycle.append({"at_monotonic_s": self.clock(), "event": "worker_removed",
                               "worker_id": worker_id})
        return removed

    def select(self, estimated_kv_bytes, *, model=None, artifact_sha256=None, request_id=None):
        now = self.clock()
        eligible = []
        excluded = {"not_ready": 0, "draining": 0, "stale": 0, "no_slots": 0,
                    "model_mismatch": 0, "artifact_mismatch": 0, "memory_pressure": 0}
        if self.policy == "power-of-two":
            key = (model, artifact_sha256)
            pool = self.compatible.get(key, ())
            cursor = self.cursors[key]
            inspected = 0
            seen = set()
            # Two eligible choices keep the hot path constant-time. Invalid/tombstoned
            # entries are skipped; a full scan below preserves exact no-capacity behavior.
            while inspected < len(pool) and inspected < 32 and len(eligible) < 2:
                worker_id = pool[(cursor + inspected) % len(pool)]
                worker = self.workers.get(worker_id)
                if (worker_id not in seen and worker is not None and self.compatibility_key.get(worker_id) ==
                        (worker.model, worker.artifact_sha256) and
                        _eligible(worker, now, self.stale_after_s, estimated_kv_bytes,
                                  model, artifact_sha256) is None):
                    eligible.append(worker)
                    seen.add(worker_id)
                inspected += 1
            self.cursors[key] = cursor + max(1, inspected)
        if self.policy != "power-of-two" or not eligible:
            for worker in self.workers.values():
                reason = _eligible(worker, now, self.stale_after_s, estimated_kv_bytes,
                                   model, artifact_sha256)
                if reason is None:
                    eligible.append(worker)
                else:
                    excluded[reason] += 1
        if not eligible:
            self.history.append({"at_monotonic_s": now, "request_id": request_id,
                                 "outcome": "rejected", "estimated_kv_bytes": estimated_kv_bytes,
                                 "model": model, "artifact_sha256": artifact_sha256,
                                 "excluded": excluded})
            return None
        if self.policy == "round-robin":
            eligible.sort(key=lambda worker: worker.worker_id)
            worker = eligible[self.round_robin_cursor % len(eligible)]
            self.round_robin_cursor += 1
            reason = "round-robin"
        else:
            def score(worker):
                capacity = worker.execution_concurrency + worker.queue_capacity
                load = worker.accounted_load / capacity
                return load, worker.memory_pressure, worker.assigned_total, worker.worker_id
            worker = min(eligible, key=score)
            reason = ("power-of-two-load-then-memory" if self.policy == "power-of-two"
                      else "lowest-load-then-memory")
        capacity = worker.execution_concurrency + worker.queue_capacity
        load = worker.accounted_load / capacity
        worker.in_flight_routes += 1
        worker.assigned_total += 1
        self.placements += 1
        placement = Placement(worker.worker_id, worker.url, reason, self.policy, load, worker.memory_pressure)
        self.history.append({"at_monotonic_s": now, "request_id": request_id,
                             "outcome": "placed", "estimated_kv_bytes": estimated_kv_bytes,
                             "model": model, "artifact_sha256": artifact_sha256,
                             "placement": placement.as_dict()})
        return placement

    def release(self, worker_id):
        worker = self.workers.get(worker_id)
        if worker is None:
            return
        if worker.in_flight_routes <= 0:
            raise RuntimeError("route accounting underflow")
        worker.in_flight_routes -= 1

    def snapshot(self):
        now = self.clock()
        return {"policy": self.policy, "stale_after_s": self.stale_after_s,
                "placements": self.placements,
                "placement_history": list(self.history),
                "lifecycle_history": list(self.lifecycle),
                "workers": [worker.snapshot(now, self.stale_after_s)
                            for worker in sorted(self.workers.values(), key=lambda item: item.worker_id)]}

    def summary(self):
        """Return bounded operational state without materializing worker records."""
        now = self.clock()
        counts = {"registered": len(self.workers), "ready": 0, "available": 0,
                  "draining": 0, "stale": 0, "unready": 0, "in_flight_routes": 0}
        uniform_model = uniform_artifact = None
        model_seen = artifact_seen = False
        mixed_model = mixed_artifact = False
        for worker in self.workers.values():
            stale = now - worker.last_seen > self.stale_after_s
            counts["ready"] += worker.ready
            counts["draining"] += worker.draining
            counts["stale"] += stale
            counts["unready"] += not worker.ready
            counts["in_flight_routes"] += worker.in_flight_routes
            counts["available"] += worker.ready and not worker.draining and not stale
            if not model_seen:
                uniform_model = worker.model
                model_seen = True
            elif worker.model != uniform_model:
                mixed_model = True
            if not artifact_seen:
                uniform_artifact = worker.artifact_sha256
                artifact_seen = True
            elif worker.artifact_sha256 != uniform_artifact:
                mixed_artifact = True
        return {"policy": self.policy, "stale_after_s": self.stale_after_s,
                "membership_revision": self.membership_revision,
                "placements": self.placements, "worker_counts": counts,
                "uniform_model": None if mixed_model else uniform_model,
                "uniform_artifact_sha256": None if mixed_artifact else uniform_artifact,
                "audit_counts": {"placements": len(self.history),
                                 "lifecycle": len(self.lifecycle)}}

    def page(self, *, limit=100, cursor=None):
        if type(limit) is not int or not 1 <= limit <= 500:
            raise ValueError("limit must be an integer in 1..500")
        after = ""
        revision = self.membership_revision
        if cursor is not None:
            try:
                padding = "=" * (-len(cursor) % 4)
                value = json.loads(base64.urlsafe_b64decode(cursor + padding))
                if set(value) != {"revision", "after"} or type(value["revision"]) is not int or not isinstance(value["after"], str):
                    raise ValueError
                revision, after = value["revision"], value["after"]
            except (ValueError, TypeError, UnicodeDecodeError, json.JSONDecodeError):
                raise ValueError("cursor is invalid") from None
            if revision != self.membership_revision:
                raise RuntimeError("cursor membership revision is stale")
        now = self.clock()
        worker_ids = sorted(worker_id for worker_id in self.workers if worker_id > after)
        selected = worker_ids[:limit]
        next_cursor = None
        if len(worker_ids) > limit:
            payload = json.dumps({"revision": revision, "after": selected[-1]},
                                 separators=(",", ":")).encode()
            next_cursor = base64.urlsafe_b64encode(payload).decode().rstrip("=")
        return {"membership_revision": revision, "workers": [
                    self.workers[worker_id].snapshot(now, self.stale_after_s)
                    for worker_id in selected],
                "next_cursor": next_cursor, "limit": limit}


def _nonnegative_int(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _optional_string(value, name):
    if value is not None and (not isinstance(value, str) or not value):
        raise ValueError(f"{name} must be a nonempty string or null")
    return value


def _eligible(worker, now, stale_after_s, estimated_kv_bytes, model, artifact_sha256):
    if not worker.ready:
        return "not_ready"
    if worker.draining:
        return "draining"
    if now - worker.last_seen > stale_after_s:
        return "stale"
    if worker.available_slots < 1:
        return "no_slots"
    if model is not None and worker.model != model:
        return "model_mismatch"
    if artifact_sha256 is not None and worker.artifact_sha256 != artifact_sha256:
        return "artifact_mismatch"
    free_memory = worker.estimated_kv_budget_bytes - worker.estimated_kv_reserved_bytes
    if estimated_kv_bytes > free_memory:
        return "memory_pressure"
    return None
