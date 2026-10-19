"""Explainable adaptive batch-width decisions with bounded memory and latency."""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class AdaptiveDecision:
    width: int
    reason: str
    demand: int
    queue_depth: int
    active_sequences: int
    oldest_wait_s: float
    estimated_memory_pressure: float

    def as_dict(self):
        return asdict(self)


class AdaptiveBatchController:
    """Choose a safe target width from observable runtime state.

    This controller is deliberately deterministic. It does not predict GPU
    execution time and its memory input is the admission reservation estimate,
    not allocator telemetry.
    """

    def __init__(self, max_width=4, *, latency_target_s=.25,
                 memory_soft_limit=.85, memory_hard_limit=.95):
        if type(max_width) is not int or not 1 <= max_width <= 8:
            raise ValueError("max_width must be in 1..8")
        if not math.isfinite(latency_target_s) or latency_target_s <= 0:
            raise ValueError("latency_target_s must be finite and positive")
        if not 0 < memory_soft_limit < memory_hard_limit <= 1:
            raise ValueError("memory limits must satisfy 0 < soft < hard <= 1")
        self.max_width = max_width
        self.latency_target_s = latency_target_s
        self.memory_soft_limit = memory_soft_limit
        self.memory_hard_limit = memory_hard_limit
        self.last_decision = None
        self.decision_count = 0

    def choose(self, *, queue_depth, active_sequences, oldest_wait_s,
               estimated_memory_pressure):
        if min(queue_depth, active_sequences) < 0:
            raise ValueError("queue and active counts cannot be negative")
        if not math.isfinite(oldest_wait_s) or oldest_wait_s < 0:
            raise ValueError("oldest_wait_s must be finite and nonnegative")
        if not math.isfinite(estimated_memory_pressure) or estimated_memory_pressure < 0:
            raise ValueError("estimated_memory_pressure must be finite and nonnegative")
        demand = min(self.max_width, max(1, queue_depth + active_sequences))
        width, reason = demand, "demand"
        if estimated_memory_pressure >= self.memory_hard_limit:
            width, reason = 1, "memory-hard-limit"
        elif estimated_memory_pressure >= self.memory_soft_limit:
            width, reason = min(demand, 2), "memory-soft-limit"
        elif queue_depth and oldest_wait_s >= self.latency_target_s:
            width, reason = self.max_width, "latency-target"
        elif queue_depth == 0:
            width, reason = max(1, active_sequences), "no-waiters"
        decision = AdaptiveDecision(width, reason, demand, queue_depth, active_sequences,
                                    oldest_wait_s, estimated_memory_pressure)
        self.last_decision = decision
        self.decision_count += 1
        return decision

    def snapshot(self):
        return {"policy": "deterministic_queue_latency_memory_heuristic",
                "max_width": self.max_width,
                "latency_target_s": self.latency_target_s,
                "memory_soft_limit": self.memory_soft_limit,
                "memory_hard_limit": self.memory_hard_limit,
                "decision_count": self.decision_count,
                "last_decision": self.last_decision.as_dict() if self.last_decision else None}
