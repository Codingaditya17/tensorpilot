"""Deterministic KV-cache reservation estimates for bounded admission."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Estimate:
    prompt_tokens_upper_bound: int
    total_tokens_upper_bound: int
    kv_bytes: int


class ResourceAdmission:
    """Reserve estimated cache bytes; estimates are never reported as measurements."""

    def __init__(self, budget_bytes=512 * 1024**2, *, bytes_per_token=16 * 1024,
                 sequence_overhead_bytes=2 * 1024**2):
        for name, value in (("budget_bytes", budget_bytes), ("bytes_per_token", bytes_per_token),
                            ("sequence_overhead_bytes", sequence_overhead_bytes)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.budget_bytes = budget_bytes
        self.bytes_per_token = bytes_per_token
        self.sequence_overhead_bytes = sequence_overhead_bytes
        self.reserved_bytes = 0
        self.reservations = 0

    def estimate(self, prompt, max_tokens):
        # One token per UTF-8 byte is deliberately conservative for normal text.
        # It is an admission estimate, not tokenizer output or measured memory.
        prompt_upper = max(1, len(prompt.encode("utf-8")))
        total = prompt_upper + max_tokens
        return Estimate(prompt_upper, total,
                        self.sequence_overhead_bytes + total * self.bytes_per_token)

    def reserve(self, estimate):
        if estimate.kv_bytes > self.budget_bytes - self.reserved_bytes:
            return False
        self.reserved_bytes += estimate.kv_bytes
        self.reservations += 1
        return True

    def release(self, estimate):
        self.reserved_bytes -= estimate.kv_bytes
        self.reservations -= 1
        if self.reserved_bytes < 0 or self.reservations < 0:
            raise RuntimeError("admission reservation accounting underflow")

    def snapshot(self):
        return {"estimate_kind": "conservative_kv_reservation_not_measured_memory",
                "estimated_kv_reserved_bytes": self.reserved_bytes,
                "estimated_kv_budget_bytes": self.budget_bytes,
                "estimated_bytes_per_token": self.bytes_per_token,
                "estimated_sequence_overhead_bytes": self.sequence_overhead_bytes,
                "reservations": self.reservations,
                "pressure_ratio": self.reserved_bytes / self.budget_bytes}
