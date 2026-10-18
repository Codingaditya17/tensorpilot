"""Per-service metrics; labels never contain prompts, tokens, or request IDs."""
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest
import time


class Metrics:
    def __init__(self):
        self.registry = CollectorRegistry()
        options = {"registry": self.registry}
        self.http = Counter("tensorpilot_http_requests", "Finished HTTP handlers by status",
                            ["status"], **options)
        self.admitted = Counter("tensorpilot_admitted_requests", "Jobs accepted into the scheduler", **options)
        self.rejected = Counter("tensorpilot_rejected_requests", "Jobs rejected at admission", **options)
        self.outcomes = Counter("tensorpilot_jobs", "Worker outcomes; completed means generated",
                                ["outcome"], **options)
        self.tokens = Counter("tensorpilot_output_tokens", "Generated non-EOS tokens, including partial jobs", **options)
        self.queue_depth = Gauge("tensorpilot_queue_depth", "Waiting jobs, excluding active execution", **options)
        self.active = Gauge("tensorpilot_active_requests", "Active model executions", **options)
        self.call_age = Gauge("tensorpilot_worker_call_age_seconds", "Elapsed current backend call; zero when idle", **options)
        self.watchdog_trips = Counter("tensorpilot_worker_stalls", "Backend stall watchdog trips", **options)
        self.mlx_active = Gauge("tensorpilot_mlx_active_bytes", "Last sampled MLX active allocation, not process RSS", **options)
        self.mlx_peak = Gauge("tensorpilot_mlx_peak_bytes_process", "MLX process-lifetime allocation peak", **options)
        self.memory_age = Gauge("tensorpilot_memory_sample_age_seconds", "Age of cached backend memory sample", **options)
        self.scheduled = Counter("tensorpilot_scheduled_requests", "Jobs dispatched by scheduling decision",
                                 ["reason"], **options)
        self.queue_wait = Histogram("tensorpilot_queue_wait_seconds", "Admission to worker start, all dequeued jobs",
                                    buckets=(.001, .01, .05, .1, .25, .5, 1, 2, 5, 15, 60, 120), **options)
        self.ttft = Histogram("tensorpilot_ttft_seconds", "Admission to first non-EOS token, where observed",
                             buckets=(.01, .05, .1, .25, .5, 1, 2, 5, 15, 60, 120), **options)

    def render(self, engine):
        self.queue_depth.set(engine.queue.qsize())
        self.active.set(int(engine.active is not None))
        self.call_age.set(time.perf_counter() - engine.call_started_s if engine.call_started_s is not None else 0)
        if engine.last_memory_sample_s is not None:
            self.memory_age.set(time.perf_counter() - engine.last_memory_sample_s)
        return generate_latest(self.registry)

    def update_memory(self, sample):
        self.mlx_active.set(sample["mlx_active_bytes"])
        self.mlx_peak.set(sample["mlx_peak_bytes_process"])
