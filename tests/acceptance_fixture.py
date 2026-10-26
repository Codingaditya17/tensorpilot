"""Builds a synthetic but structurally real evaluation session on disk for the acceptance tests.

Records go through the real ``load.summarize``, so rates, percentiles and SLO attainment follow production semantics. Nothing
here is a measurement; it only has to be shaped like the evidence the stage runner writes.
"""
import hashlib
import json
import tarfile
from itertools import count
from pathlib import Path

from tensorpilot import gpu_bench, load, suite
from tensorpilot.acceptance import load_criteria

STAMP = "20261101T000000Z"
UUID = "GPU-11111111-2222-3333-4444-555555555555"
POD = "pod-abc123"
WORKLOAD = "w" * 64
_ids = count()


def record(index, *, tokens=20, ttft=0.2, itl=0.04, elapsed=1.0, outcome="completed", error=None, finish="length",
           max_tokens=None, request_id=None, generated=None):
    ok = outcome == "completed"
    rec = {"index": index, "ok": ok, "class": "short", "max_tokens": max_tokens or tokens, "outcome": outcome,
           "client_elapsed_s": elapsed, "client_ttft_s": ttft if ok else None, "client_tpot_s": itl if ok else None,
           "client_inter_token_s": [itl] * (tokens - 1) if ok else [], "client_last_token_s": elapsed,
           "dispatch_offset_s": index * 0.01, "received_output_tokens": tokens if ok else 0, "prompt_tokens": 16,
           "http_status": 200 if ok else 503, "request_id": request_id or f"req-{next(_ids)}"}
    if ok:
        rec["server"] = {"generated_tokens": tokens if generated is None else generated, "finish_reason": finish,
                         "queue_wait_s": 0.01, "inter_token_latency_s": [itl] * (tokens - 1), "phase_profile": {}}
    if error:
        rec["error_code"] = error
    return rec


def good_records(n, **kw):
    return [record(i, **kw) for i in range(n)]


class Session:
    def __init__(self, root, *, requests=12):
        self.root, self.requests = Path(root), requests
        self.criteria = load_criteria()
        self.criteria["scale"]["min_total_requests"] = 100     # the fixture is small; a test restores the real number
        self.stages = {}
        self.finished = {}
        self.ledger = []

    # ----- low level writers -------------------------------------------------------------------------------
    def stage_dir(self, stage):
        if stage not in self.stages:
            directory = self.root / f"{stage}-{STAMP}"
            (directory / "bench").mkdir(parents=True, exist_ok=True)
            self.stages[stage] = {"dir": directory, "experiments": {}}
        return self.stages[stage]["dir"]

    def experiment(self, stage, name, records, *, tps=500.0, repeat=1, label="closed", rate=None, slo=True, elapsed=20,
                   status="completed", reason=None, queue=None, oom=0, allocated=3_000_000_000, telemetry=True,
                   workload=WORKLOAD, vram=20000.0, reservations=0, drop_journal=0):
        directory = self.stage_dir(stage)
        model = suite.STAGES.get(stage, {}).get("model", "7b")
        definition = self.criteria["latency"]["slo_by_model"][model] if slo is True else (slo or None)
        tokens = sum(r["server"]["generated_tokens"] for r in records if r["ok"])
        summary = load.summarize(records, max(tokens / tps, 0.5), definition)
        report = {"summary": summary, "metadata": {"workload_sha256": workload},
                  "experiment": {"id": name, "kind": "open_loop" if rate else "closed_loop", "repeat": repeat,
                                 "arrival": label, "rate_per_s": rate},
                  "telemetry": {"gpu": {"0": {"memory_used_mib": {"max": vram}}}}}
        folder = directory / "bench" / name
        (folder / "telemetry").mkdir(parents=True, exist_ok=True)
        stem = f"repeat{repeat:02d}_{label}"
        (folder / f"{stem}.json").write_text(json.dumps(report))
        lines = [json.dumps({"type": "metadata"})] + [json.dumps({"type": "request", "record": r})
                                                      for r in records[:len(records) - drop_journal]]
        (folder / f"{stem}.requests.jsonl").write_text("\n".join(lines) + "\n")
        if telemetry and repeat == 1:
            (folder / "telemetry" / "gpu.jsonl").write_text(
                "\n".join(json.dumps({"t": float(t), "util_gpu_pct": 90.0}) for t in range(elapsed)) + "\n")
            depth = queue if queue is not None else [0] * elapsed
            rows = [{"t": float(t), "queue_depth": depth[t % len(depth)] if isinstance(depth, list) else depth,
                     "cuda_oom_events": oom if t == elapsed - 1 else 0, "cuda_allocated_bytes": allocated,
                     "admission_reservations": reservations} for t in range(elapsed)]
            (folder / "telemetry" / "server.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        self.stages[stage]["experiments"][name] = {
            "artifacts": [], "status": status, "abort_reason": reason, "elapsed_s": float(elapsed),
            "telemetry_overhead": {"interval_s": 1.0}}
        if stage in self.finished:     # a test changed an experiment after the stage was sealed: keep the manifest honest
            self.write_manifest(stage)

    def write_manifest(self, stage):
        options = self.finished[stage]
        s = suite.STAGES[stage]
        digest = gpu_bench.config_fingerprint(suite.config_for(stage)) if options["config_ok"] else "0" * 64
        target = {"model": suite.MODELS[s["model"]]["repo"],
                  "revision": options["revision"] or suite.MODELS[s["model"]]["revision"],
                  "dtype": s["dtype"], "quantization": s["quantization"], "engine": s["engine"]}
        (self.stage_dir(stage) / "bench" / "manifest.json").write_text(json.dumps(
            {"config_sha256": digest, "target": target, "experiments": self.stages[stage]["experiments"]}))

    def finish_stage(self, stage, *, uuid=UUID, pod=POD, image="ghcr.io/x/cuda-experiment@sha256:" + "a" * 64,
                     config_ok=True, revision=None):
        directory = self.stage_dir(stage)
        self.finished[stage] = {"config_ok": config_ok, "revision": revision}
        self.write_manifest(stage)
        (directory / "preflight.json").write_text(json.dumps(
            {"ready": True, "checks": [{"name": "nvidia_smi_visible",
                                         "telemetry": {"devices": [{"name": "NVIDIA GeForce RTX 4090"}]}}]}))
        (directory / "smoke.json").write_text(json.dumps({"summary": {"successful_requests": 5}}))
        info = {"gpus": [{"name": "RTX 4090", "uuid": uuid}], "build_info": {"source_sha": "b" * 40, "image_tag": "t"},
                "environment": {"RUNPOD_POD_ID": pod, "TENSORPILOT_IMAGE_REF": image}}
        (directory / "stage-info.json").write_text(json.dumps(info))

    # ----- the whole good session --------------------------------------------------------------------------
    def build(self, *, serial_tps=50.0, batch_tps=300.0, vllm_tps=500.0):
        n = self.requests
        for stage, tps in (("7b-bf16-prod-serial", serial_tps), ("7b-bf16-prod", batch_tps), ("vllm-7b-bf16-prod", vllm_tps)):
            self.experiment(stage, "cold-start", good_records(4), tps=tps, slo=False)
            for point in ("cmp-c1", "cmp-c8", "cmp-c32") + (() if "serial" in stage else ("cmp-c128",)):
                for repeat in (1, 2, 3):
                    self.experiment(stage, point, good_records(n), tps=tps, repeat=repeat)
        self.scale_and_ladder("7b-bf16-prod")
        self.scale_and_ladder("14b-nf4-prod")
        self.scale_and_ladder("vllm-7b-bf16-prod")
        self.experiment("7b-bf16-prod", "sustained-c32", good_records(1200, tokens=20), tps=batch_tps, elapsed=30)
        self.experiment("7b-bf16-prod", "cancellation", good_records(n), tps=batch_tps)
        self.experiment("7b-bf16-prod", "timeouts", good_records(n), tps=batch_tps)
        capacity = suite.STAGES["7b-bf16-prod"]["queue_size"]
        overload = good_records(n) + [record(n + i, outcome="rejected", error="overloaded") for i in range(8)]
        self.experiment("7b-bf16-prod", "overload", overload, tps=batch_tps, slo=False, rate=400.0, label="rate400.0",
                        queue=[0, 400, capacity, capacity])
        self.experiment("7b-bf16-prod", "recovery-after-overload", good_records(64), tps=batch_tps, slo=False)
        for stage in ("7b-bf16-prod-serial", "7b-bf16-prod", "14b-nf4-prod", "vllm-7b-bf16-prod"):
            self.finish_stage(stage)
        (self.stage_dir("7b-bf16-prod") / "invariance.json").write_text(json.dumps(invariance(1.0)))
        probe = {"passed": True, "prompts": 32, "exact_matches": 32, "near_tie": 0, "unexplained": 0}
        for stage in ("probe-7b", "probe-0.5b"):
            (self.stage_dir(stage) / "divergence-probe.json").write_text(json.dumps(probe))
        (self.stage_dir("0.5b-quick-batch") / "invariance.json").write_text(json.dumps(invariance(0.96875)))
        probe_near = dict(probe, exact_matches=31, near_tie=1)
        (self.stage_dir("probe-0.5b") / "divergence-probe.json").write_text(json.dumps(probe_near))
        quality = self.stage_dir("quality-7b")
        (quality / "comparison.json").write_text(json.dumps(
            comparison(1.03, 0.95)))
        (quality / "reference-bf16.json").write_text(json.dumps(capture(-1.5)))
        (self.stage_dir("quality-14b") / "nf4-absolute.json").write_text(json.dumps(capture(-1.3)))
        fault = self.stage_dir("faults-7b") / "fault"
        fault.mkdir(parents=True, exist_ok=True)
        (fault / "fault-report.json").write_text(json.dumps(
            {"passed": True, "kill": {"recovery_s": 41.0}, "checks": [{"name": "one_record_per_request", "passed": True, "detail": "ok"}]}))
        self.ledger = [{"stage": s, "elapsed_min": 10.0, "cap_min": 30.0, "exit_status": 0} for s in ("7b-bf16-prod", "faults-7b")]
        return self

    def scale_and_ladder(self, stage):
        n = self.requests
        tps = {"7b-bf16-prod": 300.0, "14b-nf4-prod": 120.0, "vllm-7b-bf16-prod": 500.0}[stage]
        for level in self.criteria["scale"]["concurrency_levels"]:
            self.experiment(stage, f"scale-c{level}", good_records(n), tps=tps)
        for rate in (1, 2, 4, 8, 12, 16):
            bad = rate >= 12
            records = [record(i, ttft=5.0 if bad and i % 2 else 0.3) for i in range(n)]
            self.experiment(stage, "ladder-poisson", records, tps=tps, rate=float(rate), label=f"rate{rate}.0")


def invariance(fraction):
    exact = round(fraction * 32)
    entry = {"exact_match_fraction": fraction, "comparable": 32, "exact_matches": exact, "mean_prefix_fraction": fraction,
             "earliest_divergence": None if exact == 32 else 7, "median_divergence": None if exact == 32 else 7}
    return {"together_vs_solo": entry, "staggered_vs_solo": entry}


def comparison(ratio, top1):
    return {"positions_compared": 400, "perplexity_reference": 5.0, "perplexity_test": 5.0 * ratio,
            "perplexity_ratio_test_over_reference": ratio, "top1_agreement": top1, "top5_overlap": 0.9,
            "kl_reference_to_test_nats_topk_support": 0.02, "greedy_mean_prefix_agreement": 0.8}


def capture(logprob):
    return {"corpus_sha256": "c" * 64, "teacher_forced": {"a": {"logprob_of_next": [logprob] * 50}}}


def archive(directory, name, content=b"evidence\n"):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / "payload.txt"
    source.write_bytes(content)
    path = directory / f"{name}.tar.gz"
    with tarfile.open(path, "w:gz") as tar:
        tar.add(source, arcname="payload.txt")
    source.unlink()
    (directory / f"{name}.tar.gz.sha256").write_text(hashlib.sha256(path.read_bytes()).hexdigest() + f"  {path.name}\n")
    return path
