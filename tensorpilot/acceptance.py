"""Automated acceptance checks for the production-scale evaluation: PASS, FAIL or INCONCLUSIVE per criterion.

The thresholds live in ``benchmarks/acceptance/criteria.json`` and are committed before any run. This module reads
evidence directories only (it never talks to a server and never runs a benchmark), applies those thresholds, and writes a
verdict. Three rules keep the verdict honest:

* Missing or partial evidence is INCONCLUSIVE, never a quiet pass. A stage that was not run cannot satisfy its criterion.
* The evaluation verdict is decided by the gate items only: correctness, safety, accounting, sustained-load reliability,
  reproducibility and budget (plus that the promised measurements exist). Performance targets (speedup over serial, latency at
  the operating point) are reported separately: an unmet target is a performance FAIL, not a reason to change a threshold.
* Failures are never excluded to improve a number: timeouts, rejections and failures are counted in every rate and every
  SLO attainment, and only deliberate client cancellations and documented overload rejections are excluded where a
  criterion says so.

    python -m tensorpilot.acceptance SESSION_DIR --ledger L1.jsonl --ledger L2.jsonl --archives ARCHIVE_DIR \\
        --report report.md --spent-usd 6.40 --output verdict.json --markdown verdict.md

Exit status: 0 gate PASS, 1 gate FAIL, 2 gate INCONCLUSIVE.
"""
import argparse
import hashlib
import json
import os
import re
import statistics
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

from . import gpu_bench, report_v10, suite

SCHEMA = "tensorpilot.acceptance/v1"
# Where the committed criteria and run configs live: the source checkout, the working directory, or the experiment image.
ROOTS = (Path(__file__).resolve().parent.parent, Path.cwd(), Path("/opt/tensorpilot"))
PASS, FAIL, INCONCLUSIVE = "PASS", "FAIL", "INCONCLUSIVE"
PERFORMANCE = "performance"
PROTOCOL_ERROR_CODES = {"invalid_stream", "token_count_mismatch"}
REVISION = re.compile(r"^[0-9a-f]{40}$")
# Credential shapes that must never appear in an evidence archive. The literal Pod key is also checked when supplied.
SECRET_PATTERNS = {
    "bearer token": re.compile(rb"Bearer\s+[A-Za-z0-9._~+/=-]{16,}"),
    "huggingface token": re.compile(rb"\bhf_[A-Za-z0-9]{20,}"),
    "github token": re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{30,}"),
    "runpod key": re.compile(rb"\brpa_[A-Za-z0-9]{20,}"),
    "api secret key": re.compile(rb"\bsk-[A-Za-z0-9]{20,}"),
    "aws access key": re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
    "private key block": re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
}


def repo_file(*parts):
    """The first existing copy of a committed file, or None."""
    for root in ROOTS:
        candidate = root.joinpath(*parts)
        if candidate.exists():
            return candidate
    return None


def load_criteria(path=None):
    path = Path(path) if path else repo_file("benchmarks", "acceptance", "criteria.json")
    if path is None:
        raise FileNotFoundError("benchmarks/acceptance/criteria.json not found; pass --criteria")
    return json.loads(path.read_text())


class Item:
    def __init__(self, ident, criterion, title, category, status, detail, **measured):
        self.ident, self.criterion, self.title, self.category = ident, criterion, title, category
        self.status, self.detail, self.measured = status, detail, measured

    def as_dict(self):
        return {"id": self.ident, "criterion": self.criterion, "title": self.title, "category": self.category,
                "status": self.status, "detail": self.detail, "measured": self.measured}


def worst(statuses):
    statuses = list(statuses)
    if FAIL in statuses:
        return FAIL
    if INCONCLUSIVE in statuses or not statuses:
        return INCONCLUSIVE
    return PASS


# ----- evidence access ------------------------------------------------------------------------------------------

def _json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def _jsonl(path):
    rows = []
    try:
        for line in Path(path).read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    except OSError:
        pass
    return rows


def request_records(stage, experiment):
    """Every non-warmup request record of an experiment, with the file it came from."""
    folder = stage["dir"] / "bench" / experiment
    for path in sorted(folder.glob("repeat*.requests.jsonl")):
        for row in _jsonl(path):
            if row.get("type") == "request":
                yield path.name, row["record"]


def tp_stages(session):
    return {name: s for name, s in session.items()
            if suite.STAGES.get(name, {}).get("engine") == "tensorpilot" and name.endswith(("-prod", "-prod-serial"))}


def bench_stages(session):
    return {name: s for name, s in session.items() if s["reports"]}


def median_of(reports, key):
    values = [r["summary"].get(key) for r in reports if r["summary"].get(key) is not None]
    return statistics.median(values) if values else None


def missing(ident, criterion, title, category, what):
    return Item(ident, criterion, title, category, INCONCLUSIVE, f"{what} was not found in the evidence")


# ----- 1. model correctness -------------------------------------------------------------------------------------

def check_loads(session, stage_name, label):
    ident = f"1.loads-and-generates-{label}"
    title = f"{label} loads and generates on a real GPU"
    stage = session.get(stage_name)
    if stage is None:
        return missing(ident, 1, title, "correctness", f"stage {stage_name}")
    preflight, smoke = stage.get("preflight") or {}, _json(stage["dir"] / "smoke.json") or {}
    done = sum(r["summary"]["successful_requests"] for rs in stage["reports"].values() for r in rs)
    smoke_ok = (smoke.get("summary") or {}).get("successful_requests", 0)
    gpu = any(d.get("name") for c in preflight.get("checks", []) for d in (c.get("telemetry") or {}).get("devices", []))
    ok = bool(preflight.get("ready")) and smoke_ok >= 5 and done > 0
    detail = (f"preflight ready={preflight.get('ready')}, smoke {smoke_ok} completed, {done} benchmark requests completed, "
              f"CUDA device seen={gpu}")
    return Item(ident, 1, title, "correctness", PASS if ok and gpu else FAIL if preflight else INCONCLUSIVE, detail,
                smoke_completed=smoke_ok, benchmark_completed=done)


def check_protocol(session, criteria):
    total = invalid = 0
    for name, stage in tp_stages(session).items():
        for experiment in stage["reports"]:
            for _, r in request_records(stage, experiment):
                outcome = r.get("outcome") or ("completed" if r.get("ok") else "failed")
                if outcome in ("client_cancelled", "rejected", "timed_out"):
                    continue
                total += 1
                if r.get("error_code") in PROTOCOL_ERROR_CODES or (outcome == "failed" and r.get("error_code") == "incomplete_stream"):
                    invalid += 1
                elif outcome == "completed" and r.get("received_output_tokens") != (r.get("server") or {}).get("generated_tokens"):
                    invalid += 1
    minimum = criteria["correctness"]["protocol_valid_min"]
    if not total:
        return Item("1.protocol-valid", 1, "At least 99.9% protocol-valid responses", "correctness", INCONCLUSIVE,
                    "no TensorPilot request records were found")
    fraction = 1 - invalid / total
    return Item("1.protocol-valid", 1, "At least 99.9% protocol-valid responses", "correctness",
                PASS if fraction >= minimum else FAIL,
                f"{total - invalid} of {total} responses were protocol-valid ({fraction:.5f}, need {minimum}); "
                "cancelled, rejected and timed-out requests are not responses to validate",
                responses=total, invalid=invalid, fraction=fraction)


def check_integrity(session):
    """No silent truncation, no duplicate completions, no unaccounted requests."""
    truncated, duplicated, unaccounted, requests, checked = [], [], [], 0, 0
    tensorpilot = tp_stages(session)
    for name, stage in bench_stages(session).items():
        for experiment, reports in stage["reports"].items():
            ids = set()
            for report in reports:
                s = report["summary"]
                checked += 1
                if s["requests_total"] != sum(s["outcomes"].values()):
                    unaccounted.append(f"{name}/{experiment}: outcomes sum {sum(s['outcomes'].values())} != {s['requests_total']}")
                requests += s["requests_total"]
            for file, r in request_records(stage, experiment):
                if (r.get("outcome") == "completed" or r.get("ok")) and r.get("request_id"):
                    if r["request_id"] in ids:
                        duplicated.append(f"{name}/{experiment}: {r['request_id']}")
                    ids.add(r["request_id"])
                server = r.get("server") or {}
                if r.get("ok") and name in tensorpilot:
                    made, limit = server.get("generated_tokens"), r.get("max_tokens")
                    if server.get("finish_reason") not in ("stop", "length") or made is None or made > limit or \
                            (server.get("finish_reason") == "length" and made != limit):
                        truncated.append(f"{name}/{experiment}#{r.get('index')}")
            journal = sum(1 for _ in request_records(stage, experiment))
            expected = sum(r["summary"]["requests_total"] for r in reports)
            if journal != expected:
                unaccounted.append(f"{name}/{experiment}: journal has {journal} records, summaries say {expected}")
    if not checked:
        return Item("1.no-truncation-duplicates-or-lost-requests", 1, "Zero silent truncation, duplicate completions or "
                    "unaccounted requests", "correctness", INCONCLUSIVE, "no experiment reports were found")
    bad = len(truncated) + len(duplicated) + len(unaccounted)
    detail = (f"{requests} requests across {checked} runs: {len(truncated)} truncated, {len(duplicated)} duplicated, "
              f"{len(unaccounted)} unaccounted" + (f"; first: {(truncated + duplicated + unaccounted)[0]}" if bad else ""))
    return Item("1.no-truncation-duplicates-or-lost-requests", 1,
                "Zero silent truncation, duplicate completions or unaccounted requests", "correctness",
                FAIL if bad else PASS, detail, requests=requests, truncated=len(truncated), duplicated=len(duplicated),
                unaccounted=len(unaccounted))


def probe_reports(root, stage):
    """Every divergence-probe.json for ``stage`` under ``root``: the same probe may be run with different packing settings."""
    found = []
    for directory in sorted(Path(root).iterdir()):
        match = report_v10.STAGE_DIR.match(directory.name) if directory.is_dir() else None
        report = _json(directory / "divergence-probe.json") if match and match["stage"] == stage else None
        if report:
            found.append(report)
    return found


def effective_pack(stage, stage_name):
    """The prefill packing the worker of ``stage`` really ran with: the operator override recorded in stage-info, else the default."""
    override = ((_json(stage["dir"] / "stage-info.json") or {}).get("environment") or {}).get("PREFILL_PACK_TOKENS") if stage else None
    if override is not None:
        return int(override)
    return int(suite.STAGES.get(stage_name, {}).get("prefill_pack", 0))


def check_exactness(session, stage_name, probe_stage, criterion_id, title, root):
    """Exact, or every diverged request reproduced by a probe with the SAME packing setting and classified as a near-tie.

    A probe that merely passes proves nothing about a specific divergence: batch shapes, and so the flips, depend on the schedule.
    """
    stage = session.get(stage_name)
    invariance = (stage or {}).get("invariance")
    if not invariance:
        return Item(criterion_id, 1, title, "correctness", INCONCLUSIVE,
                    f"no invariance.json in {stage_name}: the batch-invariance check did not run or did not complete")
    fractions = {k: invariance[k]["exact_match_fraction"] for k in ("together_vs_solo", "staggered_vs_solo")}
    if all(v == 1.0 for v in fractions.values()):
        return Item(criterion_id, 1, title, "correctness", PASS, "every request matched its solo output exactly", **fractions)
    diverged = {r["index"] for k in fractions for r in invariance[k].get("diverged_requests", [])}
    pack = effective_pack(stage, stage_name)
    probes = probe_reports(root, probe_stage)
    unexplained = [r["index"] for p in probes for r in p.get("diverged_requests", []) if r.get("verdict") != "near_tie"]
    if unexplained:
        return Item(criterion_id, 1, title, "correctness", FAIL,
                    f"exact-match fractions {fractions}; a probe found divergence NOT explained by rounding (requests {sorted(unexplained)})",
                    **fractions, unexplained=sorted(unexplained))
    matching = [p for p in probes if (p.get("settings") or {}).get("prefill_pack_tokens") == pack]
    if not matching:
        return Item(criterion_id, 1, title, "correctness", INCONCLUSIVE,
                    f"exact-match fractions {fractions}, diverged requests {sorted(diverged)}: no {probe_stage} probe ran with prefill "
                    f"packing {pack} (the setting that produced them)", **fractions)
    explained = {r["index"] for p in matching for r in p.get("diverged_requests", []) if r.get("verdict") == "near_tie"}
    reproduced = diverged & explained
    if diverged - explained:
        return Item(criterion_id, 1, title, "correctness", INCONCLUSIVE,
                    f"exact-match fractions {fractions}: the probe (packing {pack}) did not reproduce requests "
                    f"{sorted(diverged - explained)}, so their divergence is not yet explained by logits", **fractions,
                    explained=sorted(reproduced))
    return Item(criterion_id, 1, title, "correctness", PASS,
                f"exact-match fractions {fractions}; every diverged request {sorted(diverged)} was reproduced by the probe "
                f"(packing {pack}) and is a rounding-level near-tie (documented numerical divergence)", **fractions,
                explained=sorted(reproduced))


def _perplexity(capture):
    values = [x for sequence in capture["teacher_forced"].values() for x in sequence["logprob_of_next"]]
    return (statistics.fmean(values), len(values)) if values else (None, 0)


def check_nf4_7b(session, criteria):
    title = "NF4 7B fidelity against the bf16 reference meets the declared threshold"
    stage = session.get("quality-7b")
    comparison = (stage or {}).get("comparison")
    if not comparison:
        return missing("1.nf4-fidelity-7b", 1, title, "correctness", "quality-7b comparison.json")
    t = criteria["correctness"]["nf4_vs_bf16_7b"]
    ratio, top1, top5 = (comparison.get(k) for k in ("perplexity_ratio_test_over_reference", "top1_agreement", "top5_overlap"))
    ok = None not in (ratio, top1, top5) and ratio <= t["perplexity_ratio_max"] and top1 >= t["top1_agreement_min"] \
        and top5 >= t["top5_overlap_min"]
    return Item("1.nf4-fidelity-7b", 1, title, "correctness", PASS if ok else FAIL,
                f"perplexity ratio {ratio} (max {t['perplexity_ratio_max']}), top-1 agreement {top1} (min "
                f"{t['top1_agreement_min']}), top-5 overlap {top5} (min {t['top5_overlap_min']})",
                perplexity_ratio=ratio, top1_agreement=top1, top5_overlap=top5)


def check_nf4_14b(session, criteria):
    title = "NF4 14B is at least as good as the 7B bf16 reference on the same text"
    q14, q7 = session.get("quality-14b"), session.get("quality-7b")
    absolute = _json(q14["dir"] / "nf4-absolute.json") if q14 else None
    reference = _json(q7["dir"] / "reference-bf16.json") if q7 else None
    if not absolute or not reference:
        return missing("1.nf4-fidelity-14b", 1, title, "correctness", "quality-14b/nf4-absolute.json and quality-7b/reference-bf16.json")
    if absolute.get("corpus_sha256") != reference.get("corpus_sha256"):
        return Item("1.nf4-fidelity-14b", 1, title, "correctness", INCONCLUSIVE, "the captures used different corpora")
    (lp14, n14), (lp7, _) = _perplexity(absolute), _perplexity(reference)
    if lp14 is None or lp7 is None:
        return Item("1.nf4-fidelity-14b", 1, title, "correctness", INCONCLUSIVE, "a capture holds no teacher-forced positions")
    import math
    ratio = math.exp(-lp14) / math.exp(-lp7)
    limit = criteria["correctness"]["nf4_14b_perplexity_ratio_vs_7b_bf16_max"]
    return Item("1.nf4-fidelity-14b", 1, title, "correctness", PASS if ratio <= limit else FAIL,
                f"14B NF4 perplexity is {ratio:.3f}x the 7B bf16 reference over {n14} positions (max {limit}). There is no bf16 "
                "14B reference on a 24 GB card, so this is a sanity bound, not a measure of quantization loss",
                perplexity_ratio_vs_7b=ratio, positions=n14)


# ----- 2. scale and throughput ----------------------------------------------------------------------------------

LEVEL_KEYS = ("successful_requests_per_s", "completed_output_tokens_per_s")
LATENCY_KEYS = ("p50_client_latency_s", "p95_client_latency_s", "p99_client_latency_s",
                "p50_client_ttft_s", "p95_client_ttft_s", "p99_client_ttft_s")


def check_levels(session, stage_name, label, criteria):
    ident, title = f"2.concurrency-levels-{label}", f"Throughput and latency reported at every concurrency level ({label})"
    stage = session.get(stage_name)
    if stage is None:
        return missing(ident, 2, title, "completeness", f"stage {stage_name}")
    problems, rows = [], {}
    manifest = (stage.get("manifest") or {}).get("experiments", {})
    for level in criteria["scale"]["concurrency_levels"]:
        name = f"scale-c{level}"
        reports = stage["reports"].get(name) or []
        status = (manifest.get(name) or {}).get("status")
        reason = (manifest.get(name) or {}).get("abort_reason")
        if not reports:
            problems.append(f"c{level}: no report")
        elif status == "aborted" and reason == "time_cap":
            problems.append(f"c{level}: cut short by its time cap")
        elif status not in ("completed", "aborted"):
            problems.append(f"c{level}: status {status}")
        else:
            s = reports[0]["summary"]
            rows[level] = {k: median_of(reports, k) for k in (*LEVEL_KEYS, *LATENCY_KEYS, "error_rate", "rejection_rate",
                                                              "p95_dispatch_lag_s")}
            lag, limit = rows[level]["p95_dispatch_lag_s"], criteria["scale"]["max_p95_client_dispatch_lag_s"]
            if lag is not None and lag > limit:   # the client shares the Pod's CPUs: a starved client measures itself
                problems.append(f"c{level}: the load generator fell behind its own schedule (p95 dispatch lag {lag:.2f} s)")
            if s["successful_requests"] and any(rows[level][k] is None for k in LATENCY_KEYS):
                problems.append(f"c{level}: a latency percentile is missing")
    return Item(ident, 2, title, "completeness", PASS if not problems else INCONCLUSIVE,
                "all seven levels were measured and reported (rejections and timeouts included, not required to be absent)"
                if not problems else "; ".join(problems), levels=rows)


def check_volume(session, criteria):
    total = sum(r["summary"]["requests_total"] for s in bench_stages(session).values()
                for reports in s["reports"].values() for r in reports)
    need, stretch = criteria["scale"]["min_total_requests"], criteria["scale"]["stretch_total_requests"]
    return Item("2.request-volume", 2, "At least 10,000 requests executed overall", "completeness",
                PASS if total >= need else INCONCLUSIVE,
                f"{total} requests executed across all stages (need {need}; the {stretch}-request stretch target is "
                f"{'met' if total >= stretch else 'not met'})", total_requests=total, stretch_met=total >= stretch)


def ladder_summary(stage, name="ladder-poisson"):
    rows = report_v10.ladder_rows(stage, name)
    best = report_v10.max_sustainable(rows)
    saturated = any(r["rate"] > (best or 0) and not (r["attainment"] is not None and r["attainment"] >= report_v10.OBJECTIVE["attainment"]
                                                         and (r["error_rate"] or 0) <= report_v10.OBJECTIVE["error_rate"])
                    for r in rows)
    return rows, best, saturated


def check_saturation(session, stage_name, label):
    ident, title = f"2.saturation-found-{label}", f"Maximum sustainable arrival rate found by driving to saturation ({label})"
    stage = session.get(stage_name)
    if stage is None or not stage["reports"].get("ladder-poisson"):
        return missing(ident, 2, title, "completeness", f"{stage_name} ladder-poisson")
    rows, best, saturated = ladder_summary(stage)
    if best is None:
        return Item(ident, 2, title, "completeness", INCONCLUSIVE,
                    f"even the lowest offered rate ({rows[0]['rate']}/s) missed the objective: the maximum is below the ladder; "
                    "rerun with lower rates", ladder=rows)
    if not saturated:
        return Item(ident, 2, title, "completeness", INCONCLUSIVE,
                    f"every offered rate up to {rows[-1]['rate']}/s met the objective: {best}/s is only a lower bound", ladder=rows)
    return Item(ident, 2, title, "completeness", PASS,
                f"{best} req/s is the highest rate that, with every lower rate, met the objective; a higher rate did not",
                max_sustainable_rate=best, ladder=rows)


def check_speedup(session, criteria):
    title = "Continuous batching is at least 1.5x the serial baseline on comparable workloads"
    serial, batch = session.get("7b-bf16-prod-serial"), session.get("7b-bf16-prod")
    if serial is None or batch is None:
        return missing("2.speedup-over-serial", 2, title, PERFORMANCE, "7b-bf16-prod-serial and 7b-bf16-prod")
    need, ratios, notes = criteria["scale"]["min_speedup_over_serial"], {}, []
    for point in criteria["scale"]["speedup_points"]:
        a, b = median_of(serial["reports"].get(point, []), "completed_output_tokens_per_s"), \
            median_of(batch["reports"].get(point, []), "completed_output_tokens_per_s")
        if a and b:
            ratios[point] = b / a
        else:
            notes.append(f"{point}: missing")
    if notes or not ratios:
        return Item("2.speedup-over-serial", 2, title, PERFORMANCE, INCONCLUSIVE, "; ".join(notes) or "no points", ratios=ratios)
    ok = all(v >= need for v in ratios.values())
    return Item("2.speedup-over-serial", 2, title, PERFORMANCE, PASS if ok else FAIL,
                ", ".join(f"{k}: {v:.2f}x" for k, v in ratios.items()) + f" (need {need}x; median of repeats, wall-clock tokens/s)",
                ratios=ratios)


# ----- 3. reliability -------------------------------------------------------------------------------------------

def check_sustained(session, criteria):
    title = "At sustainable load at least 99.9% of requests succeed"
    stage = session.get("7b-bf16-prod")
    reports = (stage or {}).get("reports", {}).get(criteria["reliability"]["sustained_experiment"])
    if not reports:
        return missing("3.sustained-success", 3, title, "reliability", "the 7B sustained-load experiment")
    done = total = 0
    for r in reports:
        s = r["summary"]
        counted = s["requests_total"] - s["client_cancelled_requests"] - s["rejected_requests"]
        total += counted
        done += s["successful_requests"]
    rel = criteria["reliability"]
    if total < rel["min_requests_for_three_nines"]:
        return Item("3.sustained-success", 3, title, "reliability", INCONCLUSIVE,
                    f"only {total} requests: too few to demonstrate {rel['sustained_success_min']}", requests=total)
    rate = done / total
    return Item("3.sustained-success", 3, title, "reliability", PASS if rate >= rel["sustained_success_min"] else FAIL,
                f"{done} of {total} succeeded ({rate:.5f}, need {rel['sustained_success_min']}); intentionally cancelled "
                "requests and documented overload rejections are excluded, timeouts and failures are not",
                success_rate=rate, requests=total)


def check_oom(session, criteria):
    title = "Zero CUDA out-of-memory events"
    events, sources, logged = 0, 0, []
    for name, stage in tp_stages(session).items():
        for experiment in stage["reports"]:
            for row in _jsonl(stage["dir"] / "bench" / experiment / "telemetry" / "server.jsonl"):
                sources += 1
                events = max(events, row.get("cuda_oom_events") or 0)
        log = stage["dir"] / "engine.log"
        if log.exists() and re.search(r"out of memory|OutOfMemoryError", log.read_text(errors="replace"), re.I):
            logged.append(name)
    if not sources:
        return Item("3.zero-cuda-oom", 3, title, "reliability", INCONCLUSIVE, "no worker telemetry was found")
    bad = events > criteria["reliability"]["max_cuda_oom_events"] or logged
    return Item("3.zero-cuda-oom", 3, title, "reliability", FAIL if bad else PASS,
                f"{sources} telemetry samples: max cuda_oom_events {events}; OOM text in engine logs: {logged or 'none'}",
                max_oom_events=events, logs_with_oom=logged)


def check_overload(session, criteria):
    title = "At overload the queue stays bounded, rejections are controlled and nothing leaks"
    stage = session.get("7b-bf16-prod")
    reports = (stage or {}).get("reports", {}).get("overload")
    if not reports:
        return missing("3.overload", 3, title, "reliability", "the 7B overload experiment")
    capacity = suite.STAGES["7b-bf16-prod"]["queue_size"]
    samples = _jsonl(stage["dir"] / "bench" / "overload" / "telemetry" / "server.jsonl")
    depth = max((s.get("queue_depth") or 0 for s in samples), default=None)
    s = reports[0]["summary"]
    rejected = sum(r["summary"]["rejected_requests"] for r in reports)
    odd = {k: v for k, v in (s.get("errors_by_code") or {}).items() if k not in ("overloaded", "deadline_exceeded")}
    reasons, status = [], PASS
    if depth is None:
        return Item("3.overload", 3, title, "reliability", INCONCLUSIVE, "no queue telemetry for the overload experiment")
    if depth > capacity:
        reasons.append(f"queue depth {depth} exceeded its capacity {capacity}")
        status = FAIL
    if rejected == 0:
        reasons.append("nothing was rejected: the overload never exceeded the queue, so the bound was not exercised")
        status = INCONCLUSIVE if status == PASS else status
    if sum(odd.values()) > 0.001 * max(s["requests_total"], 1):
        reasons.append(f"uncontrolled failures: {odd}")
        status = FAIL
    after = (stage["reports"].get("recovery-after-overload") or [None])[0]
    if after is None:
        reasons.append("no recovery-after-overload experiment")
        status = INCONCLUSIVE if status == PASS else status
    else:
        rec = after["summary"]
        rate = rec["successful_requests"] / max(rec["requests_total"], 1)
        if rate < criteria["reliability"]["post_overload_success_min"]:
            reasons.append(f"only {rate:.3f} of requests succeeded right after overload")
            status = FAIL
        first = _jsonl(stage["dir"] / "bench" / "recovery-after-overload" / "telemetry" / "server.jsonl")
        origin = _jsonl(stage["dir"] / "bench" / "cold-start" / "telemetry" / "server.jsonl")
        early = [row for row in first if row.get("t", 0) <= 10]
        if early and (min(row.get("queue_depth") or 0 for row in early) > 0
                      or min(row.get("admission_reservations") or 0 for row in early) > 0):
            reasons.append("the queue or admission reservations had not drained within 10 s of recovery beginning")
            status = FAIL
        if first and origin:
            drift = (first[-1].get("cuda_allocated_bytes") or 0) - (origin[0].get("cuda_allocated_bytes") or 0)
            if drift > criteria["reliability"]["overload_leak_tolerance_mib"] * 1024**2:
                reasons.append(f"CUDA allocated memory grew by {drift / 1024**2:.0f} MiB across overload")
                status = FAIL
    return Item("3.overload", 3, title, "reliability", status,
                "; ".join(reasons) or f"max queue depth {depth} of {capacity}, {rejected} controlled rejections, clean recovery",
                max_queue_depth=depth, queue_capacity=capacity, rejected=rejected)


def check_crash(session, criteria):
    title = "A worker crash recovers within 60 seconds with request accounting intact"
    stage = session.get("faults-7b")
    report = _json(stage["dir"] / "fault" / "fault-report.json") if stage else None
    if not report:
        return missing("3.crash-recovery", 3, title, "reliability", "faults-7b fault-report.json")
    recovery = (report.get("kill") or {}).get("recovery_s")
    limit = criteria["reliability"]["recovery_max_s"]
    failed = [c["name"] for c in report["checks"] if not c["passed"]]
    if recovery is None:
        return Item("3.crash-recovery", 3, title, "reliability", FAIL if failed else INCONCLUSIVE,
                    f"recovery time was not recorded; failed checks: {failed or 'none'}")
    ok = report.get("passed") and recovery <= limit
    return Item("3.crash-recovery", 3, title, "reliability", PASS if ok else FAIL,
                f"recovered in {recovery:.1f} s (limit {limit} s); failed accounting checks: {failed or 'none'}",
                recovery_s=recovery, failed_checks=failed)


# ----- 4. latency -----------------------------------------------------------------------------------------------

def _model_of(stage_name):
    return suite.STAGES.get(stage_name, {}).get("model")


def check_slo_declared(session, criteria):
    title = "Workload-specific SLOs were declared before execution and used unchanged"
    checked, wrong = 0, []
    for name, stage in bench_stages(session).items():
        declared = criteria["latency"]["slo_by_model"].get(_model_of(name))
        for experiment, reports in stage["reports"].items():
            for r in reports:
                slo = (r["summary"].get("slo") or {}).get("definition")
                if slo is None or declared is None:
                    continue
                checked += 1
                if slo != declared:
                    wrong.append(f"{name}/{experiment}: {slo}")
    if not checked:
        return Item("4.slo-declared-before-execution", 4, title, "reproducibility", INCONCLUSIVE, "no SLO-bearing report found")
    return Item("4.slo-declared-before-execution", 4, title, "reproducibility", FAIL if wrong else PASS,
                f"{checked} runs carry exactly the SLO committed in criteria.json" if not wrong else
                f"{len(wrong)} runs used a different SLO, first: {wrong[0]}", checked=checked)


def check_operating_point(session, stage_name, label, criteria):
    ident, title = f"4.latency-at-operating-point-{label}", f"P95 TTFT <= 2 s and P95 inter-token <= 150 ms at the accepted-load operating point ({label})"
    stage = session.get(stage_name)
    if stage is None or not stage["reports"].get("ladder-poisson"):
        return missing(ident, 4, title, PERFORMANCE, f"{stage_name} ladder-poisson")
    rows, best, _ = ladder_summary(stage)
    if best is None:
        return Item(ident, 4, title, PERFORMANCE, FAIL,
                    "no offered rate met the declared objective, so there is no accepted-load operating point", ladder=rows)
    report = next(r for r in stage["reports"]["ladder-poisson"] if r["experiment"]["rate_per_s"] == best)
    s = report["summary"]
    ttft, itl = s.get("p95_client_ttft_s"), s.get("p95_client_inter_token_s")
    need_t, need_i = criteria["latency"]["ttft_p95_s_max"], criteria["latency"]["itl_p95_s_max"]
    if ttft is None or itl is None:
        return Item(ident, 4, title, PERFORMANCE, INCONCLUSIVE, "the operating point has no TTFT or inter-token samples")
    ok = ttft <= need_t and itl <= need_i
    return Item(ident, 4, title, PERFORMANCE, PASS if ok else FAIL,
                f"at {best} req/s: P95 TTFT {ttft:.3f} s (max {need_t}), P95 inter-token {itl * 1000:.1f} ms (max {need_i * 1000:.0f}); "
                f"P50/P95/P99 e2e {s.get('p50_client_latency_s')}/{s.get('p95_client_latency_s')}/{s.get('p99_client_latency_s')} s",
                operating_rate=best, p95_ttft_s=ttft, p95_itl_s=itl)


def check_failures_visible(session):
    title = "Timeouts, rejections and failures are counted in every rate and SLO attainment"
    checked, bad = 0, []
    for name, stage in bench_stages(session).items():
        for experiment, reports in stage["reports"].items():
            for r in reports:
                s = r["summary"]
                checked += 1
                counted = s["requests_total"] - s["client_cancelled_requests"]
                slo = s.get("slo")
                if s.get("error_rate") is None and counted:
                    bad.append(f"{name}/{experiment}: no error rate")
                elif slo and counted and abs(slo["attainment"] - slo["good_requests"] / counted) > 1e-9:
                    bad.append(f"{name}/{experiment}: attainment is not good/(non-cancelled requests)")
                elif slo and slo["good_requests"] > s["successful_requests"]:
                    bad.append(f"{name}/{experiment}: more good requests than completed ones")
    if not checked:
        return Item("4.failures-not-hidden", 4, title, "reproducibility", INCONCLUSIVE, "no reports found")
    return Item("4.failures-not-hidden", 4, title, "reproducibility", FAIL if bad else PASS,
                f"{checked} runs checked" if not bad else "; ".join(bad[:3]), checked=checked)


# ----- 5. vLLM comparison ---------------------------------------------------------------------------------------

def _gpu_vram_mib(reports):
    values = [((r.get("telemetry") or {}).get("gpu", {}).get("0", {}).get("memory_used_mib") or {}).get("max") for r in reports]
    values = [v for v in values if v is not None]
    return max(values) if values else None


def vllm_pair_7b(session):
    """Which TensorPilot stage vLLM is compared with. A lean re-run (vllm-7b-bf16-cmp) is paired with the TensorPilot control that ran
    on the same Pod and card (7b-bf16-cmp); without that control it falls back to the archived 7b-bf16-prod, and the same-hardware
    item then reports, truthfully, that the two engines did not share a card."""
    if "vllm-7b-bf16-prod" not in session and "vllm-7b-bf16-cmp" in session:
        return ("7b-bf16-cmp" if "7b-bf16-cmp" in session else "7b-bf16-prod", "vllm-7b-bf16-cmp", "7b")
    return ("7b-bf16-prod", "vllm-7b-bf16-prod", "7b")


def check_vllm(session, criteria):
    out = []
    pairs = [vllm_pair_7b(session)]
    if "vllm-14b-nf4-prod" in session:
        pairs.append(("14b-nf4-prod", "vllm-14b-nf4-prod", "14b"))
    title_hw = "vLLM and TensorPilot ran on the same physical GPU, model revision and precision"
    title_rep = "Three measured repetitions per key configuration, on identical workloads"
    title_met = "Throughput, goodput, latency, VRAM and error rates are reported for both engines"
    for tp_name, vl_name, label in pairs:
        tp, vl = session.get(tp_name), session.get(vl_name)
        if tp is None or vl is None:
            for ident, title in ((f"5.same-hardware-{label}", title_hw), (f"5.repeats-and-workloads-{label}", title_rep),
                                 (f"5.metrics-reported-{label}", title_met)):
                out.append(missing(ident, 5, title, "completeness", f"stages {tp_name} and {vl_name}"))
            continue
        a, b = _json(tp["dir"] / "stage-info.json"), _json(vl["dir"] / "stage-info.json")
        ta, tb = (tp.get("manifest") or {}).get("target", {}), (vl.get("manifest") or {}).get("target", {})
        same_model = all(ta.get(k) == tb.get(k) for k in ("model", "revision", "dtype", "quantization"))
        if not a or not b or not a.get("gpus") or not b.get("gpus"):
            status, detail = INCONCLUSIVE, "stage-info.json with GPU UUIDs is missing, so one physical card cannot be proven"
        elif not tb or not ta:
            # A stage that failed before its first request (vLLM's engine never started) has a stage-info.json but no benchmark
            # manifest. That is a missing comparison, not a model or hardware mismatch.
            missing_stage = vl_name if not tb else tp_name
            status, detail = INCONCLUSIVE, f"{missing_stage} has no benchmark manifest (it produced no measurements), so nothing was compared"
        else:
            uuids = ({g["uuid"] for g in a["gpus"]}, {g["uuid"] for g in b["gpus"]})
            pods = (a.get("environment", {}).get("RUNPOD_POD_ID"), b.get("environment", {}).get("RUNPOD_POD_ID"))
            same_card = uuids[0] == uuids[1] and (pods[0] == pods[1])
            status = PASS if same_card and same_model else FAIL
            detail = (f"GPU {sorted(uuids[0])} vs {sorted(uuids[1])}, pods {pods[0]} vs {pods[1]}, model/revision/precision "
                      f"{'match' if same_model else 'DIFFER'}")
        out.append(Item(f"5.same-hardware-{label}", 5, title_hw, "completeness", status, detail))
        problems, ratios = [], {}
        for point in criteria["vllm"]["points"]:
            ra, rb = tp["reports"].get(point, []), vl["reports"].get(point, [])
            if len(ra) < criteria["vllm"]["min_repeats"] or len(rb) < criteria["vllm"]["min_repeats"]:
                problems.append(f"{point}: {len(ra)} TensorPilot / {len(rb)} vLLM repeats")
                continue
            digests = {(r.get("metadata") or {}).get("workload_sha256") for r in ra + rb}
            if len(digests) != 1:
                problems.append(f"{point}: the two engines did not receive the same workload")
            ratios[point] = {"tensorpilot_tok_s": median_of(ra, "completed_output_tokens_per_s"),
                             "vllm_tok_s": median_of(rb, "completed_output_tokens_per_s")}
        out.append(Item(f"5.repeats-and-workloads-{label}", 5, title_rep, "completeness",
                        PASS if not problems else INCONCLUSIVE if all("repeats" in p for p in problems) else FAIL,
                        "; ".join(problems) or "every key configuration has at least three repeats per engine and a shared workload hash",
                        throughput=ratios))
        gaps = []
        for point in criteria["vllm"]["points"]:
            for engine, stage in (("tensorpilot", tp), ("vllm", vl)):
                reports = stage["reports"].get(point, [])
                if not reports:
                    continue
                s = reports[0]["summary"]
                if _gpu_vram_mib(reports) is None:
                    gaps.append(f"{engine}/{point}: no VRAM")
                if (s.get("slo") or {}).get("goodput_requests_per_s") is None or s.get("error_rate") is None:
                    gaps.append(f"{engine}/{point}: no goodput or error rate")
        out.append(Item(f"5.metrics-reported-{label}", 5, title_met, "completeness", PASS if not gaps else INCONCLUSIVE,
                        "; ".join(gaps[:4]) or "throughput, goodput, latency percentiles, VRAM and error rates are present for both"))
    return out


def check_gap_explained(report_path, criteria):
    title = "Performance gaps to vLLM are explained with evidence"
    marker = criteria["evidence"]["gap_analysis_marker"]
    if not report_path or not Path(report_path).exists():
        return Item("5.gaps-explained", 5, title, "completeness", INCONCLUSIVE, "no written report was supplied (--report)")
    text = Path(report_path).read_text()
    if marker not in text:
        return Item("5.gaps-explained", 5, title, "completeness", INCONCLUSIVE, f"the report has no '{marker}' section")
    section = text.split(marker, 1)[1]
    cites = len(re.findall(r"\d+(?:\.\d+)?\s*(?:ms|s|tok/s|%|x|GiB|MiB)", section))
    return Item("5.gaps-explained", 5, title, "completeness", PASS if cites >= 3 else INCONCLUSIVE,
                f"the '{marker}' section cites {cites} measurements (at least 3 needed: an explanation without numbers is an opinion)")


# ----- 6. evidence ----------------------------------------------------------------------------------------------

def check_telemetry(session, criteria):
    title = "Continuous GPU telemetry covers every measured experiment"
    ev = criteria["evidence"]
    checked, bad = 0, []
    for name, stage in bench_stages(session).items():
        manifest = (stage.get("manifest") or {}).get("experiments", {})
        for experiment in stage["reports"]:
            info = manifest.get(experiment) or {}
            if info.get("telemetry_overhead") is None and info.get("status") != "completed":
                continue
            samples = _jsonl(stage["dir"] / "bench" / experiment / "telemetry" / "gpu.jsonl")
            elapsed = info.get("elapsed_s") or 0
            checked += 1
            if not samples:
                bad.append(f"{name}/{experiment}: no GPU samples")
                continue
            times = sorted(s["t"] for s in samples)
            gap = max((b - a for a, b in zip(times, times[1:])), default=0)
            if elapsed and len(samples) < ev["gpu_telemetry_min_coverage"] * elapsed:
                bad.append(f"{name}/{experiment}: {len(samples)} samples for {elapsed:.0f} s")
            elif gap > ev["gpu_telemetry_max_gap_factor"] * 1.0:
                bad.append(f"{name}/{experiment}: a {gap:.1f} s gap")
    if not checked:
        return Item("6.gpu-telemetry", 6, title, "reproducibility", INCONCLUSIVE, "no experiments with telemetry found")
    return Item("6.gpu-telemetry", 6, title, "reproducibility", FAIL if bad else PASS,
                f"{checked} experiments" + (": " + "; ".join(bad[:3]) if bad else " have continuous samples"), checked=checked)


def check_reproducible(session):
    title = "Configurations, model revisions and the image are pinned and reproducible"
    checked, bad, notes = 0, [], []
    for name, stage in bench_stages(session).items():
        manifest = stage.get("manifest") or {}
        path = repo_file("benchmarks", "gpu", f"v10-{name}.json")
        if path is not None and manifest.get("config_sha256"):
            checked += 1
            if gpu_bench.config_fingerprint(json.loads(path.read_text())) != manifest["config_sha256"]:
                bad.append(f"{name}: the run's config is not the committed config")
        if not REVISION.fullmatch(str((manifest.get("target") or {}).get("revision", ""))):
            bad.append(f"{name}: model revision is not a commit")
        info = _json(stage["dir"] / "stage-info.json") or {}
        build = info.get("build_info") or {}
        ref = (info.get("environment") or {}).get("TENSORPILOT_IMAGE_REF", "")
        if not build or build.get("source_sha") in (None, "unknown"):
            notes.append(f"{name}: no build info")
        if "@sha256:" not in ref:
            notes.append(f"{name}: no image digest declared (TENSORPILOT_IMAGE_REF)")
    if not checked:
        return Item("6.pinned-and-reproducible", 6, title, "reproducibility", INCONCLUSIVE, "no stage with a manifest found")
    status = FAIL if bad else INCONCLUSIVE if notes else PASS
    return Item("6.pinned-and-reproducible", 6, title, "reproducibility", status,
                "; ".join((bad + notes)[:4]) or f"{checked} stages ran the committed config, a pinned revision and a declared image")


def scan_archive(path, literal_secrets=()):
    """Findings inside a tar.gz: credential shapes and any literal secret supplied by the caller."""
    findings = []
    literals = [s.encode() for s in literal_secrets if s]
    with tarfile.open(path) as tar:
        for member in tar:
            if not member.isfile() or member.size > 256 * 1024**2:
                continue
            data = tar.extractfile(member).read()
            for label, pattern in SECRET_PATTERNS.items():
                if pattern.search(data):
                    findings.append(f"{member.name}: {label}")
            if any(secret in data for secret in literals):
                findings.append(f"{member.name}: the Pod API key")
    return findings


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_archives(paths, literal_secrets=()):
    title = "Evidence archives are SHA-256 verified and secret-free"
    archives = [p for p in paths if str(p).endswith(".tar.gz")]
    if not archives:
        return Item("6.archives", 6, title, "reproducibility", INCONCLUSIVE, "no .tar.gz archives were supplied (--archives)")
    problems = []
    for archive in archives:
        recorded = Path(str(archive) + ".sha256")
        if not recorded.exists():
            problems.append(f"{Path(archive).name}: no .sha256 file")
        elif recorded.read_text().split()[0] != sha256_file(archive):
            problems.append(f"{Path(archive).name}: SHA-256 does not match")
        try:
            problems += [f"{Path(archive).name}: {f}" for f in scan_archive(archive, literal_secrets)]
        except (tarfile.TarError, OSError) as exc:
            problems.append(f"{Path(archive).name}: unreadable ({exc})")
    return Item("6.archives", 6, title, "reproducibility", FAIL if problems else PASS,
                "; ".join(problems[:4]) or f"{len(archives)} archives verified and scanned", archives=len(archives))


# ----- 7. budget ------------------------------------------------------------------------------------------------

def read_ledgers(paths):
    entries = []
    for path in paths:
        entries += [e for e in _jsonl(path) if "stage" in e]
    return entries


def check_caps(entries, criteria):
    title = "Every stage stopped within its hard cap"
    if not entries:
        return Item("7.stage-caps-respected", 7, title, "budget", INCONCLUSIVE, "no ledger was supplied (--ledger)")
    grace = criteria["budget"]["stage_cap_grace_minutes"]
    over = [f"{e['stage']}: {e['elapsed_min']} min against a {e['cap_min']} min cap" for e in entries
            if e["elapsed_min"] > e["cap_min"] + grace]
    return Item("7.stage-caps-respected", 7, title, "budget", FAIL if over else PASS,
                "; ".join(over) or f"{len(entries)} stage runs, none beyond cap + {grace} min",
                stage_minutes=round(sum(e["elapsed_min"] for e in entries), 1))


def check_spend(entries, spent_usd, criteria):
    title = "Total spend stayed within the approved budget"
    limit = criteria["budget"]["usd"]
    if spent_usd is None:
        minutes = sum(e["elapsed_min"] for e in entries)
        return Item("7.within-budget", 7, title, "budget", INCONCLUSIVE,
                    f"pass --spent-usd from the Runpod billing page. The ledger shows {minutes:.0f} stage minutes, but boot, "
                    "downloads, loads and idle time are not in it", stage_minutes=minutes)
    return Item("7.within-budget", 7, title, "budget", PASS if spent_usd <= limit else FAIL,
                f"spent ${spent_usd:.2f} of ${limit:.2f}", spent_usd=spent_usd)


# ----- assembling the verdict -----------------------------------------------------------------------------------

def evaluate(session_dir, *, criteria=None, ledgers=(), archives=(), report=None, spent_usd=None, secrets=()):
    criteria = criteria or load_criteria()
    root = Path(session_dir)
    session = report_v10.load_session(root)
    items = [
        check_loads(session, "7b-bf16-prod", "7B BF16"),
        check_loads(session, "14b-nf4-prod", "14B NF4"),
        check_protocol(session, criteria),
        check_integrity(session),
        check_exactness(session, "7b-bf16-prod", "probe-7b", "1.exact-match-7b-bf16",
                        "7B BF16 batched output equals solo output, or every divergence is explained", root),
        check_exactness(session, "0.5b-quick-batch", "probe-0.5b", "1.quick-session-divergence-explained",
                        "The quick session's 31/32 divergence is explained with logits", root),
        *([check_exactness(session, "0.5b-quick-batch-packed", "probe-0.5b", "1.packed-session-divergence-explained",
                           "The packed run's divergences are explained with logits", root)]
          if "0.5b-quick-batch-packed" in session else []),
        check_nf4_7b(session, criteria),
        check_nf4_14b(session, criteria),
        check_levels(session, "7b-bf16-prod", "7b", criteria),
        check_levels(session, "14b-nf4-prod", "14b", criteria),
        check_volume(session, criteria),
        check_saturation(session, "7b-bf16-prod", "7b"),
        check_saturation(session, "14b-nf4-prod", "14b"),
        check_speedup(session, criteria),
        check_sustained(session, criteria),
        check_oom(session, criteria),
        check_overload(session, criteria),
        check_crash(session, criteria),
        check_slo_declared(session, criteria),
        check_operating_point(session, "7b-bf16-prod", "7b", criteria),
        check_operating_point(session, "14b-nf4-prod", "14b", criteria),
        check_failures_visible(session),
        *check_vllm(session, criteria),
        check_gap_explained(report, criteria),
        check_telemetry(session, criteria),
        check_reproducible(session),
        check_archives([Path(a) for a in archives], secrets),
    ]
    entries = read_ledgers([Path(p) for p in ledgers])
    items += [check_caps(entries, criteria), check_spend(entries, spent_usd, criteria)]
    rows = [i.as_dict() for i in items]
    gate = [i for i in items if i.category != PERFORMANCE]
    perf = [i for i in items if i.category == PERFORMANCE]
    return {"schema": SCHEMA, "at_utc": datetime.now(timezone.utc).isoformat(),
            "criteria_sha256": hashlib.sha256(json.dumps(criteria, sort_keys=True).encode()).hexdigest(),
            "evaluation": worst(i.status for i in gate), "performance": worst(i.status for i in perf),
            "counts": {s: sum(i.status == s for i in items) for s in (PASS, FAIL, INCONCLUSIVE)},
            "rule": "PASS only if every gate criterion passes; any gate FAIL is FAIL; otherwise missing evidence is INCONCLUSIVE. "
                    "Performance targets are reported separately and never change the evaluation verdict.",
            "items": rows}


def render_markdown(verdict):
    lines = ["# TensorPilot production-scale acceptance", "",
             f"**Evaluation: {verdict['evaluation']}**  ·  **Performance targets: {verdict['performance']}**", "",
             verdict["rule"], "", "| # | Criterion | Group | Verdict | Evidence |", "|---|---|---|---|---|"]
    for row in verdict["items"]:
        detail = row["detail"].replace("|", "/").replace("\n", " ")
        lines.append(f"| {row['criterion']} | {row['title']} | {row['category']} | **{row['status']}** | {detail} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("session_dir")
    parser.add_argument("--criteria", default=None)
    parser.add_argument("--ledger", action="append", default=[])
    parser.add_argument("--archives", nargs="*", default=[], help="archive files, or a directory holding them")
    parser.add_argument("--report", help="the written technical report (for the vLLM gap analysis)")
    parser.add_argument("--spent-usd", type=float)
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown")
    args = parser.parse_args()
    archives = []
    for entry in args.archives:
        path = Path(entry)
        archives += sorted(path.glob("*.tar.gz")) if path.is_dir() else [path]
    secrets = [os.environ.get("TENSORPILOT_API_KEY", "")]
    verdict = evaluate(args.session_dir, criteria=load_criteria(args.criteria), ledgers=args.ledger, archives=archives,
                       report=args.report, spent_usd=args.spent_usd, secrets=secrets)
    Path(args.output).write_text(json.dumps(verdict, indent=2) + "\n")
    if args.markdown:
        Path(args.markdown).write_text(render_markdown(verdict))
    for row in verdict["items"]:
        print(f"{row['status']:<13}{row['id']:<46}{row['detail'][:110]}")
    print(f"\nEVALUATION {verdict['evaluation']}   PERFORMANCE {verdict['performance']}")
    return {PASS: 0, FAIL: 1, INCONCLUSIVE: 2}[verdict["evaluation"]]


if __name__ == "__main__":
    sys.exit(main())
