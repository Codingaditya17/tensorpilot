"""Turn v1.0 stage evidence into a written report with charts. It reads evidence and invents nothing.

    python -m tensorpilot.report_v10 SESSION_DIR --output report/ [--baseline benchmarks/baselines/v0.9.1-4-rtx4090]

SESSION_DIR holds the stage evidence directories that ``scale_session.sh`` wrote (and its ledger). The report states, for
every planned experiment, whether it completed, was aborted (and why), was skipped, or was never run, so a gap is visible
instead of silently absent. A figure appears only if the evidence contains it; a comparison appears only if both sides
were run. Points built from fewer than 100 completed requests are marked small-sample (hollow markers, a dagger in tables).

Charts are static SVG with light and dark palettes. Series colors follow the entity (an engine or a model keeps its color in
every chart); the first three validated categorical slots are used, so no chart carries more than three series. Every chart
has its numbers in a table next to it, which also covers the one light-mode color (aqua) below 3:1 contrast.
"""
import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path
from xml.sax.saxutils import escape

from . import suite

SMALL = 100
STAGE_DIR = re.compile(r"^(?P<stage>.+)-(?P<stamp>\d{8}T\d{6}Z)$")
ENGINE_COLORS = {"tensorpilot-batch": 1, "tensorpilot-serial": 2, "vllm": 3}
MODEL_COLORS = {"0.5b": 1, "7b": 2, "14b": 3}
OBJECTIVE = {"attainment": 0.95, "error_rate": 0.01}
PAIRS_ENGINE = (("7b-bf16-batch", "vllm-7b-bf16"), ("14b-nf4-batch", "vllm-14b-nf4"))
PAIRS_SERIAL = (("0.5b-serial", "0.5b-batch", {"serial-c8": "sweep-c8", "regression-c1": "sweep-c1"}),
                ("7b-bf16-serial", "7b-bf16-batch", {"serial-c8": "sweep-c8", "serial-c1": "sweep-c1"}))

SVG_STYLE = ("<style>:root{--surface:#fcfcfb;--text:#0b0b0b;--muted:#52514e;--grid:#e3e2de;--s1:#2a78d6;--s2:#eb6834;"
             "--s3:#1baf7a}@media (prefers-color-scheme: dark){:root{--surface:#1a1a19;--text:#f0efec;--muted:#c3c2b7;"
             "--grid:#34332f;--s1:#3987e5;--s2:#d95926;--s3:#199e70}}text{font:12px system-ui,sans-serif;fill:var(--text)}"
             ".m{fill:var(--muted)}.t{font-size:14px;font-weight:600}</style>")


# ----- loading evidence -------------------------------------------------------------------------------------

def _json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def load_session(root):
    """{stage: evidence} for the newest directory of each stage found under ``root``."""
    root = Path(root)
    ledger = {}
    for line in (root / "ledger.jsonl").read_text().splitlines() if (root / "ledger.jsonl").exists() else []:
        entry = json.loads(line)
        ledger.setdefault(entry["stage"], []).append(entry)
    found = {}
    for directory in sorted(p for p in root.iterdir() if p.is_dir()):
        match = STAGE_DIR.match(directory.name)
        if match:
            found[match["stage"]] = directory  # sorted by name, so the newest stamp wins
    stages = {}
    for stage, directory in found.items():
        reports = {}
        bench = directory / "bench"
        for experiment_dir in sorted(p for p in bench.iterdir() if p.is_dir()) if bench.exists() else []:
            files = sorted(f for f in experiment_dir.glob("repeat*.json") if not f.name.endswith(".workload.json"))
            reports[experiment_dir.name] = [_json(f) for f in files if _json(f)]
        stages[stage] = {"dir": directory, "manifest": _json(bench / "manifest.json"), "reports": reports,
                         "invariance": _json(directory / "invariance.json"),
                         "comparison": _json(directory / "comparison.json"),
                         "fault": _json(directory / "fault" / "fault-report.json"),
                         "preflight": _json(directory / "preflight.json"),
                         "ledger": (ledger.get(stage) or [None])[-1]}
    return stages


# ----- aggregation ------------------------------------------------------------------------------------------

def spread(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {"median": statistics.median(values), "min": min(values), "max": max(values), "n": len(values)}


def completed(reports):
    return sum(r["summary"]["successful_requests"] for r in reports)


def smallest(reports):
    """Completed requests in the smallest repeat. Percentiles are computed per repeat, so this decides small-sample."""
    return min((r["summary"]["successful_requests"] for r in reports), default=0)


def point(reports, key):
    return spread([r["summary"].get(key) for r in reports])


def nested(reports, *path):
    values = []
    for report in reports:
        node = report
        for step in path:
            node = (node or {}).get(step) if isinstance(node, dict) else None
        values.append(node)
    return spread(values)


def sweep_rows(stage):
    rows = []
    for name, reports in stage["reports"].items():
        match = re.fullmatch(r"sweep-c(\d+)", name)
        if match and reports:
            rows.append((int(match[1]), reports))
    rows.sort()
    out = []
    for concurrency, reports in rows:
        gpu = [r.get("telemetry", {}).get("gpu", {}).get("0", {}) for r in reports]
        out.append({"concurrency": concurrency, "repeats": len(reports), "completed": completed(reports),
                    "small": smallest(reports) < SMALL,
                    "tok_s_wall": point(reports, "completed_output_tokens_per_s"),
                    "tok_s_active": point(reports, "active_window_output_tokens_per_s"),
                    "req_s": point(reports, "successful_requests_per_s"),
                    "ttft_p50": point(reports, "p50_client_ttft_s"), "ttft_p95": point(reports, "p95_client_ttft_s"),
                    "ttft_p99": point(reports, "p99_client_ttft_s"), "e2e_p50": point(reports, "p50_client_latency_s"),
                    "e2e_p95": point(reports, "p95_client_latency_s"), "e2e_p99": point(reports, "p99_client_latency_s"),
                    "tpot_p50": point(reports, "p50_client_tpot_s"), "tpot_p95": point(reports, "p95_client_tpot_s"),
                    "itl_p95": point(reports, "p95_client_inter_token_s"), "error_rate": point(reports, "error_rate"),
                    "rejected": sum(r["summary"]["rejected_requests"] for r in reports),
                    "timed_out": sum(r["summary"]["timed_out_requests"] for r in reports),
                    "goodput_req_s": spread([(r["summary"].get("slo") or {}).get("goodput_requests_per_s") for r in reports]),
                    "attainment": spread([(r["summary"].get("slo") or {}).get("attainment") for r in reports]),
                    "gpu_util_mean": spread([(g.get("utilization_gpu_pct") or {}).get("mean") for g in gpu]),
                    "gpu_mem_max_mib": spread([(g.get("memory_used_mib") or {}).get("max") for g in gpu]),
                    "power_mean_w": spread([(g.get("power_w") or {}).get("mean") for g in gpu]),
                    "batch_width_mean": nested(reports, "telemetry", "active_sequences", "mean"),
                    "decode_tok_s": spread([r.get("telemetry", {}).get("active_decode_tokens_per_s") for r in reports])})
    return out


def ladder_rows(stage, name="ladder-poisson"):
    rows = []
    for report in stage["reports"].get(name, []):
        s = report["summary"]
        slo = s.get("slo") or {}
        rows.append({"rate": report["experiment"]["rate_per_s"], "completed": s["successful_requests"],
                     "small": s["successful_requests"] < SMALL, "tok_s": s["completed_output_tokens_per_s"],
                     "ttft_p95": s.get("p95_client_ttft_s"), "e2e_p95": s.get("p95_client_latency_s"),
                     "error_rate": s["error_rate"], "attainment": slo.get("attainment"),
                     "goodput": slo.get("goodput_requests_per_s")})
    return sorted(rows, key=lambda r: r["rate"])


def max_sustainable(rows):
    """Highest offered rate such that it and every lower rate met the declared objective; None if none did."""
    best = None
    for row in rows:
        if row["attainment"] is not None and row["attainment"] >= OBJECTIVE["attainment"] and \
                (row["error_rate"] or 0) <= OBJECTIVE["error_rate"]:
            best = row["rate"]
        else:
            break
    return best


# ----- charts -----------------------------------------------------------------------------------------------

def _ticks(top):
    """Round axis values from zero: the smallest nice step that needs at most six intervals to cover ``top``."""
    if top <= 0:
        return [0, 1]
    base = 10 ** math.floor(math.log10(top))
    step = next(base * factor for factor in (0.1, 0.2, 0.25, 0.5, 1, 2, 5, 10) if top / (base * factor) <= 6)
    return [round(i * step, 10) for i in range(int(top / step) + 2)]


def line_chart(path, *, title, x_label, y_label, series, log_x=True, note=""):
    """series: [{name, slot, points: [(x, y, small)]}]. One y-axis; never two scales on one plot."""
    width, height, left, right, top, bottom = 760, 420, 76, 170, 56, 62
    xs = sorted({x for s in series for x, _, _ in s["points"]})
    ys = [y for s in series if not s.get("reference") for _, y, _ in s["points"] if y is not None]
    if not xs or not ys:
        return False
    top_value = _ticks(max(ys) * 1.08)[-1]
    lo, hi = (math.log(min(xs)), math.log(max(xs))) if log_x and len(xs) > 1 else (min(xs), max(xs))

    def px(x):
        value = math.log(x) if log_x and len(xs) > 1 else x
        return left + (width - left - right) * ((value - lo) / (hi - lo) if hi > lo else 0.5)

    def py(y):
        return height - bottom - (height - top - bottom) * (y / top_value)

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
           f'role="img" aria-labelledby="t d">{SVG_STYLE}<title id="t">{escape(title)}</title>',
           f'<desc id="d">{escape(y_label)} against {escape(x_label)} for {", ".join(escape(s["name"]) for s in series)}. '
           f'The same numbers are in the table below the chart.</desc>',
           f'<rect width="{width}" height="{height}" fill="var(--surface)"/>',
           f'<text class="t" x="{left}" y="24">{escape(title)}</text>',
           f'<text class="m" x="{left}" y="42">{escape(note)}</text>']
    for tick in _ticks(top_value):
        if tick <= top_value:
            out.append(f'<line x1="{left}" x2="{width - right}" y1="{py(tick):.1f}" y2="{py(tick):.1f}" '
                       'stroke="var(--grid)" stroke-width="1"/>'
                       f'<text class="m" x="{left - 8}" y="{py(tick) + 4:.1f}" text-anchor="end">{tick:g}</text>')
    for x in xs:
        out.append(f'<text class="m" x="{px(x):.1f}" y="{height - bottom + 18}" text-anchor="middle">{x:g}</text>')
    out.append(f'<text class="m" x="{(left + width - right) / 2:.0f}" y="{height - 14}" text-anchor="middle">'
               f'{escape(x_label)}</text><text class="m" x="14" y="{(top + height - bottom) / 2:.0f}" '
               f'transform="rotate(-90 14 {(top + height - bottom) / 2:.0f})" text-anchor="middle">{escape(y_label)}</text>')
    for index, s in enumerate(series):
        if s.get("reference"):
            # A reference line (not data): dashed, muted, no markers, so it cannot be mistaken for a series.
            points = [(px(x), py(y)) for x, y, _ in s["points"] if y is not None and y <= top_value]
            out.append('<polyline fill="none" stroke="var(--muted)" stroke-width="1.5" stroke-dasharray="6 4" points="%s"/>'
                       % " ".join(f"{x:.1f},{y:.1f}" for x, y in points))
            out.append(f'<line x1="{width - right + 14}" x2="{width - right + 34}" y1="{top + 8 + 18 * index}" '
                       f'y2="{top + 8 + 18 * index}" stroke="var(--muted)" stroke-width="1.5" stroke-dasharray="6 4"/>'
                       f'<text class="m" x="{width - right + 40}" y="{top + 12 + 18 * index}">{escape(s["name"])}</text>')
            continue
        color = f"var(--s{s['slot']})"
        points = [(px(x), py(y), small) for x, y, small in s["points"] if y is not None]
        if len(points) > 1:
            out.append('<polyline fill="none" stroke="%s" stroke-width="2" points="%s"/>'
                       % (color, " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in points)))
        for x, y, small in points:
            fill = "var(--surface)" if small else color
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{fill}" stroke="{color}" stroke-width="2"/>')
        if points:
            x, y, _ = points[-1]
            out.append(f'<text x="{x + 10:.1f}" y="{y + 4:.1f}">{escape(s["name"])}</text>')
        out.append(f'<line x1="{width - right + 14}" x2="{width - right + 34}" y1="{top + 8 + 18 * index}" '
                   f'y2="{top + 8 + 18 * index}" stroke="{color}" stroke-width="2"/>'
                   f'<text x="{width - right + 40}" y="{top + 12 + 18 * index}">{escape(s["name"])}</text>')
    out.append("</svg>")
    Path(path).write_text("\n".join(out))
    return True


# ----- markdown helpers -------------------------------------------------------------------------------------

def fmt(value, digits=2, unit=""):
    return "n/a" if value is None else f"{value:.{digits}f}{unit}"


def cell(spread_value, digits=2, small=False):
    if spread_value is None:
        return "n/a"
    text = fmt(spread_value["median"], digits)
    if spread_value["n"] > 1:
        text += f" ({fmt(spread_value['min'], digits)}–{fmt(spread_value['max'], digits)})"
    return text + ("†" if small else "")


def table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return lines


def engine_key(stage):
    parameters = suite.STAGES.get(stage)
    if not parameters:
        return None
    if parameters["engine"] == "vllm":
        return "vllm"
    return "tensorpilot-serial" if parameters["execution"] == "serial" else "tensorpilot-batch"


def experiment_status(stage_name, evidence):
    """Every planned experiment with its status: completed, aborted(reason), skipped or not run."""
    planned = [e["id"] for e in suite.config_for(stage_name)["experiments"]] if stage_name in suite.STAGES else []
    manifest = (evidence["manifest"] or {}).get("experiments", {})
    rows = []
    for name in planned:
        entry = manifest.get(name)
        if entry is None:
            rows.append((name, "NOT RUN", "no record in the manifest"))
        elif entry["status"] == "completed":
            rows.append((name, "completed", f"{entry.get('elapsed_s')} s of {entry.get('cap_s')} s cap"))
        else:
            rows.append((name, entry["status"].upper(), entry.get("abort_reason") or ""))
    return rows


# ----- the report -------------------------------------------------------------------------------------------

def build(session, output, baseline=None):
    output = Path(output)
    charts = output / "charts"
    charts.mkdir(parents=True, exist_ok=True)
    stages = load_session(session)
    md = ["# TensorPilot v1.0 GPU evaluation report", "",
          "Generated from the stage evidence listed below. Nothing here is estimated: a figure appears only if the evidence "
          "contains it, and a comparison appears only if both sides were run. **†** marks a point built from fewer than "
          f"{SMALL} completed requests; its p95 and p99 are effectively the maximum. Hollow chart markers mean the same.", ""]
    if not stages:
        md += ["**No stage evidence was found.** Nothing was measured.", ""]
        (output / "report.md").write_text("\n".join(md))
        return {"stages": 0, "charts": 0}
    made = 0

    md += ["## Provenance", ""]
    rows = []
    for name, ev in sorted(stages.items()):
        target = (ev["manifest"] or {}).get("target") or {}
        server = (ev["manifest"] or {}).get("server_at_start") or {}
        pre = ev["preflight"] or {}
        gpu = ((pre.get("torch") or {}).get("devices") or [{}])[0].get("name") or "not recorded"
        driver = (((pre.get("telemetry") or {}).get("devices") or [{}])[0]).get("driver_version") or "not recorded"
        rows.append((name, target.get("model", "n/a"), (target.get("revision") or "n/a")[:10], target.get("engine", "n/a"),
                     target.get("quantization", "n/a"), target.get("dtype", "n/a"), gpu, driver,
                     (ev["ledger"] or {}).get("exit_status", "n/a"), (ev["ledger"] or {}).get("elapsed_min", "n/a"),
                     server.get("execution_mode", "n/a")))
    md += table(["stage", "model", "revision", "engine", "quantization", "dtype", "GPU", "driver", "exit", "minutes",
                 "mode"], rows) + [""]

    md += ["## Experiment status", "", "Every planned experiment is listed. Failed, aborted, skipped and never-run experiments "
           "are results too.", ""]
    for name, ev in sorted(stages.items()):
        status = experiment_status(name, ev)
        if status:
            md += [f"### {name}", ""] + table(["experiment", "status", "detail"], status) + [""]

    # concurrency sweeps
    sweeps = {name: sweep_rows(ev) for name, ev in stages.items()}
    sweeps = {name: rows for name, rows in sweeps.items() if rows}
    for name, rows in sorted(sweeps.items()):
        md += [f"## Concurrency sweep: {name}", "",
               "Closed loop, mixed prompt and output lengths; values are the median over repeats with (min–max).", ""]
        md += table(["clients", "repeats", "completed", "output tok/s (wall)", "tok/s (active window)", "TTFT p50 s",
                     "TTFT p95 s", "TTFT p99 s", "e2e p50 s", "e2e p95 s", "e2e p99 s", "TPOT p95 s", "error rate",
                     "rejected", "timed out", "goodput req/s", "SLO attainment", "GPU util % (mean)",
                     "GPU mem max MiB", "power W", "mean batch width", "decode tok/s (GPU-active)"],
                    [(r["concurrency"], r["repeats"], f"{r['completed']}{'†' if r['small'] else ''}",
                      cell(r["tok_s_wall"], 1), cell(r["tok_s_active"], 1), cell(r["ttft_p50"], 3), cell(r["ttft_p95"], 3),
                      cell(r["ttft_p99"], 3), cell(r["e2e_p50"], 2), cell(r["e2e_p95"], 2), cell(r["e2e_p99"], 2),
                      cell(r["tpot_p95"], 4), cell(r["error_rate"], 3), r["rejected"], r["timed_out"],
                      cell(r["goodput_req_s"], 2), cell(r["attainment"], 3), cell(r["gpu_util_mean"], 0),
                      cell(r["gpu_mem_max_mib"], 0), cell(r["power_mean_w"], 0), cell(r["batch_width_mean"], 1),
                      cell(r["decode_tok_s"], 1)) for r in rows]) + [""]
        slot = ENGINE_COLORS.get(engine_key(name), 1)
        for key, label, file in (("tok_s_wall", "Output tokens/s (wall clock)", "throughput"),
                                 ("ttft_p95", "TTFT p95 (s)", "ttft"), ("e2e_p95", "End-to-end latency p95 (s)", "latency"),
                                 ("goodput_req_s", "Goodput under the declared SLO (requests/s)", "goodput")):
            series = [{"name": name, "slot": slot, "points": [(r["concurrency"], (r[key] or {}).get("median"), r["small"])
                                                               for r in rows]}]
            if key == "tok_s_wall":
                series.append({"name": "active window", "slot": 2, "points": [
                    (r["concurrency"], (r["tok_s_active"] or {}).get("median"), r["small"]) for r in rows]})
            if line_chart(charts / f"{file}-{name}.svg", title=f"{label}: {name}", x_label="concurrent clients",
                          y_label=label, series=series, note="hollow marker = fewer than 100 completed requests"):
                md += [f"![{label}, {name}](charts/{file}-{name}.svg)", ""]
                made += 1

    # ladders
    for name, ev in sorted(stages.items()):
        rows = ladder_rows(ev)
        if rows:
            best = max_sustainable(rows)
            md += [f"## Open-loop rate ladder: {name}", "",
                   f"Objective: SLO attainment ≥ {OBJECTIVE['attainment']:.0%} and error rate ≤ {OBJECTIVE['error_rate']:.0%}. "
                   f"**Maximum sustainable rate under it: "
                   f"{'none of the offered rates met it' if best is None else f'{best:g} requests/s'}.**", ""]
            md += table(["offered req/s", "completed", "output tok/s", "TTFT p95 s", "e2e p95 s", "error rate",
                         "SLO attainment", "goodput req/s"],
                        [(r["rate"], f"{r['completed']}{'†' if r['small'] else ''}", fmt(r["tok_s"], 1),
                          fmt(r["ttft_p95"], 3), fmt(r["e2e_p95"], 2), fmt(r["error_rate"], 3), fmt(r["attainment"], 3),
                          fmt(r["goodput"], 2)) for r in rows]) + [""]
            series = [{"name": "goodput", "slot": 1, "points": [(r["rate"], r["goodput"], r["small"]) for r in rows]},
                      {"name": "ideal: every offered request meets the SLO", "reference": True,
                       "points": [(r["rate"], r["rate"], False) for r in rows]}]
            if line_chart(charts / f"ladder-{name}.svg", title=f"Goodput against offered rate: {name}",
                          x_label="offered requests/s", y_label="goodput (requests/s)", series=series,
                          note="the dashed line is the ideal; goodput falls below it as the SLO is missed or requests fail"):
                md += [f"![Goodput against offered rate, {name}](charts/ladder-{name}.svg)", ""]
                made += 1

    # serial against batch, same model
    for serial, batch, mapping in PAIRS_SERIAL:
        if serial in stages and batch in stages:
            md += [f"## Serial against continuous batching: {serial} vs {batch}", "",
                   "The same model on the same card; only the execution mode differs.", ""]
            rows = []
            for s_name, b_name in mapping.items():
                s_reports, b_reports = stages[serial]["reports"].get(s_name), stages[batch]["reports"].get(b_name)
                if s_reports and b_reports:
                    s_tok, b_tok = point(s_reports, "completed_output_tokens_per_s"), point(b_reports, "completed_output_tokens_per_s")
                    s_p95, b_p95 = point(s_reports, "p95_client_latency_s"), point(b_reports, "p95_client_latency_s")
                    small = smallest(s_reports) < SMALL or smallest(b_reports) < SMALL
                    rows.append((f"{s_name} vs {b_name}", cell(s_tok, 1), cell(b_tok, 1),
                                 fmt(b_tok["median"] / s_tok["median"], 2, "×") if s_tok and s_tok["median"] else "n/a",
                                 cell(s_p95, 2), cell(b_p95, 2) + ("†" if small else "")))
            md += (table(["pair", "serial tok/s", "batch tok/s", "ratio", "serial e2e p95 s", "batch e2e p95 s"], rows)
                   if rows else ["No matched pair of experiments was run."]) + [""]

    # engines
    for ours, theirs in PAIRS_ENGINE:
        if ours in stages and theirs in stages:
            a, b = sweeps.get(ours, []), {r["concurrency"]: r for r in sweeps.get(theirs, [])}
            md += [f"## TensorPilot against vLLM: {ours} vs {theirs}", "",
                   "Identical workloads (asserted by the suite's tests), same card, never run at the same time. These "
                   "are different engines on different CUDA toolkits (torch 2.9.1 with CUDA 12.8 against torch 2.13.0 with "
                   "CUDA 13.0), so differences are not attributable to one cause. The ratio is vLLM ÷ TensorPilot.", ""]
            rows = []
            for r in a:
                v = b.get(r["concurrency"])
                if not v:
                    continue
                ratio = (v["tok_s_wall"]["median"] / r["tok_s_wall"]["median"]
                         if v["tok_s_wall"] and r["tok_s_wall"] and r["tok_s_wall"]["median"] else None)
                rows.append((r["concurrency"], cell(r["tok_s_wall"], 1), cell(v["tok_s_wall"], 1), fmt(ratio, 2, "×"),
                             cell(r["ttft_p95"], 3), cell(v["ttft_p95"], 3), cell(r["e2e_p95"], 2), cell(v["e2e_p95"], 2),
                             cell(r["goodput_req_s"], 2), cell(v["goodput_req_s"], 2)))
            md += table(["clients", "TP tok/s", "vLLM tok/s", "vLLM÷TP", "TP TTFT p95", "vLLM TTFT p95", "TP e2e p95",
                         "vLLM e2e p95", "TP goodput", "vLLM goodput"], rows) + [""]
            for key, label, file in (("tok_s_wall", "Output tokens/s", "engines-throughput"),
                                     ("e2e_p95", "End-to-end latency p95 (s)", "engines-latency")):
                series = [{"name": "TensorPilot", "slot": 1, "points": [(r["concurrency"], (r[key] or {}).get("median"),
                                                                         r["small"]) for r in a]},
                          {"name": "vLLM", "slot": 3, "points": [(c, (v[key] or {}).get("median"), v["small"])
                                                                  for c, v in sorted(b.items())]}]
                if line_chart(charts / f"{file}-{ours}.svg", title=f"{label}: {ours} vs {theirs}",
                              x_label="concurrent clients", y_label=label, series=series,
                              note="same workload, same card, run one after the other"):
                    md += [f"![{label}, {ours} vs vLLM](charts/{file}-{ours}.svg)", ""]
                    made += 1
            mismatch = ((stages[theirs]["reports"] or {}).get("sweep-c1") or [{}])[0].get("limitations")
            if mismatch:
                md += ["Comparability notes recorded by the vLLM run:", ""] + [f"- {line}" for line in mismatch] + [""]

    # model scaling at fixed concurrency
    scaling = {}
    for name, rows in sweeps.items():
        key = suite.STAGES.get(name, {}).get("model")
        if key and engine_key(name) == "tensorpilot-batch" and "long" not in name:
            scaling[key] = rows
    if len(scaling) >= 2:
        md += ["## Scaling with model size (TensorPilot, continuous batching)", ""]
        rows = []
        for concurrency in (1, 8, 32):
            for model in ("0.5b", "7b", "14b"):
                r = next((x for x in scaling.get(model, []) if x["concurrency"] == concurrency), None)
                if r:
                    rows.append((concurrency, model, cell(r["tok_s_wall"], 1), cell(r["ttft_p95"], 3), cell(r["e2e_p95"], 2),
                                 cell(r["gpu_mem_max_mib"], 0)))
        md += table(["clients", "model", "tok/s", "TTFT p95 s", "e2e p95 s", "GPU mem max MiB"], rows) + [""]
        series = [{"name": model, "slot": MODEL_COLORS[model],
                   "points": [(r["concurrency"], (r["tok_s_wall"] or {}).get("median"), r["small"]) for r in scaling[model]]}
                  for model in ("0.5b", "7b", "14b") if model in scaling]
        if line_chart(charts / "scaling-by-model.svg", title="Output tokens/s by model size", x_label="concurrent clients",
                      y_label="output tokens/s", series=series, note="14B is NF4; 0.5B and 7B are 16-bit"):
            md += ["![Output tokens/s by model size](charts/scaling-by-model.svg)", ""]
            made += 1

    # overload, cancellation, timeouts
    robust = []
    for name, ev in sorted(stages.items()):
        for exp in ("cancellation", "timeouts", "overload", "recovery-after-overload"):
            for report in ev["reports"].get(exp, []):
                s, d = report["summary"], report.get("server_counters_delta") or {}
                robust.append((name, exp, s["requests_total"], json.dumps(s["outcomes"]), fmt(s["error_rate"], 3),
                               s["rejected_requests"], s["timed_out_requests"], s["client_cancelled_requests"],
                               json.dumps({k.replace("tensorpilot_", ""): v for k, v in d.items()}) or "n/a",
                               (report.get("telemetry") or {}).get("cuda_oom_events_delta")))
    if robust:
        md += ["## Cancellation, timeouts and overload", "",
               "Client outcomes beside the server's own counters, so a request that one side counts and the other does not is "
               "visible. Rejections are expected under overload; an OOM event is never expected.", ""]
        md += table(["stage", "experiment", "requests", "client outcomes", "error rate", "rejected", "timed out",
                     "cancelled by client", "server counter deltas", "OOM events"], robust) + [""]

    for name, ev in sorted(stages.items()):
        fault = ev["fault"]
        if fault:
            md += [f"## Worker crash and recovery: {name}", "",
                   f"SIGKILL under load, supervised restart, then an audit of the request journal. "
                   f"**Overall: {'PASSED' if fault['passed'] else 'FAILED'}.**", ""]
            md += table(["check", "result", "detail"], [(c["name"], "pass" if c["passed"] else "FAIL", c["detail"])
                                                         for c in fault["checks"]]) + [""]
            states = fault.get("readiness_timeline") or []
            if states:
                md += ["Readiness timeline (seconds on the load run's clock): " +
                       ", ".join(f"{s['offset_s']:.1f} s {s['state']}" for s in states), ""]

    for name, ev in sorted(stages.items()):
        inv = ev["invariance"]
        if inv:
            md += [f"## Batch invariance: {name}", "",
                   "Does a request's output depend on what else is in the batch? Each request is compared with its solo run. "
                   "Divergence late in a sequence is a numeric near-tie; divergence at the start is a defect.", ""]
            md += table(["comparison", "identical", "mean prefix", "earliest divergence", "median divergence"],
                        [(k.replace("_", " "), f"{inv[k]['exact_matches']}/{inv[k]['comparable']}",
                          fmt(inv[k]["mean_prefix_fraction"], 3), inv[k]["earliest_divergence"],
                          inv[k]["median_divergence"]) for k in ("together_vs_solo", "staggered_vs_solo")]) + [""]

    for name, ev in sorted(stages.items()):
        cmp = ev["comparison"]
        if cmp:
            md += [f"## NF4 fidelity against bf16: {name}", "",
                   "A 14B model has no bf16 reference on a 24 GB card; this 7B pair is the closest available evidence about "
                   "NF4 fidelity, not a substitute for measuring 14B.", ""]
            md += table(["metric", "value"], [("positions compared", cmp["positions_compared"]),
                                              ("perplexity bf16 → NF4", f"{fmt(cmp['perplexity_reference'], 3)} → {fmt(cmp['perplexity_test'], 3)}"),
                                              ("perplexity ratio", fmt(cmp["perplexity_ratio_test_over_reference"], 4)),
                                              ("top-1 agreement", fmt(cmp["top1_agreement"], 4)),
                                              ("top-5 overlap", fmt(cmp["top5_overlap"], 4)),
                                              ("KL (nats, top-k support)", fmt(cmp["kl_reference_to_test_nats_topk_support"], 5)),
                                              ("greedy prefix agreement", fmt(cmp["greedy_mean_prefix_agreement"], 3))]) + [""]

    overhead = {}
    for name, ev in stages.items():
        for flag in ("on", "off"):
            reports = ev["reports"].get(f"overhead-telemetry-{flag}")
            if reports:
                overhead.setdefault(name, {})[flag] = reports
    for name, pair in sorted(overhead.items()):
        if len(pair) == 2:
            on, off = point(pair["on"], "completed_output_tokens_per_s"), point(pair["off"], "completed_output_tokens_per_s")
            overlap = on["min"] <= off["max"] and off["min"] <= on["max"]
            md += [f"## Telemetry overhead: {name}", "",
                   f"Identical workload with the sampler on and off. Throughput on: {cell(on, 1)} tok/s; off: {cell(off, 1)} "
                   f"tok/s; difference {fmt((on['median'] / off['median'] - 1) * 100, 2, '%')} (median). "
                   + ("The ranges overlap, so the difference is not distinguishable from run-to-run variation." if overlap
                      else "The ranges do not overlap."), ""]

    if baseline:
        base = _json(Path(baseline) / "summary.json")
        ev = stages.get("0.5b-serial")
        if base and ev and ev["reports"].get("regression-c1"):
            now = point(ev["reports"]["regression-c1"], "completed_output_tokens_per_s")
            was = base["experiments"]["first-closed-c1"]["output_tokens_per_s_wall"]
            md += ["## Regression against the immutable v0.9.1-4 baseline", "",
                   f"Same workload shape, serial worker, one RTX 4090. Baseline {was} tok/s; now {cell(now, 1)} tok/s "
                   f"({fmt(now['median'] / was, 3, '×')} of baseline).", ""]

    md += ["## Limitations recorded by the runs", ""]
    seen = []
    for ev in stages.values():
        for line in (ev["manifest"] or {}).get("limitations", []):
            if line not in seen:
                seen.append(line)
    md += [f"- {line}" for line in seen] + [""]
    md += ["## What this report does not establish", "",
           "- Anything not listed under Experiment status as *completed* was not measured.",
           "- SLO thresholds are declared in the suite, not derived from user requirements.",
           "- Hypotheses about causes (for example NF4 dequantization overhead, or the Python event loop at high batch width) "
           "are not measurements; the telemetry and counters in this report are what would test them.", ""]
    (output / "report.md").write_text("\n".join(md))
    (output / "tables.json").write_text(json.dumps({n: sweeps.get(n, []) for n in stages}, indent=2, default=str))
    return {"stages": len(stages), "charts": made}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("session", help="directory of stage evidence and the ledger")
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline", help="the immutable baseline directory to compare against")
    args = parser.parse_args()
    result = build(args.session, args.output, args.baseline)
    print(f"report: {Path(args.output) / 'report.md'} ({result['stages']} stage(s), {result['charts']} chart(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
