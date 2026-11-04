"""Experiment reports: JSON (for tools), Markdown (for humans/PRs), HTML (standalone)."""

from __future__ import annotations

import html
import json
from datetime import datetime
from typing import Any

import numpy as np

from kernelpilot.analysis.compare import best_so_far
from kernelpilot.store import Store

SIM_BANNER = (
    "SIMULATED RESULTS: no GPU was used. Latencies and profiler counters come from "
    "KernelPilot's analytic model; numerical checks were emulated on the CPU. "
    "Do not quote these numbers as hardware performance."
)

SPARK = "▁▂▃▄▅▆▇█"


def _fmt_ms(v: float | None) -> str:
    if v is None:
        return "-"
    if v < 0.1:
        return f"{v * 1000:.1f} us"
    return f"{v:.3f} ms"


def _fmt_x(v: float | None) -> str:
    return "-" if v is None else f"{v:.2f}x"


def sparkline(samples: list[float], bins: int = 16) -> str:
    if not samples:
        return ""
    counts, _ = np.histogram(samples, bins=bins)
    top = counts.max() or 1
    return "".join(SPARK[min(len(SPARK) - 1, int(c / top * (len(SPARK) - 1)))] for c in counts)


def _resources(c: dict[str, Any]) -> dict[str, Any]:
    md = c.get("metadata") or {}
    prof = (c.get("profile") or {}).get("metrics", {})
    return {
        "registers": md.get("n_regs") or prof.get("registers_per_thread"),
        "spills": md.get("n_spills"),
        "shared_bytes": md.get("shared")
        if md.get("shared") is not None
        else prof.get("shared_mem_per_block"),
        "occupancy_pct": prof.get("achieved_occupancy_pct")
        or (round(md["est_occupancy"] * 100, 1) if md.get("est_occupancy") is not None else None),
        "dram_pct": prof.get("dram_throughput_pct"),
        "occupancy_source": "profiler"
        if prof.get("achieved_occupancy_pct")
        else ("estimate" if md.get("est_occupancy") is not None else None),
    }


def _latency(c: dict[str, Any] | None) -> dict[str, Any] | None:
    if not c or not c.get("samples"):
        return None
    s = c["samples"]
    return {
        "n": len(s),
        "p50": c["p50_ms"],
        "p95": c["p95_ms"],
        "min": c["min_ms"],
        "max": c["max_ms"],
        "std": c["std_ms"],
        "spark": sparkline(s),
        "samples": s,
    }


def build_report(store: Store, exp_id: str) -> dict[str, Any]:
    exp = store.experiment(exp_id)
    if exp is None:
        raise KeyError(f"no experiment {exp_id!r}")
    summary = exp.get("summary") or {}
    problems = []
    for ps in summary.get("problems", []):
        cands = store.candidates(exp["id"], ps["problem"])
        by_id = {c["id"]: c for c in cands}
        best = by_id.get(ps.get("best_id"))
        base = by_id.get(ps.get("baseline_id"))
        problems.append(
            {
                **ps,
                "baseline": {
                    "label": base["label"],
                    "latency": _latency(base),
                    "resources": _resources(base),
                }
                if base
                else None,
                "best": {
                    "label": best["label"],
                    "latency": _latency(best),
                    "resources": _resources(best),
                    "origin": best["origin"],
                    "note": best["note"],
                }
                if best
                else None,
                "compile_failures": [
                    {
                        "id": c["id"],
                        "label": c["label"],
                        "kind": c["compile_error_kind"],
                        "error": c["compile_error"],
                    }
                    for c in cands
                    if c["status"] == "compile_failed"
                ],
                "validation_failures": [
                    {
                        "id": c["id"],
                        "label": c["label"],
                        "status": c["valid_status"],
                        "detail": _validation_detail(c.get("validation")),
                    }
                    for c in cands
                    if c["status"] in ("invalid", "unverified")
                ],
                "candidates": [
                    {
                        "id": c["id"],
                        "seq": c["seq"],
                        "label": c["label"],
                        "origin": c["origin"],
                        "status": c["status"],
                        "p50_ms": c["p50_ms"],
                        "p95_ms": c["p95_ms"],
                        "speedup": c["speedup"],
                        "compile_ms": c["compile_ms"],
                        "retained": bool(c["retained"]),
                        **_resources(c),
                    }
                    for c in cands
                ],
                "curve": best_so_far(cands),
            }
        )
    decisions = [
        {
            "step": d["step"],
            "kind": d["kind"],
            "problem": d["problem"],
            "tool": d["tool"],
            "rationale": d["rationale"],
            "candidate_id": d["candidate_id"],
            "evidence": d["evidence"],
        }
        for d in store.decisions(exp["id"])
        if d["kind"] not in ("trial", "tool_call")
    ]
    env = exp.get("environment") or {}
    return {
        "experiment": {
            k: exp[k]
            for k in ("id", "name", "strategy", "status", "backend", "created_at", "finished_at")
        },
        "simulated": bool(exp["simulated"]),
        "params": exp.get("params"),
        "fingerprint": exp.get("fingerprint"),
        "environment": {
            k: env.get(k)
            for k in (
                "python",
                "platform",
                "torch",
                "triton",
                "gpu_name",
                "compute_capability",
                "driver",
                "cuda_runtime",
                "ncu_version",
            )
        },
        "summary": {k: v for k, v in summary.items() if k != "problems"},
        "problems": problems,
        "decisions": decisions,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }


def _validation_detail(v: dict[str, Any] | None) -> str:
    if not v:
        return ""
    if v.get("error"):
        return v["error"]
    bad = [c for c in v.get("checks", []) if not c["ok"]]
    if not bad:
        return ""
    c = max(bad, key=lambda c: c["max_abs_err"])
    extra = f", {c['message']}" if c.get("message") else ""
    return f"{c['problem']}: max_abs={c['max_abs_err']:.3g} (atol {c['atol']:.2g}, rtol {c['rtol']:.2g}){extra}"


def to_json(report: dict[str, Any]) -> str:
    return json.dumps(report, indent=2, default=str)


# -- markdown ------------------------------------------------------------------


def to_markdown(r: dict[str, Any]) -> str:
    exp, s = r["experiment"], r["summary"]
    out = [f"# KernelPilot report: {exp['name']}", ""]
    if r["simulated"]:
        out += [f"> **{SIM_BANNER}**", ""]
    gm = s.get("geomean_speedup")
    out += [
        f"- experiment: `{exp['id']}` ({exp['strategy']}, status {exp['status']})",
        f"- backend: {s.get('backend', exp['backend'])}",
        f"- geomean speedup vs untuned Triton baseline: **{_fmt_x(gm)}** over "
        f"{s.get('geomean_over', 0)}/{s.get('problems_total', 0)} problems (correctness-verified candidates only)",
        f"- candidates evaluated: {s.get('candidates')}, compile failures: {s.get('compile_failed')}, "
        f"numerical failures: {s.get('invalid')}, unverified: {s.get('unverified')}",
        f"- wall time: {s.get('wall_s')} s",
    ]
    cache = s.get("cache")
    if cache:
        out.append(
            f"- cache: {cache['session_hits']} hits / {cache['session_misses']} misses this run"
        )
    if r.get("params"):
        p = r["params"]
        out.append(
            f"- params: seed={p.get('seed')} reps={p.get('reps')} warmup={p.get('warmup')}"
            + (f" budget={p['budget']}" if "budget" in p else "")
        )
    out.append("")

    out += [
        "## Results",
        "",
        "| problem | baseline p50 | best p50 | speedup | best config | evals | to-best | failed |",
        "|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    for p in r["problems"]:
        best = p.get("best") or {}
        out.append(
            f"| `{p['problem']}` | {_fmt_ms(p.get('baseline_p50'))} | {_fmt_ms(p.get('best_p50'))} | "
            f"{_fmt_x(p.get('speedup'))} | `{best.get('label', '-')}` | {p['evaluated']} | "
            f"{p.get('evals_to_best') or '-'} | {p['compile_failed'] + p['invalid']} |"
        )
    out.append("")

    out += [
        "## Latency distributions",
        "",
        "| problem | which | n | p50 | p95 | min | max | histogram |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for p in r["problems"]:
        for which in ("baseline", "best"):
            lat = (p.get(which) or {}).get("latency")
            if lat:
                out.append(
                    f"| `{p['problem']}` | {which} | {lat['n']} | {_fmt_ms(lat['p50'])} | "
                    f"{_fmt_ms(lat['p95'])} | {_fmt_ms(lat['min'])} | {_fmt_ms(lat['max'])} | `{lat['spark']}` |"
                )
    out.append("")

    out += [
        "## Resource usage (best candidate)",
        "",
        "| problem | regs/thread | spills | shared mem | occupancy | DRAM % of peak |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for p in r["problems"]:
        res = (p.get("best") or {}).get("resources")
        if not res:
            continue
        occ = res.get("occupancy_pct")
        occ_s = "-" if occ is None else f"{occ:.0f}% ({res['occupancy_source']})"
        dram = res.get("dram_pct")
        out.append(
            f"| `{p['problem']}` | {res.get('registers') or '-'} | {res.get('spills') if res.get('spills') is not None else '-'} | "
            f"{res.get('shared_bytes') if res.get('shared_bytes') is not None else '-'} | {occ_s} | "
            f"{'-' if dram is None else f'{dram:.0f}%'} |"
        )
    out.append("")

    failures = [(p["problem"], f) for p in r["problems"] for f in p["compile_failures"]]
    out += ["## Compilation failures", ""]
    if failures:
        for prob, f in failures:
            first = (f["error"] or "").splitlines()[0][:220]
            out.append(f"- `{prob}` `{f['label']}` ({f['kind']}): {first}")
    else:
        out.append("None.")
    out.append("")

    vfail = [(p["problem"], f) for p in r["problems"] for f in p["validation_failures"]]
    out += ["## Numerical validation failures", ""]
    if vfail:
        for prob, f in vfail:
            out.append(f"- `{prob}` `{f['label']}` ({f['status']}): {f['detail']}")
    else:
        out.append("None.")
    out.append("")

    if r["decisions"]:
        out += ["## Optimization decisions", ""]
        for d in r["decisions"]:
            if not d.get("rationale"):
                continue
            tag = d["kind"]
            cand = f" [`{d['candidate_id']}`]" if d.get("candidate_id") else ""
            out.append(f"{d['step']}. **{tag}**{cand}: {d['rationale']}")
        out.append("")

    out += ["## Reproduce", "", "```bash"]
    params = r.get("params") or {}
    out.append(
        f"KP_SEED={params.get('seed', 0)} KP_REPS={params.get('reps', 50)} "
        f"KP_WARMUP={params.get('warmup', 10)} kernelpilot ...  # see experiment params above"
    )
    out += ["```", "", f"Environment fingerprint: `{json.dumps(r.get('fingerprint'))}`", ""]
    return "\n".join(out)


# -- html ----------------------------------------------------------------------

CSS = """
:root{--bg:#f7f7f4;--panel:#fff;--ink:#1d1f21;--muted:#6b6f73;--line:#e3e3de;--accent:#2f6f5e;
--bad:#b4442c;--warn:#a87b16;--bar:#3c8d76;--bar2:#c98a3a}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#121416;--panel:#1a1d20;--ink:#e6e6e3;
--muted:#9a9ea3;--line:#2b2f33;--accent:#6cc3a8;--bad:#e2795f;--warn:#e0b04a;--bar:#58b597;--bar2:#e0a35a}}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 ui-sans-serif,system-ui,sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:32px 0 8px;border-bottom:1px solid var(--line);padding-bottom:4px}
.banner{background:var(--warn);color:#111;padding:10px 14px;border-radius:6px;font-weight:600;margin:12px 0}
.kpis{display:flex;flex-wrap:wrap;gap:12px;margin:16px 0}.kpi{background:var(--panel);border:1px solid var(--line);
border-radius:8px;padding:10px 14px;min-width:140px}.kpi b{display:block;font-size:20px}.kpi span{color:var(--muted);font-size:12px}
table{border-collapse:collapse;width:100%;background:var(--panel);border:1px solid var(--line);font-size:13px}
th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}code{font-family:ui-monospace,Menlo,monospace;font-size:12px}
.wrap{overflow-x:auto}.bad{color:var(--bad)}.muted{color:var(--muted)}svg text{fill:var(--muted);font-size:11px}
"""


def _svg_bars(rows: list[tuple[str, float | None]], width: int = 680) -> str:
    rows = [(k, v) for k, v in rows if v]
    if not rows:
        return ""
    top = max(v for _, v in rows) * 1.1
    h = 22 * len(rows) + 30
    label_w = 220
    scale = (width - label_w - 60) / top
    parts = [
        f'<svg viewBox="0 0 {width} {h}" width="100%" role="img" aria-label="speedup per problem">'
    ]
    one_x = label_w + scale * 1.0
    for i, (k, v) in enumerate(rows):
        y = 10 + i * 22
        parts.append(f'<text x="0" y="{y + 13}">{html.escape(k)}</text>')
        parts.append(
            f'<rect x="{label_w}" y="{y}" width="{scale * v:.1f}" height="16" rx="2" fill="var(--bar)"/>'
        )
        parts.append(f'<text x="{label_w + scale * v + 6:.1f}" y="{y + 13}">{v:.2f}x</text>')
    parts.append(
        f'<line x1="{one_x:.1f}" x2="{one_x:.1f}" y1="4" y2="{h - 20}" stroke="var(--ink)" stroke-dasharray="3 3"/>'
    )
    parts.append(f'<text x="{one_x - 8:.1f}" y="{h - 6}">1x</text></svg>')
    return "".join(parts)


def _svg_curve(curve: list[dict[str, Any]], width: int = 320, height: int = 120) -> str:
    pts = [(c["n"], c["speedup"]) for c in curve if c["speedup"]]
    if len(pts) < 2:
        return ""
    n_max = max(p[0] for p in pts)
    top = max(p[1] for p in pts) * 1.08
    lo = min(1.0, min(p[1] for p in pts)) * 0.95

    def xy(n, s):
        return (
            30 + (n - 1) / max(1, n_max - 1) * (width - 40),
            height - 18 - (s - lo) / (top - lo) * (height - 30),
        )

    d = " ".join(
        f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y) in enumerate(xy(*p) for p in pts)
    )
    _x1, y1 = xy(1, 1.0)
    return (
        f'<svg viewBox="0 0 {width} {height}" width="{width}" role="img" aria-label="best-so-far speedup">'
        f'<line x1="30" x2="{width - 10}" y1="{y1:.1f}" y2="{y1:.1f}" stroke="var(--line)"/>'
        f'<path d="{d}" fill="none" stroke="var(--accent)" stroke-width="2"/>'
        f'<text x="30" y="{height - 4}">eval 1</text><text x="{width - 50}" y="{height - 4}">eval {n_max}</text>'
        f'<text x="0" y="12">{top / 1.08:.2f}x</text></svg>'
    )


def to_html(r: dict[str, Any]) -> str:
    e = html.escape
    exp, s = r["experiment"], r["summary"]
    body = [
        f"<h1>{e(exp['name'])}</h1><div class=muted>experiment <code>{e(exp['id'])}</code> · "
        f"{e(exp['strategy'])} · {e(str(s.get('backend', exp['backend'])))}</div>"
    ]
    if r["simulated"]:
        body.append(f'<div class="banner">{e(SIM_BANNER)}</div>')
    kpis = [
        ("geomean speedup", _fmt_x(s.get("geomean_speedup"))),
        ("problems improved", f"{s.get('geomean_over', 0)}/{s.get('problems_total', 0)}"),
        ("candidates", str(s.get("candidates"))),
        ("compile failures", str(s.get("compile_failed"))),
        ("numerical failures", str(s.get("invalid"))),
    ]
    body.append(
        '<div class="kpis">'
        + "".join(f'<div class="kpi"><b>{e(v)}</b><span>{e(k)}</span></div>' for k, v in kpis)
        + "</div>"
    )
    body.append("<h2>Speedup over untuned Triton baseline</h2>")
    body.append(_svg_bars([(p["problem"], p.get("speedup")) for p in r["problems"]]))

    body.append(
        "<h2>Per-problem results</h2><div class=wrap><table><tr><th>problem</th><th class=n>baseline p50</th>"
        "<th class=n>best p50</th><th class=n>speedup</th><th>best config</th><th>latency (best)</th>"
        "<th>search progress</th></tr>"
    )
    for p in r["problems"]:
        best = p.get("best") or {}
        lat = best.get("latency") or {}
        body.append(
            f"<tr><td><code>{e(p['problem'])}</code></td><td class=n>{_fmt_ms(p.get('baseline_p50'))}</td>"
            f"<td class=n>{_fmt_ms(p.get('best_p50'))}</td><td class=n><b>{_fmt_x(p.get('speedup'))}</b></td>"
            f"<td><code>{e(best.get('label', '-'))}</code></td>"
            f"<td><code>{e(lat.get('spark', ''))}</code><br><span class=muted>p95 {_fmt_ms(lat.get('p95'))}</span></td>"
            f"<td>{_svg_curve(p['curve'])}</td></tr>"
        )
    body.append("</table></div>")

    body.append(
        "<h2>Failures</h2><div class=wrap><table><tr><th>problem</th><th>candidate</th><th>stage</th><th>detail</th></tr>"
    )
    any_fail = False
    for p in r["problems"]:
        for f in p["compile_failures"]:
            any_fail = True
            body.append(
                f"<tr><td><code>{e(p['problem'])}</code></td><td><code>{e(f['label'])}</code></td>"
                f"<td class=bad>compile ({e(str(f['kind']))})</td><td><code>{e((f['error'] or '')[:300])}</code></td></tr>"
            )
        for f in p["validation_failures"]:
            any_fail = True
            body.append(
                f"<tr><td><code>{e(p['problem'])}</code></td><td><code>{e(f['label'])}</code></td>"
                f"<td class=bad>{e(f['status'])}</td><td>{e(f['detail'])}</td></tr>"
            )
    if not any_fail:
        body.append("<tr><td colspan=4 class=muted>none</td></tr>")
    body.append("</table></div>")

    if r["decisions"]:
        body.append("<h2>Decisions</h2><ol>")
        for d in r["decisions"]:
            if d.get("rationale"):
                body.append(f"<li><b>{e(d['kind'])}</b> {e(d['rationale'])}</li>")
        body.append("</ol>")
    body.append(
        f"<h2>Environment</h2><pre><code>{e(json.dumps(r['environment'], indent=2))}</code></pre>"
    )
    return (
        f"<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
        f"<title>KernelPilot report</title><style>{CSS}</style></head><body><main>{''.join(body)}</main></body></html>"
    )
