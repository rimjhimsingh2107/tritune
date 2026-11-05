import { useMemo, useState } from "react";
import { api } from "../api";
import { bytes, ms, num, x } from "../format";
import { useAsync } from "../hooks";
import type { Experiment, Trace, TraceEntry } from "../types";
import { CandidatePanel } from "./CandidatePanel";
import { Bars, Curves, Histogram } from "./charts";
import { TracePanel } from "./TracePanel";

const TABS = ["trace", "candidates", "performance", "resources", "failures", "report"] as const;
type Tab = (typeof TABS)[number];

function curveOf(entries: TraceEntry[]) {
  const base = entries.find((e) => e.origin === "baseline" && e.benchmark.p50_ms != null);
  let best: number | null = null;
  return entries.map((e, i) => {
    if (e.status === "measured" && e.benchmark.p50_ms != null && (best == null || e.benchmark.p50_ms < best)) best = e.benchmark.p50_ms;
    return { n: i + 1, best_p50_ms: best, speedup: base && best ? base.benchmark.p50_ms! / best : null };
  });
}

function Candidates({ trace }: { trace: Trace }) {
  const [sort, setSort] = useState<"seq" | "p50" | "speedup">("seq");
  const [sel, setSel] = useState<string | null>(null);
  const rows = useMemo(() => {
    const r = [...trace.entries];
    if (sort === "p50") r.sort((a, b) => (a.benchmark.p50_ms ?? 1e9) - (b.benchmark.p50_ms ?? 1e9));
    if (sort === "speedup") r.sort((a, b) => (b.speedup ?? -1) - (a.speedup ?? -1));
    return r;
  }, [trace, sort]);
  return (
    <div className="split">
      <div className="panel scroll">
        <table>
          <thead><tr>
            <th className="sortable" onClick={() => setSort("seq")}>#</th><th>problem</th><th>config</th><th>origin</th><th>status</th>
            <th className="r sortable" onClick={() => setSort("p50")}>p50 ↕</th><th className="r">p95</th>
            <th className="r sortable" onClick={() => setSort("speedup")}>speedup ↕</th>
          </tr></thead>
          <tbody>
            {rows.map((e) => (
              <tr key={e.id} className="click" onClick={() => setSel(e.id)} style={sel === e.id ? { outline: "1px solid var(--amber)" } : undefined}>
                <td className="num">{e.seq}{e.retained ? " ★" : ""}</td>
                <td className="mono">{e.problem}</td>
                <td className="mono">{e.label}</td>
                <td>{e.origin}</td>
                <td className={`st ${e.status === "measured" ? "ok" : e.status === "compile_failed" ? "bad" : "inv"}`}>{e.status}</td>
                <td className="r num">{ms(e.benchmark.p50_ms)}</td>
                <td className="r num">{ms(e.benchmark.p95_ms)}</td>
                <td className="r num">{x(e.speedup)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="sticky">{sel ? <CandidatePanel id={sel} /> : <div className="panel pad sub">select a row</div>}</div>
    </div>
  );
}

function Performance({ exp, trace }: { exp: Experiment; trace: Trace }) {
  const probs = exp.summary?.problems ?? [];
  return (
    <>
      <h2>Speedup vs untuned Triton baseline (verified candidates only)</h2>
      <div className="panel pad"><Bars rows={probs.map((p) => ({ label: p.problem, value: p.speedup }))} /></div>
      <h2>Best-so-far by evaluation</h2>
      <div className="grid2">
        {probs.map((p) => (
          <div className="panel pad" key={p.problem}>
            <div className="mono" style={{ marginBottom: 4 }}>{p.problem}</div>
            <Curves series={[{ name: exp.strategy, color: "var(--amber)", points: curveOf(trace.entries.filter((e) => e.problem === p.problem)) }]} height={150} />
          </div>
        ))}
      </div>
      <h2>Latency distributions: baseline vs best</h2>
      <div className="grid2">
        {probs.map((p) => {
          const base = trace.entries.find((e) => e.id === p.baseline_id);
          const best = trace.entries.find((e) => e.id === p.best_id);
          return (
            <div className="panel pad" key={p.problem}>
              <div className="mono">{p.problem}</div>
              {base?.benchmark.samples && <><div className="sub">baseline · p50 {ms(base.benchmark.p50_ms)} · p95 {ms(base.benchmark.p95_ms)}</div><Histogram samples={base.benchmark.samples} height={70} /></>}
              {best && best.id !== base?.id && best.benchmark.samples && <><div className="sub">best · p50 {ms(best.benchmark.p50_ms)} · p95 {ms(best.benchmark.p95_ms)}</div><Histogram samples={best.benchmark.samples} height={70} /></>}
            </div>
          );
        })}
      </div>
    </>
  );
}

function Resources({ trace }: { trace: Trace }) {
  const rows = trace.entries.filter((e) => e.metadata && e.compile.ok);
  return (
    <div className="panel scroll">
      <table>
        <thead><tr><th>#</th><th>config</th><th className="r">regs</th><th className="r">spills</th><th className="r">shared</th>
          <th className="r">occupancy</th><th>limiter</th><th className="r">DRAM %</th><th className="r">SM %</th><th className="r">p50</th></tr></thead>
        <tbody>
          {rows.map((e) => {
            const md = e.metadata as Record<string, number | string | undefined>;
            const pr = e.profile ?? {};
            const occ = pr.achieved_occupancy_pct ?? (typeof md.est_occupancy === "number" ? md.est_occupancy * 100 : undefined);
            return (
              <tr key={e.id}>
                <td className="num">{e.seq}</td><td className="mono">{e.label}</td>
                <td className="r num">{String(md.n_regs ?? "–")}</td>
                <td className="r num" style={{ color: md.n_spills ? "var(--red)" : undefined }}>{String(md.n_spills ?? "–")}</td>
                <td className="r num">{bytes(md.shared)}</td>
                <td className="r num">{occ != null ? `${occ.toFixed(0)}%${pr.achieved_occupancy_pct == null ? "*" : ""}` : "–"}</td>
                <td>{String(md.occupancy_limiter ?? "–")}</td>
                <td className="r num">{pr.dram_throughput_pct != null ? pr.dram_throughput_pct.toFixed(0) : "–"}</td>
                <td className="r num">{pr.sm_throughput_pct != null ? pr.sm_throughput_pct.toFixed(0) : "–"}</td>
                <td className="r num">{ms(e.benchmark.p50_ms)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <div className="pad sub">* static estimate from the compiler metadata; profiler columns only appear for candidates that were profiled. Only metrics the active toolchain reports are shown.</div>
    </div>
  );
}

function Failures({ trace }: { trace: Trace }) {
  const comp = trace.entries.filter((e) => e.status === "compile_failed");
  const inv = trace.entries.filter((e) => e.status === "invalid" || e.status === "unverified");
  return (
    <>
      <h2>Compilation failures ({comp.length})</h2>
      {comp.map((e) => (
        <div key={e.id} className="panel pad" style={{ marginBottom: 10 }}>
          <div className="row"><span className="num">#{e.seq}</span><span className="mono">{e.label}</span><span className="st bad">{e.compile.kind}</span></div>
          {e.hypothesis && <div className="why">{e.hypothesis}</div>}
          <pre className="errtext">{e.compile.error}</pre>
        </div>
      ))}
      {!comp.length && <div className="sub">none</div>}
      <h2>Numerical validation failures ({inv.length})</h2>
      {inv.map((e) => {
        const w = e.validation.worst;
        return (
          <div key={e.id} className="panel pad" style={{ marginBottom: 10 }}>
            <div className="row"><span className="num">#{e.seq}</span><span className="mono">{e.label}</span><span className="st inv">{e.validation.status}</span></div>
            {e.hypothesis && <div className="why">{e.hypothesis}</div>}
            {e.validation.error && <div className="err">{e.validation.error}</div>}
            {w && <div className="mono">{w.problem}: max abs {num(w.max_abs_err)} · max rel {num(w.max_rel_err)} · {(w.mismatch_frac * 100).toFixed(2)}% mismatched · NaN {w.nan_count} · Inf {w.inf_count} (atol {num(w.atol, 2)}, rtol {num(w.rtol, 2)})</div>}
          </div>
        );
      })}
      {!inv.length && <div className="sub">none</div>}
    </>
  );
}

export function ExperimentView({ id }: { id: string }) {
  const exp = useAsync(() => api.experiment(id), [id]);
  const trace = useAsync(() => api.trace(id), [id]);
  const [tab, setTab] = useState<Tab>("trace");
  if (exp.error || trace.error) return <div className="err">{exp.error ?? trace.error}</div>;
  if (!exp.data || !trace.data) return <div className="sub">loading…</div>;
  const e = exp.data;
  const s = e.summary;
  return (
    <>
      <div className="row" style={{ justifyContent: "space-between" }}>
        <div>
          <h1>{e.name}</h1>
          <div className="sub mono">{e.id} · {e.strategy}{e.params?.planner ? ` (${String(e.params.planner)})` : ""} · {e.status}</div>
        </div>
        <div className="row">
          <a className="btn" href={`#/compare?a=${e.id}`}>compare…</a>
          <button className="btn" onClick={() => { exp.reload(); trace.reload(); }}>refresh</button>
        </div>
      </div>
      {e.simulated ? <div className="banner"><b>Simulated results.</b> No GPU was used: latencies and profiler counters come from TriTune's analytic model and numerics were emulated on the CPU. Useful for exercising the workflow, not for quoting performance.</div> : null}
      {s && (
        <div className="kpis">
          <div className="kpi"><b>{x(s.geomean_speedup, 3)}</b><span>geomean speedup ({s.geomean_over}/{s.problems_total})</span></div>
          <div className="kpi"><b>{s.candidates}</b><span>candidates</span></div>
          <div className="kpi"><b style={{ color: s.compile_failed ? "var(--red)" : undefined }}>{s.compile_failed}</b><span>compile failures</span></div>
          <div className="kpi"><b style={{ color: s.invalid ? "var(--magenta)" : undefined }}>{s.invalid}</b><span>numerical failures</span></div>
          <div className="kpi"><b>{trace.data.tool_calls || "–"}</b><span>agent tool calls</span></div>
          <div className="kpi"><b>{s.wall_s}s</b><span>wall time</span></div>
        </div>
      )}
      <div className="tabs">{TABS.map((t) => <button key={t} className={tab === t ? "on" : ""} onClick={() => setTab(t)}>{t}</button>)}</div>
      {tab === "trace" && <TracePanel trace={trace.data} />}
      {tab === "candidates" && <Candidates trace={trace.data} />}
      {tab === "performance" && <Performance exp={e} trace={trace.data} />}
      {tab === "resources" && <Resources trace={trace.data} />}
      {tab === "failures" && <Failures trace={trace.data} />}
      {tab === "report" && (
        <>
          <div className="row" style={{ marginBottom: 10 }}>
            {(["md", "html", "json"] as const).map((f) => <a key={f} className="btn" href={api.reportUrl(e.id, f)} download>download .{f}</a>)}
          </div>
          <iframe title="report" src={api.reportUrl(e.id, "html")} style={{ width: "100%", height: "75vh", border: "1px solid var(--line)", borderRadius: 8, background: "white" }} />
        </>
      )}
    </>
  );
}
