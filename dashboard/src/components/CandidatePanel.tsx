import { useState } from "react";
import { api } from "../api";
import { bytes, ms, num } from "../format";
import { useAsync } from "../hooks";
import { CodeView, DiffView } from "./CodeView";
import { Histogram } from "./charts";

export function CandidatePanel({ id }: { id: string }) {
  const { data: c, error } = useAsync(() => api.candidate(id), [id]);
  const [view, setView] = useState<"diff" | "source">("diff");
  if (error) return <div className="panel pad err">{error}</div>;
  if (!c) return <div className="panel pad sub">loading…</div>;
  const md = c.metadata ?? {};
  const prof = c.profile?.ok ? c.profile.metrics : null;
  const hasParent = !!c.parent_source;

  return (
    <div className="panel pad">
      <div className="row" style={{ justifyContent: "space-between" }}>
        <div>
          <div className="mono" style={{ color: "var(--muted)" }}>{c.id} · {c.problem}</div>
          <div style={{ fontWeight: 600 }}>{c.label}</div>
        </div>
        <span className="badge">{c.status}</span>
      </div>

      {c.compile_error && <><h2>Compiler error</h2><pre className="errtext">{c.compile_error}</pre></>}

      {c.validation && (
        <>
          <h2>Correctness</h2>
          {c.validation.error && <div className="err">{c.validation.error}</div>}
          <table>
            <thead><tr><th>shape</th><th>result</th><th className="r">max abs</th><th className="r">max rel</th><th>tol (atol / rtol)</th><th>notes</th></tr></thead>
            <tbody>
              {c.validation.checks.map((k) => (
                <tr key={k.problem}>
                  <td className="mono">{k.problem}</td>
                  <td className={`st ${k.ok ? "ok" : "inv"}`}>{k.ok ? "pass" : "FAIL"}</td>
                  <td className="r num">{num(k.max_abs_err)}</td>
                  <td className="r num">{num(k.max_rel_err)}</td>
                  <td className="num">{num(k.atol, 2)} / {num(k.rtol, 2)}</td>
                  <td>{k.message}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}

      {c.samples && c.samples.length > 0 && (
        <>
          <h2>Latency ({c.samples.length} reps)</h2>
          <div className="row"><span>p50 <b className="num">{ms(c.p50_ms)}</b></span><span>p95 <b className="num">{ms(c.p95_ms)}</b></span>
            <span className="sub">{c.bench_method}</span></div>
          <Histogram samples={c.samples} />
        </>
      )}

      {(Object.keys(md).length > 0 || prof) && <>
      <h2>Resources</h2>
      <div className="kpis" style={{ margin: 0 }}>
        <div className="kpi"><b>{String(md.n_regs ?? prof?.registers_per_thread ?? "–")}</b><span>regs / thread</span></div>
        <div className="kpi"><b style={{ color: md.n_spills ? "var(--red)" : undefined }}>{String(md.n_spills ?? "–")}</b><span>spilled regs</span></div>
        <div className="kpi"><b>{bytes(md.shared ?? prof?.shared_mem_per_block)}</b><span>shared mem</span></div>
        <div className="kpi"><b>{prof?.achieved_occupancy_pct != null ? `${prof.achieved_occupancy_pct.toFixed(0)}%` :
          typeof md.est_occupancy === "number" ? `${(md.est_occupancy * 100).toFixed(0)}%*` : "–"}</b><span>occupancy</span></div>
        {prof?.dram_throughput_pct != null && <div className="kpi"><b>{prof.dram_throughput_pct.toFixed(0)}%</b><span>DRAM of peak</span></div>}
        {prof?.sm_throughput_pct != null && <div className="kpi"><b>{prof.sm_throughput_pct.toFixed(0)}%</b><span>SM of peak</span></div>}
      </div>
      {c.profile?.simulated && <div className="sub" style={{ marginTop: 6 }}>profiler numbers are simulated · * = static estimate</div>}
      </>}

      {c.ptx_summary && (
        <>
          <h2>PTX</h2>
          <ul>{((c.ptx_summary.observations as string[]) ?? []).map((o) => <li key={o}>{o}</li>)}</ul>
        </>
      )}

      <h2>Source</h2>
      {hasParent && (
        <div className="row" style={{ marginBottom: 8 }}>
          <button className={`btn ${view === "diff" ? "primary" : ""}`} onClick={() => setView("diff")}>diff vs {c.parent_label}</button>
          <button className={`btn ${view === "source" ? "primary" : ""}`} onClick={() => setView("source")}>full source</button>
        </div>
      )}
      {hasParent && view === "diff" ? <DiffView before={c.parent_source!} after={c.source} /> : <CodeView source={c.source} />}
    </div>
  );
}
