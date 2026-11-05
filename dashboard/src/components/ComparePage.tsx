import { api } from "../api";
import { x } from "../format";
import { useAsync } from "../hooks";
import { go } from "../router";
import { Curves } from "./charts";

export function ComparePage({ a, b }: { a: string | null; b: string | null }) {
  const list = useAsync(() => api.experiments(), []);
  const cmp = useAsync(() => (a && b ? api.compare(a, b) : Promise.resolve(null)), [a, b]);
  const done = (list.data ?? []).filter((e) => e.status !== "running");
  const pick = (which: "a" | "b", id: string) => {
    const q = new URLSearchParams({ a: which === "a" ? id : a ?? "", b: which === "b" ? id : b ?? "" });
    go(`/compare?${q}`);
  };
  const c = cmp.data;
  return (
    <>
      <h1>Agent vs autotuner</h1>
      <div className="sub">Same problems, same candidate budget. Who gets to a fast, correct kernel, and how many evaluations does it take?</div>
      <div className="panel pad row" style={{ marginTop: 14 }}>
        {(["a", "b"] as const).map((w) => (
          <label className="f" key={w}>{w === "a" ? "experiment A" : "experiment B"}
            <select value={(w === "a" ? a : b) ?? ""} onChange={(e) => pick(w, e.target.value)}>
              <option value="">choose…</option>
              {done.map((e) => <option key={e.id} value={e.id}>{e.strategy} · {e.name} · {e.id}</option>)}
            </select>
          </label>
        ))}
      </div>
      {cmp.error && <div className="err">{cmp.error}</div>}
      {c && (
        <>
          {c.mixed_backends && <div className="banner"><b>Mixed backends.</b> One run is simulated and the other isn't; the numbers aren't comparable.</div>}
          {(c.a.simulated || c.b.simulated) && !c.mixed_backends && <div className="banner"><b>Simulated.</b> Both runs used the simulated backend.</div>}
          <div className="kpis">
            <div className="kpi"><b style={{ color: "var(--amber)" }}>{x(c.a_geomean_shared, 3)}</b><span>{c.a.strategy} geomean</span></div>
            <div className="kpi"><b style={{ color: "var(--blue)" }}>{x(c.b_geomean_shared, 3)}</b><span>{c.b.strategy} geomean</span></div>
            <div className="kpi"><b>{c.a_wins} – {c.b_wins}</b><span>wins ({c.ties} ties)</span></div>
            <div className="kpi"><b>{c.shared_problems}</b><span>shared problems</span></div>
          </div>
          <div className="panel scroll">
            <table>
              <thead><tr><th>problem</th><th className="r">{c.a.strategy}</th><th className="r">{c.b.strategy}</th>
                <th className="r">evals to best</th><th className="r">failures</th><th>winner</th></tr></thead>
              <tbody>{c.rows.map((r) => (
                <tr key={r.problem}>
                  <td className="mono">{r.problem}</td>
                  <td className="r num">{x(r.a_speedup)}</td><td className="r num">{x(r.b_speedup)}</td>
                  <td className="r num">{r.a_evals_to_best ?? "–"} / {r.b_evals_to_best ?? "–"}</td>
                  <td className="r num">{r.a_failures} / {r.b_failures}</td>
                  <td>{r.winner === c.a.id ? c.a.strategy : r.winner === c.b.id ? c.b.strategy : r.winner ?? "–"}</td>
                </tr>))}
              </tbody>
            </table>
          </div>
          <h2>Best-so-far speedup per evaluation</h2>
          <div className="legend"><span><i style={{ background: "var(--amber)" }} />{c.a.strategy} ({c.a.id})</span>
            <span><i style={{ background: "var(--blue)" }} />{c.b.strategy} ({c.b.id})</span></div>
          <div className="grid2" style={{ marginTop: 8 }}>
            {c.rows.map((r) => (
              <div key={r.problem} className="panel pad">
                <div className="mono">{r.problem}</div>
                <Curves series={[{ name: "a", color: "var(--amber)", points: r.a_curve }, { name: "b", color: "var(--blue)", points: r.b_curve }]} height={160} />
              </div>
            ))}
          </div>
        </>
      )}
      {!c && !cmp.loading && <div className="sub" style={{ marginTop: 12 }}>Pick two experiments. <code>tritune bakeoff softmax</code> creates a matched pair.</div>}
    </>
  );
}
