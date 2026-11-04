import { useEffect, useState } from "react";
import { api } from "../api";
import { ago, x } from "../format";
import { useAsync } from "../hooks";
import { go } from "../router";

const OPS = ["softmax", "matmul", "reduction", "vector_add"];

function NewRun({ onStarted }: { onStarted: () => void }) {
  const [target, setTarget] = useState("op:softmax");
  const [shape, setShape] = useState("");
  const [dtype, setDtype] = useState("");
  const [strategy, setStrategy] = useState("agent");
  const [budget, setBudget] = useState(8);
  const [job, setJob] = useState<{ id: string; status: string; error: string | null } | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    if (!job || job.status !== "running") return;
    const t = setInterval(async () => {
      const j = await api.job(job.id);
      setJob(j);
      onStarted();
      if (j.status === "done" && j.experiment_id) go(`/exp/${j.experiment_id}`);
    }, 800);
    return () => clearInterval(t);
  }, [job, onStarted]);

  const start = async () => {
    setErr(null);
    const [kind, value] = target.split(":");
    const body: Record<string, unknown> = { strategy, budget };
    if (kind === "suite") body.suite = value;
    else {
      body.op = value;
      if (shape.trim()) body.shape = shape.split("x").map(Number);
      if (dtype) body.dtype = dtype;
    }
    try {
      setJob(await api.startRun(body));
      onStarted();
    } catch (e) {
      setErr(String(e));
    }
  };

  return (
    <div className="panel pad">
      <div className="row" style={{ alignItems: "flex-end" }}>
        <label className="f">target
          <select value={target} onChange={(e) => setTarget(e.target.value)}>
            {OPS.map((o) => <option key={o} value={`op:${o}`}>{o}</option>)}
            <option value="suite:quick">suite: quick (4)</option>
            <option value="suite:standard">suite: standard (13)</option>
          </select>
        </label>
        {target.startsWith("op:") && <>
          <label className="f">shape <input placeholder="default" value={shape} onChange={(e) => setShape(e.target.value)} size={12} /></label>
          <label className="f">dtype <select value={dtype} onChange={(e) => setDtype(e.target.value)}>
            <option value="">default</option><option>fp16</option><option>fp32</option></select></label>
        </>}
        <label className="f">strategy <select value={strategy} onChange={(e) => setStrategy(e.target.value)}>
          {["agent", "grid", "random", "halving"].map((s) => <option key={s}>{s}</option>)}</select></label>
        <label className="f">budget / problem <input type="number" min={1} max={256} value={budget} onChange={(e) => setBudget(+e.target.value)} style={{ width: 80 }} /></label>
        <button className="btn primary" onClick={start} disabled={job?.status === "running"}>
          {job?.status === "running" ? "running…" : "start run"}</button>
      </div>
      {(err || job?.error) && <div className="err" style={{ marginTop: 8 }}>{err ?? job?.error}</div>}
    </div>
  );
}

export function ExperimentList() {
  const { data, error, reload } = useAsync(() => api.experiments(), []);
  return (
    <>
      <h1>Experiments</h1>
      <div className="sub">Every candidate, measurement and decision is stored in SQLite and traceable from here.</div>
      <h2>New run</h2>
      <NewRun onStarted={reload} />
      <h2>History</h2>
      {error && <div className="err">{error}</div>}
      <div className="panel scroll">
        <table>
          <thead><tr><th>experiment</th><th>strategy</th><th>target</th><th className="r">geomean</th><th className="r">candidates</th>
            <th className="r">failed</th><th>backend</th><th>when</th></tr></thead>
          <tbody>
            {data?.map((e) => (
              <tr key={e.id} className="click" onClick={() => go(`/exp/${e.id}`)}>
                <td><div>{e.name}</div><div className="mono sub">{e.id}</div></td>
                <td><span className="chip">{e.strategy}{e.params?.planner ? ` · ${String(e.params.planner)}` : ""}</span>
                  {e.status !== "done" && <span className={`st ${e.status === "running" ? "amb" : "bad"}`}> {e.status}</span>}</td>
                <td className="mono">{e.target.length === 1 ? e.target[0] : `${e.target.length} problems`}</td>
                <td className="r num" style={{ color: (e.geomean_speedup ?? 0) > 1.01 ? "var(--teal)" : undefined }}>{x(e.geomean_speedup, 3)}</td>
                <td className="r num">{e.candidates ?? "–"}</td>
                <td className="r num">{(e.compile_failed ?? 0) + (e.invalid ?? 0)}</td>
                <td>{e.simulated ? <span className="badge sim">SIM</span> : <span className="badge real">{e.backend}</span>}</td>
                <td className="sub">{ago(e.created_at)}</td>
              </tr>
            ))}
            {data && !data.length && <tr><td colSpan={8} className="sub">No experiments yet. Start one above, or run <code>kernelpilot demo</code>.</td></tr>}
          </tbody>
        </table>
      </div>
    </>
  );
}
