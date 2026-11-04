import { Fragment, useMemo, useState } from "react";
import type { Trace, TraceEntry } from "../types";
import { ms, num, shortProblem, x } from "../format";
import { CandidatePanel } from "./CandidatePanel";
import { TraceChart, TraceLegend, entryKind } from "./charts";

function changeChips(e: TraceEntry) {
  const ch = e.changes;
  if (!ch || !Object.keys(ch).length) return null;
  return Object.entries(ch).map(([k, v]) => (
    <span key={k} className="chip chg">{k === "source" ? "source edited" : `${k} → ${String(v)}`}</span>
  ));
}

function Stage({ k, v, cls }: { k: string; v: string; cls?: string }) {
  return <div className="stage"><span className="k">{k}</span><span className={`v ${cls ? "st " + cls : ""}`} title={v}>{v}</span></div>;
}

function stages(e: TraceEntry) {
  const c = e.compile;
  const compile = c.ok == null ? ["–", ""] : c.ok ? [`ok${c.ms ? ` · ${c.ms.toFixed(0)} ms` : ""}${c.cached ? " (cached)" : ""}`, "ok"]
    : [`${c.kind ?? "error"}`, "bad"];
  const vs = e.validation.status;
  const w = e.validation.worst;
  const valid = !vs ? ["–", ""] : vs === "verified" ? ["pass", "ok"]
    : [`${vs}${w ? ` · err ${num(w.max_abs_err)}` : ""}${w?.nan_count ? ` · ${w.nan_count} NaN` : ""}`, "inv"];
  const b = e.benchmark;
  const bench = b.p50_ms == null ? ["–", ""] : [`${ms(b.p50_ms)} (p95 ${ms(b.p95_ms)})`, ""];
  const kind = entryKind(e);
  const verdict = e.origin === "baseline" ? ["reference", "amb"]
    : e.verdict ? [e.verdict.kind === "accept" ? "kept" : "rejected", e.verdict.kind === "accept" ? "ok" : kind === "reject" ? "rej" : "bad"]
    : e.status === "measured" ? [e.retained ? "best" : "measured", e.retained ? "ok" : ""] : ["–", ""];
  return { compile, valid, bench, verdict };
}

export function TracePanel({ trace }: { trace: Trace }) {
  const problems = useMemo(() => [...new Set(trace.entries.map((e) => e.problem))], [trace]);
  const [problem, setProblem] = useState(problems[0]);
  const [sel, setSel] = useState<string | null>(null);
  const entries = trace.entries.filter((e) => e.problem === problem);
  const thoughtsByStep = useMemo(() => {
    const m = new Map<number, string[]>();
    for (const t of trace.thoughts.filter((t) => t.problem === problem)) {
      m.set(t.step, [...(m.get(t.step) ?? []), t.text]);
    }
    return m;
  }, [trace, problem]);
  const final = trace.final.find((f) => f.problem === problem);
  const isAgent = trace.experiment.strategy === "agent";

  // thoughts that came right before each hypothesis (the planner's reasoning)
  const thoughtFor = (e: TraceEntry) => {
    if (!isAgent || e.step == null) return null;
    for (let s = e.step - 1; s > e.step - 4; s--) {
      const t = thoughtsByStep.get(s);
      if (t) return t[t.length - 1];
    }
    return null;
  };

  return (
    <div>
      {problems.length > 1 && (
        <div className="row" style={{ marginBottom: 10 }}>
          <span className="sub">problem</span>
          <select value={problem} onChange={(ev) => { setProblem(ev.target.value); setSel(null); }}>
            {problems.map((p) => <option key={p}>{p}</option>)}
          </select>
        </div>
      )}
      <div className="panel pad">
        <TraceChart entries={entries} selected={sel} onSelect={setSel} />
        <TraceLegend />
      </div>
      {final && <div className="banner"><b>final:</b> {final.summary}</div>}
      <div className="split" style={{ marginTop: 14 }}>
        <div className="trace">
          {entries.map((e) => {
            const s = stages(e);
            const k = entryKind(e);
            const dotCls = e.retained ? "retained" : k === "accept" ? "accept" : k === "compile" ? "compile" : k === "invalid" ? "invalid" : "";
            const thought = thoughtFor(e);
            return (
              <Fragment key={e.id}>
                {thought && thought !== e.hypothesis && !thought.includes(e.hypothesis ?? "\u0000") && <div className="thought">{thought}</div>}
                <div className="tcard">
                  <div className="rail"><div className={`dot ${dotCls}`}>{e.seq}</div></div>
                  <div className={`body ${sel === e.id ? "sel" : ""}`} onClick={() => setSel(e.id)}>
                    <div className="row" style={{ justifyContent: "space-between" }}>
                      <span className="mono" style={{ color: "var(--muted)" }}>{e.label}</span>
                      {e.speedup != null && <span className="num" style={{ color: e.speedup > 1.01 ? "var(--teal)" : "var(--muted)" }}>{x(e.speedup)}</span>}
                    </div>
                    <div className="hyp">{e.origin === "baseline" ? "Untuned Triton baseline" : e.hypothesis || "—"}</div>
                    {e.rationale && <div className="why">because: {e.rationale}</div>}
                    <div>{changeChips(e)}</div>
                    <div className="stages">
                      <Stage k="compile" v={s.compile[0]} cls={s.compile[1]} />
                      <Stage k="validate" v={s.valid[0]} cls={s.valid[1]} />
                      <Stage k="benchmark" v={s.bench[0]} />
                      <Stage k="decision" v={s.verdict[0]} cls={s.verdict[1]} />
                    </div>
                    {e.verdict && e.origin !== "baseline" && <div className="why" style={{ marginTop: 6, marginBottom: 0 }}>{e.verdict.reason}</div>}
                  </div>
                </div>
              </Fragment>
            );
          })}
          {!entries.length && <div className="sub">no candidates for {shortProblem(problem ?? "")}</div>}
        </div>
        <div className="sticky">
          {sel ? <CandidatePanel id={sel} /> : <div className="panel pad sub">Click a candidate (in the chart or the timeline) to see its source diff, compiler output, correctness checks and latency distribution.</div>}
        </div>
      </div>
    </div>
  );
}
