import type { CurvePoint, TraceEntry } from "../types";
import { ms } from "../format";

const W = 760;

function niceLog(lo: number, hi: number): number[] {
  const ticks: number[] = [];
  const start = Math.floor(Math.log10(lo));
  const end = Math.ceil(Math.log10(hi));
  for (let e = start; e <= end; e++) for (const m of [1, 2, 5]) {
    const v = m * 10 ** e;
    if (v >= lo * 0.999 && v <= hi * 1.001) ticks.push(v);
  }
  return ticks.length >= 2 ? ticks : [lo, hi];
}

export function entryKind(e: TraceEntry): "compile" | "invalid" | "accept" | "reject" | "baseline" | "pending" {
  if (e.status === "compile_failed") return "compile";
  if (e.status === "invalid" || e.status === "unverified") return "invalid";
  if (e.origin === "baseline") return "baseline";
  if (e.verdict?.kind === "accept") return "accept";
  if (e.status === "measured") return "reject";
  return "pending";
}

const COLORS: Record<string, string> = {
  compile: "var(--red)", invalid: "var(--magenta)", accept: "var(--teal)",
  reject: "var(--gray)", baseline: "var(--amber)", pending: "var(--dim)",
};

/** The signature view: every candidate in order, latency on a log axis,
 * failures parked in bands above the plot, and the incumbent as a step line. */
export function TraceChart({ entries, selected, onSelect }: {
  entries: TraceEntry[]; selected?: string | null; onSelect: (id: string) => void;
}) {
  const H = 260, L = 64, R = 16, T = 46, B = 28;
  const measured = entries.filter((e) => e.benchmark.p50_ms != null);
  if (!entries.length) return null;
  const vals = measured.flatMap((e) => [e.benchmark.p50_ms!, e.benchmark.p95_ms ?? e.benchmark.p50_ms!]);
  const lo = vals.length ? Math.min(...vals) * 0.85 : 0.01;
  const hi = vals.length ? Math.max(...vals) * 1.15 : 1;
  const xs = (i: number) => L + (entries.length === 1 ? 0.5 : i / (entries.length - 1)) * (W - L - R);
  const ys = (v: number) => T + (1 - (Math.log(v) - Math.log(lo)) / (Math.log(hi) - Math.log(lo))) * (H - T - B);

  // incumbent (best-so-far among verified + accepted/baseline)
  let best: number | null = null;
  const steps: [number, number][] = [];
  entries.forEach((e, i) => {
    const k = entryKind(e);
    const p = e.benchmark.p50_ms;
    if (p != null && (k === "accept" || k === "baseline" || (e.origin !== "agent" && e.status === "measured"))) {
      if (best == null || p < best) best = p;
    }
    if (best != null) steps.push([xs(i), ys(best)]);
  });
  const stepPath = steps.map(([x, y], i) => (i === 0 ? `M${x},${y}` : `H${x}V${y}`)).join("");

  return (
    <svg className="chart" viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label="optimization trace">
      <text x={4} y={14}>compile ✗</text>
      <text x={4} y={32}>invalid ✗</text>
      <line className="grid" x1={L} x2={W - R} y1={T - 6} y2={T - 6} strokeDasharray="2 3" />
      {niceLog(lo, hi).map((t) => (
        <g key={t}>
          <line className="grid" x1={L} x2={W - R} y1={ys(t)} y2={ys(t)} />
          <text x={L - 6} y={ys(t) + 3} textAnchor="end">{ms(t)}</text>
        </g>
      ))}
      {stepPath && <path d={stepPath} fill="none" stroke="var(--amber)" strokeWidth={1.5} strokeDasharray="5 3" />}
      {entries.map((e, i) => {
        const k = entryKind(e);
        const cx = xs(i);
        const sel = e.id === selected;
        if (k === "compile" || k === "invalid") {
          const cy = k === "compile" ? 10 : 28;
          return (
            <g key={e.id} onClick={() => onSelect(e.id)} style={{ cursor: "pointer" }}>
              <title>{`#${e.seq} ${e.label}\n${k === "compile" ? e.compile.kind + " error" : "validation " + e.validation.status}`}</title>
              <path d={`M${cx - 5},${cy - 5}L${cx + 5},${cy + 5}M${cx + 5},${cy - 5}L${cx - 5},${cy + 5}`} stroke={COLORS[k]} strokeWidth={sel ? 3 : 2} />
              <line x1={cx} x2={cx} y1={cy + 7} y2={H - B} stroke={COLORS[k]} strokeOpacity={0.15} />
            </g>
          );
        }
        const p50 = e.benchmark.p50_ms;
        if (p50 == null) return null;
        const p95 = e.benchmark.p95_ms ?? p50;
        return (
          <g key={e.id} onClick={() => onSelect(e.id)} style={{ cursor: "pointer" }}>
            <title>{`#${e.seq} ${e.label}\np50 ${ms(p50)}  p95 ${ms(p95)}${e.verdict ? "\n" + e.verdict.reason : ""}`}</title>
            <line x1={cx} x2={cx} y1={ys(p50)} y2={ys(p95)} stroke={COLORS[k]} strokeWidth={2} strokeOpacity={0.5} />
            <circle cx={cx} cy={ys(p50)} r={sel ? 7 : 5} fill={COLORS[k]} stroke={e.retained ? "var(--ink)" : "none"} strokeWidth={2} />
          </g>
        );
      })}
      <text x={L} y={H - 8}>#1</text>
      <text x={W - R} y={H - 8} textAnchor="end">#{entries.length} candidates →</text>
    </svg>
  );
}

export function TraceLegend() {
  return (
    <div className="legend">
      <span><i style={{ background: "var(--amber)" }} />baseline</span>
      <span><i style={{ background: "var(--teal)" }} />kept (new incumbent)</span>
      <span><i style={{ background: "var(--gray)" }} />measured, not kept</span>
      <span><i style={{ background: "var(--magenta)" }} />failed validation</span>
      <span><i style={{ background: "var(--red)" }} />failed to compile</span>
      <span>– – incumbent latency · ring = retained · bar = p50→p95</span>
    </div>
  );
}

export function Bars({ rows }: { rows: { label: string; value: number | null; hint?: string }[] }) {
  const vals = rows.map((r) => r.value ?? 0);
  const top = Math.max(1.2, ...vals) * 1.08;
  const LW = 240, H = rows.length * 24 + 24;
  const sx = (v: number) => LW + (v / top) * (W - LW - 60);
  return (
    <svg className="chart" viewBox={`0 0 ${W} ${H}`} width="100%">
      {rows.map((r, i) => (
        <g key={r.label}>
          <text x={0} y={i * 24 + 16}>{r.label}</text>
          {r.value != null && <rect x={LW} y={i * 24 + 5} width={Math.max(1, sx(r.value) - LW)} height={15} rx={2}
            fill={r.value > 1.01 ? "var(--teal)" : "var(--gray)"} />}
          <text x={(r.value != null ? sx(r.value) : LW) + 6} y={i * 24 + 16}>{r.value != null ? `${r.value.toFixed(2)}×` : "no verified result"}</text>
        </g>
      ))}
      <line x1={sx(1)} x2={sx(1)} y1={0} y2={H - 16} stroke="var(--ink)" strokeDasharray="3 3" strokeOpacity={0.6} />
      <text x={sx(1)} y={H - 4} textAnchor="middle">1×</text>
    </svg>
  );
}

export function Curves({ series, height = 180 }: {
  series: { name: string; color: string; points: CurvePoint[] }[]; height?: number;
}) {
  const w = 420, L = 40, R = 10, T = 10, B = 22;
  const all = series.flatMap((s) => s.points.filter((p) => p.speedup != null));
  if (!all.length) return <div className="sub">no data</div>;
  const nMax = Math.max(...series.map((s) => s.points.length));
  const hi = Math.max(1.05, ...all.map((p) => p.speedup!)) * 1.05;
  const lo = Math.min(1, ...all.map((p) => p.speedup!)) * 0.95;
  const sx = (n: number) => L + ((n - 1) / Math.max(1, nMax - 1)) * (w - L - R);
  const sy = (v: number) => T + (1 - (v - lo) / (hi - lo)) * (height - T - B);
  return (
    <svg className="chart" viewBox={`0 0 ${w} ${height}`} width="100%">
      {[lo, 1, (lo + hi) / 2, hi].filter((v, i, a) => a.indexOf(v) === i && v >= lo).map((v) => (
        <g key={v}><line className="grid" x1={L} x2={w - R} y1={sy(v)} y2={sy(v)} />
          <text x={L - 4} y={sy(v) + 3} textAnchor="end">{v.toFixed(2)}×</text></g>
      ))}
      {series.map((s) => {
        const pts = s.points.filter((p) => p.speedup != null);
        const d = pts.map((p, i) => `${i ? "L" : "M"}${sx(p.n)},${sy(p.speedup!)}`).join("");
        return <path key={s.name} d={d} fill="none" stroke={s.color} strokeWidth={2} />;
      })}
      <text x={L} y={height - 6}>eval 1</text>
      <text x={w - R} y={height - 6} textAnchor="end">eval {nMax}</text>
    </svg>
  );
}

export function Histogram({ samples, height = 90 }: { samples: number[]; height?: number }) {
  if (!samples?.length) return null;
  const w = 360, bins = 24;
  const lo = Math.min(...samples), hi = Math.max(...samples);
  const span = hi - lo || hi * 0.01 || 1;
  const counts = new Array(bins).fill(0);
  samples.forEach((s) => counts[Math.min(bins - 1, Math.floor(((s - lo) / span) * bins))]++);
  const top = Math.max(...counts);
  const bw = (w - 20) / bins;
  return (
    <svg className="chart" viewBox={`0 0 ${w} ${height}`} width="100%">
      {counts.map((c, i) => (
        <rect key={i} x={10 + i * bw} y={height - 16 - (c / top) * (height - 26)} width={bw - 1}
          height={(c / top) * (height - 26)} fill="var(--blue)" fillOpacity={0.75} />
      ))}
      <text x={10} y={height - 3}>{ms(lo)}</text>
      <text x={w - 10} y={height - 3} textAnchor="end">{ms(hi)}</text>
    </svg>
  );
}
