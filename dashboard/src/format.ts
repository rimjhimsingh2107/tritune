import type { Num } from "./types";

export function ms(v: number | null | undefined): string {
  if (v == null) return "–";
  if (v < 0.1) return `${(v * 1000).toFixed(1)} µs`;
  return `${v.toFixed(3)} ms`;
}

export function x(v: number | null | undefined, digits = 2): string {
  return v == null ? "–" : `${v.toFixed(digits)}×`;
}

export function num(v: Num | undefined, digits = 3): string {
  if (v == null) return "–";
  if (typeof v === "string") return v === "nan" ? "NaN" : v === "inf" ? "∞" : "−∞";
  if (v === 0) return "0";
  if (Math.abs(v) < 1e-3 || Math.abs(v) >= 1e5) return v.toExponential(2);
  return v.toPrecision(digits);
}

export function bytes(v: unknown): string {
  if (typeof v !== "number") return "–";
  if (v >= 1024 * 1024) return `${(v / 1024 / 1024).toFixed(1)} MB`;
  if (v >= 1024) return `${(v / 1024).toFixed(1)} KB`;
  return `${v} B`;
}

export function ago(ts: number): string {
  const s = Date.now() / 1000 - ts;
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return new Date(ts * 1000).toLocaleDateString();
}

export function shortProblem(key: string): string {
  return key.replace(/:fp(16|32)$/, (m) => ` ${m.slice(1)}`).replace(":", " ");
}
