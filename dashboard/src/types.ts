// Shapes returned by the FastAPI backend (src/kernelpilot/server.py).
// Non-finite floats arrive as the strings "inf" / "-inf" / "nan".
export type Num = number | "inf" | "-inf" | "nan" | null;

export interface ExperimentRow {
  id: string;
  name: string;
  strategy: string;
  status: string;
  simulated: boolean;
  backend: string;
  created_at: number;
  finished_at: number | null;
  target: string[];
  params: Record<string, unknown>;
  geomean_speedup: number | null;
  candidates: number | null;
  compile_failed: number | null;
  invalid: number | null;
}

export interface ProblemSummary {
  problem: string;
  baseline_id: string | null;
  baseline_p50: number | null;
  best_id: string | null;
  best_label: string | null;
  best_p50: number | null;
  speedup: number | null;
  evaluated: number;
  compile_failed: number;
  invalid: number;
  unverified: number;
  evals_to_best: number | null;
}

export interface Experiment {
  id: string;
  name: string;
  strategy: string;
  status: string;
  simulated: number;
  backend: string;
  target: string[];
  params: Record<string, unknown>;
  environment: Record<string, unknown>;
  fingerprint: Record<string, unknown>;
  created_at: number;
  finished_at: number | null;
  summary: {
    problems: ProblemSummary[];
    geomean_speedup: number | null;
    geomean_over: number;
    problems_total: number;
    candidates: number;
    compile_failed: number;
    invalid: number;
    unverified: number;
    wall_s: number;
    backend: string;
    cache?: { session_hits: number; session_misses: number };
  } | null;
}

export interface KernelConfig {
  variant: string;
  params: Record<string, number>;
  num_warps: number;
  num_stages: number;
}

export interface Check {
  problem: string;
  ok: boolean;
  max_abs_err: Num;
  max_rel_err: Num;
  mismatch_frac: number;
  nan_count: number;
  inf_count: number;
  atol: number;
  rtol: number;
  message: string;
}

export interface TraceEntry {
  id: string;
  seq: number;
  problem: string;
  label: string;
  origin: string;
  parent_id: string | null;
  config: KernelConfig | null;
  hypothesis: string | null;
  rationale: string | null;
  changes: Record<string, unknown> | null;
  evidence: Array<Record<string, unknown>> | Record<string, unknown> | null;
  step: number | null;
  compile: { ok: boolean | null; error: string | null; kind: string | null; ms: number | null; cached: boolean | null };
  validation: { status: string | null; error: string | null; worst: Check | null };
  benchmark: { p50_ms: number | null; p95_ms: number | null; min_ms: number | null; max_ms: number | null; samples: number[] | null };
  metadata: Record<string, unknown> | null;
  profile: Record<string, number> | null;
  status: string;
  speedup: number | null;
  retained: boolean;
  verdict: { kind: "accept" | "reject"; reason: string; evidence: Record<string, unknown> | null } | null;
}

export interface Trace {
  experiment: { id: string; name: string; strategy: string; status: string; simulated: number; backend: string };
  entries: TraceEntry[];
  thoughts: { step: number; problem: string; text: string }[];
  tool_calls: number;
  final: { problem: string; summary: string; candidate_id: string | null }[];
}

export interface CandidateDetail {
  id: string;
  label: string;
  problem: string;
  source: string;
  parent_source: string | null;
  parent_label: string | null;
  compile_log: string | null;
  compile_error: string | null;
  metadata: Record<string, unknown> | null;
  profile: { ok: boolean; tool: string; metrics: Record<string, number>; simulated: boolean; error: string | null } | null;
  validation: { ok: boolean; status: string; error: string | null; checks: Check[] } | null;
  samples: number[] | null;
  p50_ms: number | null;
  p95_ms: number | null;
  bench_method: string | null;
  ptx_summary: Record<string, unknown> | null;
  status: string;
}

export interface CurvePoint { n: number; best_p50_ms: number | null; speedup: number | null }

export interface CompareRow {
  problem: string;
  a_speedup: number | null;
  b_speedup: number | null;
  a_best: string | null;
  b_best: string | null;
  a_evaluated: number;
  b_evaluated: number;
  a_evals_to_best: number | null;
  b_evals_to_best: number | null;
  a_failures: number;
  b_failures: number;
  a_curve: CurvePoint[];
  b_curve: CurvePoint[];
  winner: string | null;
}

export interface Comparison {
  a: { id: string; name: string; strategy: string; simulated: boolean; geomean_speedup: number | null; candidates: number };
  b: { id: string; name: string; strategy: string; simulated: boolean; geomean_speedup: number | null; candidates: number };
  shared_problems: number;
  a_geomean_shared: number | null;
  b_geomean_shared: number | null;
  a_wins: number;
  b_wins: number;
  ties: number;
  rows: CompareRow[];
  mixed_backends: boolean;
}

export interface EnvInfo {
  backend: string;
  simulated: boolean;
  llm_provider: string;
  environment: { gpu_name: string | null; triton: string | null; torch: string | null; python: string };
}
