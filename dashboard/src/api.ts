import type { CandidateDetail, Comparison, EnvInfo, Experiment, ExperimentRow, Trace } from "./types";

async function get<T>(path: string): Promise<T> {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  return res.json() as Promise<T>;
}

export const api = {
  env: () => get<EnvInfo>("/api/env"),
  experiments: () => get<ExperimentRow[]>("/api/experiments"),
  experiment: (id: string) => get<Experiment>(`/api/experiments/${id}`),
  trace: (id: string) => get<Trace>(`/api/experiments/${id}/trace`),
  candidate: (id: string) => get<CandidateDetail>(`/api/candidates/${id}`),
  compare: (a: string, b: string) => get<Comparison>(`/api/compare?a=${a}&b=${b}`),
  reportUrl: (id: string, fmt: "md" | "html" | "json") => `/api/experiments/${id}/report.${fmt}`,
  async startRun(body: Record<string, unknown>): Promise<{ id: string; status: string; error: string | null }> {
    const res = await fetch("/api/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail ?? res.statusText);
    return data;
  },
  job: (id: string) => get<{ id: string; status: string; experiment_id: string | null; error: string | null }>(`/api/runs/${id}`),
};
