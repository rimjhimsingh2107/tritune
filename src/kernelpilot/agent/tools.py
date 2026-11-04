"""Typed tools the optimization agent works through.

Every tool takes a pydantic model and returns a pydantic model, and every
call goes through ToolRegistry.call, which validates input, enforces the
budget, catches exceptions and records the call in the decision log. The
heuristic planner and LLM planners use exactly the same tools.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from kernelpilot.agent.recommender import recommend
from kernelpilot.analysis.observations import observe, observe_failure
from kernelpilot.analysis.ptx import summarize_ptx
from kernelpilot.analysis.static_check import lint
from kernelpilot.backends.sim_engine import identify
from kernelpilot.candidate import Candidate
from kernelpilot.kernels import get_spec
from kernelpilot.log import get_logger, kv
from kernelpilot.problem import KernelConfig, Problem
from kernelpilot.session import Evaluation, Session
from kernelpilot.stats import compare_latency

log = get_logger("agent.tools")


class BudgetExhausted(Exception):
    pass


# --------------------------------------------------------------------------
# state shared by the tools during one agent run
# --------------------------------------------------------------------------


@dataclass
class AgentState:
    session: Session
    problem: Problem
    max_candidates: int
    baseline_id: str | None = None
    incumbent_id: str | None = None
    last_failed_id: str | None = None
    created: int = 0
    tried: set[str] = field(default_factory=set)
    hot_params: list[str] = field(default_factory=list)
    changes: dict[str, dict[str, Any]] = field(default_factory=dict)  # cand id -> what it changed
    bad_changes: list[dict[str, Any]] = field(default_factory=list)  # changes that measured slower
    done: bool = False
    final_summary: str = ""
    thought: str = ""  # planner's latest reasoning text, attached to decisions

    @property
    def spec(self):
        return get_spec(self.problem.op)

    def ev(self, cand_id: str | None) -> Evaluation:
        cid = cand_id or self.incumbent_id or self.baseline_id
        if cid is None:
            raise ValueError("no candidate id given and no incumbent yet")
        return self.session.get(cid)

    def config_of(self, ev: Evaluation) -> KernelConfig | None:
        return ev.candidate.config or identify(ev.candidate.source, ev.candidate.problem)

    def decide(self, kind: str, **kw: Any) -> None:
        kw.setdefault("problem", self.problem.key)
        self.session.decide(kind, **kw)


# --------------------------------------------------------------------------
# schemas
# --------------------------------------------------------------------------


class CandidateRef(BaseModel):
    candidate_id: str = Field(description="id returned by generate_candidate (or the baseline id)")


class OptionalCandidateRef(BaseModel):
    candidate_id: str | None = Field(None, description="defaults to the current best (incumbent)")


class GenerateCandidateIn(BaseModel):
    hypothesis: str = Field(description="what you expect this change to do and why, one sentence")
    rationale: str = Field("", description="the observation/measurement that motivated it")
    base_candidate_id: str | None = Field(None, description="candidate to start from; defaults to the incumbent")
    variant: str | None = Field(None, description="kernel variant (see inspect_kernel)")
    params: dict[str, int] = Field(default_factory=dict, description="block/tile params to override, e.g. {'BLOCK_M': 128}")
    num_warps: int | None = Field(None, ge=1, le=32)
    num_stages: int | None = Field(None, ge=1, le=8)
    source: str | None = Field(None, description="full replacement Triton module (must define launch()); "
                                                 "use only for changes params can't express")
    evidence: list[dict[str, Any]] = Field(default_factory=list, description="structured evidence, optional")


class BenchmarkIn(BaseModel):
    candidate_id: str
    reps: int | None = Field(None, ge=5, le=1000)


class CompareIn(BaseModel):
    candidate_ids: list[str] = Field(min_length=1, max_length=10)


class RecommendIn(BaseModel):
    candidate_id: str | None = Field(None, description="defaults to the incumbent")
    limit: int = Field(4, ge=1, le=10)


class SaveIn(BaseModel):
    summary: str = Field(description="short write-up of what worked, what didn't, and why")
    final_candidate_id: str | None = Field(None, description="defaults to the fastest verified candidate")


class InspectOut(BaseModel):
    problem: str
    candidate_id: str
    label: str
    origin: str
    config: dict[str, Any] | None
    source: str
    variants: dict[str, str]
    search_space: dict[str, dict[str, list[int]]]
    incumbent_id: str | None
    incumbent_p50_ms: float | None
    budget_left: int


class GenerateOut(BaseModel):
    candidate_id: str
    label: str
    duplicate_of: str | None = None
    parent_id: str | None = None
    changed: dict[str, Any] = Field(default_factory=dict)
    lint_warnings: list[str] = Field(default_factory=list)
    budget_left: int


class CompileOut(BaseModel):
    candidate_id: str
    ok: bool
    error_kind: str | None = None
    error: str | None = None
    compile_ms: float | None = None
    cached: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
    observations: list[dict[str, Any]] = Field(default_factory=list)


class ValidateOut(BaseModel):
    candidate_id: str
    ok: bool
    status: str
    summary: str
    checks: list[dict[str, Any]] = Field(default_factory=list)
    observations: list[dict[str, Any]] = Field(default_factory=list)


class BenchmarkOut(BaseModel):
    candidate_id: str
    ok: bool
    error: str | None = None
    p50_ms: float | None = None
    p95_ms: float | None = None
    std_ms: float | None = None
    reps: int = 0
    speedup_vs_baseline: float | None = None
    verdict: str | None = None
    accepted: bool = False
    incumbent_id: str | None = None
    simulated: bool = False


class MetadataOut(BaseModel):
    candidate_id: str
    metadata: dict[str, Any]
    observations: list[dict[str, Any]]


class PtxOut(BaseModel):
    candidate_id: str
    available: bool
    reason: str | None = None
    summary: dict[str, Any] = Field(default_factory=dict)


class ProfileOut(BaseModel):
    candidate_id: str
    ok: bool
    tool: str = ""
    error: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    simulated: bool = False
    observations: list[dict[str, Any]] = Field(default_factory=list)


class CompareOut(BaseModel):
    rows: list[dict[str, Any]]
    fastest_verified: str | None


class RecommendOut(BaseModel):
    based_on: str
    observations: list[dict[str, Any]]
    failed_candidate: str | None = None
    failure_observations: list[dict[str, Any]] = Field(default_factory=list)
    recommendations: list[dict[str, Any]]


class SaveOut(BaseModel):
    final_candidate_id: str | None
    label: str | None
    speedup_vs_baseline: float | None
    candidates_evaluated: int


# --------------------------------------------------------------------------
# implementations
# --------------------------------------------------------------------------


def _budget_left(st: AgentState) -> int:
    return max(0, st.max_candidates - st.created)


def inspect_kernel(st: AgentState, a: OptionalCandidateRef) -> InspectOut:
    ev = st.ev(a.candidate_id)
    cfg = st.config_of(ev)
    inc = st.session.evals.get(st.incumbent_id) if st.incumbent_id else None
    src = ev.candidate.source
    return InspectOut(
        problem=st.problem.key,
        candidate_id=ev.candidate.id,
        label=ev.candidate.label,
        origin=ev.candidate.origin,
        config=cfg.to_dict() if cfg else None,
        source=src if len(src) < 8000 else src[:8000] + "\n# ... truncated",
        variants={k: v.summary for k, v in st.spec.variants.items()},
        search_space=st.spec.space(st.problem),
        incumbent_id=st.incumbent_id,
        incumbent_p50_ms=inc.p50 if inc else None,
        budget_left=_budget_left(st),
    )


def generate_candidate(st: AgentState, a: GenerateCandidateIn) -> GenerateOut:
    if st.created >= st.max_candidates:
        raise BudgetExhausted(f"candidate budget of {st.max_candidates} used up; call save_experiment")
    base_id = a.base_candidate_id or st.incumbent_id or st.baseline_id
    base = st.session.get(base_id) if base_id else None
    changed: dict[str, Any] = {}

    if a.source:
        cfg = identify(a.source, st.problem)
        cand = Candidate(st.problem, a.source, cfg, origin="agent", parent_id=base_id, note=a.hypothesis)
        changed["source"] = "replaced"
    else:
        base_cfg = st.config_of(base) if base else st.spec.baseline(st.problem)
        if base_cfg is None:
            raise ValueError("base candidate has a custom kernel body; pass full `source` instead of params")
        changes: dict[str, Any] = dict(a.params)
        if a.variant:
            changes["variant"] = a.variant
        if a.num_warps:
            changes["num_warps"] = a.num_warps
        if a.num_stages:
            changes["num_stages"] = a.num_stages
        if not changes:
            raise ValueError("no change requested: pass variant/params/num_warps/num_stages or source")
        cfg = base_cfg.replace(**changes)
        st.spec.variant(cfg.variant)  # raises on unknown variant
        problems = st.spec.check(cfg, st.problem)
        if problems:
            raise ValueError("invalid config: " + "; ".join(problems))
        changed = {k: v for k, v in changes.items()
                   if (getattr(base_cfg, k, None) if k in ("variant", "num_warps", "num_stages")
                       else base_cfg.params.get(k)) != v}
        cand = Candidate.from_config(st.problem, cfg, origin="agent", parent_id=base_id, note=a.hypothesis)

    dup = st.session.find_duplicate(cand)
    if dup is not None:
        return GenerateOut(candidate_id=dup.candidate.id, label=dup.candidate.label, duplicate_of=dup.candidate.id,
                           parent_id=base_id, changed=changed, budget_left=_budget_left(st))

    ev, _ = st.session.register(cand)
    st.created += 1
    st.changes[cand.id] = changed
    if cand.config:
        st.tried.add(cand.config.key)
    report = lint(cand.source)
    st.decide("hypothesis", candidate_id=cand.id, rationale=a.hypothesis,
              args={"changes": changed, "base": base_id, "rationale": a.rationale},
              evidence=a.evidence or ([{"note": a.rationale}] if a.rationale else None))
    return GenerateOut(candidate_id=cand.id, label=cand.label, parent_id=base_id, changed=changed,
                       lint_warnings=[i.render() for i in report.warnings], budget_left=_budget_left(st))


def _reject(st: AgentState, ev: Evaluation, why: str, evidence: Any = None) -> None:
    st.last_failed_id = ev.candidate.id
    st.decide("reject", candidate_id=ev.candidate.id, rationale=why, evidence=evidence)


def compile_candidate(st: AgentState, a: CandidateRef) -> CompileOut:
    ev = st.ev(a.candidate_id)
    already = ev.compile is not None
    res = st.session.compile(ev)
    obs = [o.to_dict() for o in observe_failure(st.problem, ev)] if not res.ok else []
    if not res.ok and not already:
        first = (res.error or "").splitlines()[0][:200]
        _reject(st, ev, f"compile failed ({res.error_kind}): {first}", obs)
    md = {k: v for k, v in res.metadata.items() if k not in ("kernels", "config")}
    return CompileOut(candidate_id=ev.candidate.id, ok=res.ok, error_kind=res.error_kind,
                      error=(res.error or None) and res.error[:1500], compile_ms=res.compile_ms,
                      cached=res.cached, metadata=md, observations=obs)


def validate_correctness(st: AgentState, a: CandidateRef) -> ValidateOut:
    ev = st.ev(a.candidate_id)
    already = ev.validation is not None
    res = st.session.validate(ev)
    obs = [o.to_dict() for o in observe_failure(st.problem, ev)] if not res.ok else []
    if not res.ok and not already and ev.compile is not None and ev.compile.ok:
        _reject(st, ev, f"failed validation: {res.summary()}", obs)
    checks = [{"problem": c.problem, "ok": c.ok, "max_abs_err": c.max_abs_err, "max_rel_err": c.max_rel_err,
               "nan": c.nan_count, "inf": c.inf_count, "message": c.message} for c in res.checks]
    return ValidateOut(candidate_id=ev.candidate.id, ok=res.ok, status=res.status, summary=res.summary(),
                       checks=checks, observations=obs)


def _hot_param(st: AgentState, ev: Evaluation) -> None:
    if not ev.candidate.parent_id or ev.candidate.config is None:
        return
    parent = st.session.evals.get(ev.candidate.parent_id)
    pcfg = st.config_of(parent) if parent else None
    if pcfg is None:
        return
    cfg = ev.candidate.config
    for k, v in cfg.params.items():
        if pcfg.params.get(k) != v and k not in st.hot_params:
            st.hot_params.append(k)
    for k in ("num_warps", "num_stages"):
        if getattr(pcfg, k) != getattr(cfg, k) and k not in st.hot_params:
            st.hot_params.append(k)


def benchmark_candidate(st: AgentState, a: BenchmarkIn) -> BenchmarkOut:
    ev = st.ev(a.candidate_id)
    first_time = ev.bench is None
    res = st.session.benchmark(ev, a.reps)
    out = BenchmarkOut(candidate_id=ev.candidate.id, ok=res.ok, error=res.error, simulated=res.simulated,
                       incumbent_id=st.incumbent_id)
    if not res.ok:
        return out
    out.p50_ms, out.p95_ms, out.std_ms, out.reps = res.p50_ms, res.p95_ms, res.std_ms, res.n
    out.speedup_vs_baseline = st.session.speedup(ev)

    if st.incumbent_id is None:
        st.incumbent_id = ev.candidate.id
        out.accepted, out.verdict = True, "first measured candidate, this is the incumbent to beat"
        st.decide("accept", candidate_id=ev.candidate.id, rationale=out.verdict,
                  evidence={"p50_ms": res.p50_ms})
    elif ev.candidate.id != st.incumbent_id and first_time:
        inc = st.session.get(st.incumbent_id)
        v = compare_latency(res.p50_ms, res.p95_ms, inc.p50, inc.p95)
        out.verdict = v.reason
        evidence = {"new_p50_ms": res.p50_ms, "incumbent_p50_ms": inc.p50, "incumbent": inc.candidate.id,
                    "gain": round(v.gain, 4), "threshold": round(v.threshold, 4)}
        if v.better:
            out.accepted = True
            st.incumbent_id = ev.candidate.id
            _hot_param(st, ev)
            st.last_failed_id = None
            st.decide("accept", candidate_id=ev.candidate.id, rationale=f"kept: {v.reason}", evidence=evidence)
        else:
            if v.gain < -v.threshold and st.changes.get(ev.candidate.id):
                st.bad_changes.append(st.changes[ev.candidate.id])
            st.decide("reject", candidate_id=ev.candidate.id, rationale=f"not kept: {v.reason}", evidence=evidence)
    out.incumbent_id = st.incumbent_id
    return out


def inspect_compiler_metadata(st: AgentState, a: CandidateRef) -> MetadataOut:
    ev = st.ev(a.candidate_id)
    st.session.compile(ev)
    md = dict(ev.compile.metadata) if ev.compile else {}
    md.pop("config", None)
    obs = observe(st.problem, ev, st.config_of(ev))
    return MetadataOut(candidate_id=ev.candidate.id, metadata=md, observations=[o.to_dict() for o in obs])


def inspect_ptx(st: AgentState, a: CandidateRef) -> PtxOut:
    ev = st.ev(a.candidate_id)
    comp = st.session.compile(ev)
    if not comp.ok:
        return PtxOut(candidate_id=ev.candidate.id, available=False, reason="candidate did not compile")
    if not comp.ptx:
        reason = ("the simulated backend does not generate PTX" if comp.simulated
                  else "this Triton build did not expose PTX for the kernel")
        return PtxOut(candidate_id=ev.candidate.id, available=False, reason=reason)
    return PtxOut(candidate_id=ev.candidate.id, available=True, summary=summarize_ptx(comp.ptx).to_dict())


def profile_candidate(st: AgentState, a: CandidateRef) -> ProfileOut:
    ev = st.ev(a.candidate_id)
    res = st.session.profile(ev)
    obs = observe(st.problem, ev, st.config_of(ev)) if res.ok else []
    return ProfileOut(candidate_id=ev.candidate.id, ok=res.ok, tool=res.tool, error=res.error,
                      metrics={k: round(v, 3) for k, v in res.metrics.items()}, simulated=res.simulated,
                      observations=[o.to_dict() for o in obs])


def compare_candidates(st: AgentState, a: CompareIn) -> CompareOut:
    rows = []
    for cid in a.candidate_ids:
        ev = st.session.get(cid)
        rows.append({"candidate_id": cid, "label": ev.candidate.label, "status": ev.status,
                     "p50_ms": ev.p50, "p95_ms": ev.p95, "speedup_vs_baseline": st.session.speedup(ev),
                     "incumbent": cid == st.incumbent_id})
    ok = [r for r in rows if r["status"] == "measured"]
    fastest = min(ok, key=lambda r: r["p50_ms"])["candidate_id"] if ok else None
    return CompareOut(rows=rows, fastest_verified=fastest)


def recommend_optimization(st: AgentState, a: RecommendIn) -> RecommendOut:
    ev = st.ev(a.candidate_id)
    cfg = st.config_of(ev)
    obs = observe(st.problem, ev, cfg)
    failed = st.session.evals.get(st.last_failed_id) if st.last_failed_id else None
    failed_obs = observe_failure(st.problem, failed) if failed else []
    recs = recommend(
        st.problem, cfg if ev.measured else None, obs,
        failed=st.config_of(failed) if failed else None, failed_obs=failed_obs,
        tried=st.tried | {e.candidate.config.key for e in st.session.evaluations(st.problem) if e.candidate.config},
        hot_params=st.hot_params, avoid=st.bad_changes, limit=a.limit,
    )
    return RecommendOut(
        based_on=ev.candidate.id,
        observations=[o.to_dict() for o in obs],
        failed_candidate=failed.candidate.id if failed else None,
        failure_observations=[o.to_dict() for o in failed_obs],
        recommendations=[{
            **r.to_dict(),
            "generate_args": {
                "base_candidate_id": failed.candidate.id if (r.based_on == "failed" and failed) else ev.candidate.id,
                **{k: v for k, v in r.changes.items() if k in ("variant", "num_warps", "num_stages")},
                "params": {k: v for k, v in r.changes.items() if k not in ("variant", "num_warps", "num_stages")},
                "hypothesis": r.hypothesis,
                "rationale": r.rationale,
            },
        } for r in recs],
    )


def save_experiment(st: AgentState, a: SaveIn) -> SaveOut:
    pick = a.final_candidate_id or st.incumbent_id
    final = st.session.evals.get(pick) if pick else None
    if final is None or not final.measured or final.candidate.problem.key != st.problem.key:
        final = st.session.best(st.problem)
    if final is not None:
        st.session.pin(st.problem, final.candidate.id)
    st.done = True
    st.final_summary = a.summary
    st.decide("final", candidate_id=final.candidate.id if final else None, rationale=a.summary,
              evidence={"speedup": st.session.speedup(final) if final else None})
    return SaveOut(
        final_candidate_id=final.candidate.id if final else None,
        label=final.candidate.label if final else None,
        speedup_vs_baseline=st.session.speedup(final) if final else None,
        candidates_evaluated=len(st.session.evaluations(st.problem)),
    )


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------


@dataclass
class Tool:
    name: str
    description: str
    input_model: type[BaseModel]
    fn: Callable[[AgentState, Any], BaseModel]

    def schema(self) -> dict[str, Any]:
        schema = self.input_model.model_json_schema()
        schema.pop("title", None)
        return {"name": self.name, "description": self.description, "input_schema": schema}


TOOLS: list[Tool] = [
    Tool("inspect_kernel", "Show a candidate's Triton source, launch config, available variants and the "
         "search space.", OptionalCandidateRef, inspect_kernel),
    Tool("generate_candidate", "Create a new candidate from a base candidate by changing variant / params / "
         "num_warps / num_stages, or by supplying full source. Always state a hypothesis. Consumes budget.",
         GenerateCandidateIn, generate_candidate),
    Tool("compile_candidate", "Compile a candidate. Returns compiler errors or resource metadata.",
         CandidateRef, compile_candidate),
    Tool("validate_correctness", "Check numerical output against a float64 PyTorch/numpy reference on the "
         "target shape plus ragged edge shapes.", CandidateRef, validate_correctness),
    Tool("benchmark_candidate", "Time a validated candidate (p50/p95). Compares against the incumbent and "
         "keeps it only if faster beyond measurement noise.", BenchmarkIn, benchmark_candidate),
    Tool("inspect_compiler_metadata", "Registers, spills, shared memory, occupancy estimate, plus derived "
         "observations.", CandidateRef, inspect_compiler_metadata),
    Tool("inspect_ptx", "Summarize generated PTX: vectorized loads, local memory, mma, cp.async.",
         CandidateRef, inspect_ptx),
    Tool("profile_candidate", "Run Nsight Compute (or the simulated profiler) for DRAM/SM throughput and "
         "achieved occupancy.", CandidateRef, profile_candidate),
    Tool("compare_candidates", "Tabulate several candidates side by side.", CompareIn, compare_candidates),
    Tool("recommend_optimization", "Rule-based advisor: observations for a candidate plus ranked, untried "
         "config changes (with repair suggestions if the last attempt failed).", RecommendIn,
         recommend_optimization),
    Tool("save_experiment", "Finish: record the final candidate and a summary. Ends the run.",
         SaveIn, save_experiment),
]


@dataclass
class ToolResult:
    call_id: str
    name: str
    ok: bool
    content: dict[str, Any]
    error: str | None = None
    ms: float = 0.0


class ToolRegistry:
    def __init__(self, state: AgentState, tools: list[Tool] | None = None):
        self.state = state
        self.tools = {t.name: t for t in (tools or TOOLS)}
        self.calls = 0

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self.tools.values()]

    def call(self, name: str, args: dict[str, Any] | None, call_id: str = "") -> ToolResult:
        self.calls += 1
        t0 = time.perf_counter()
        tool = self.tools.get(name)
        if tool is None:
            res = ToolResult(call_id, name, False, {}, f"unknown tool {name!r}; available: {sorted(self.tools)}")
        else:
            try:
                parsed = tool.input_model.model_validate(args or {})
                out = tool.fn(self.state, parsed)
                res = ToolResult(call_id, name, True, out.model_dump(mode="json"))
            except ValidationError as exc:
                res = ToolResult(call_id, name, False, {}, f"invalid arguments: {exc.errors(include_url=False)}")
            except (BudgetExhausted, KeyError, ValueError) as exc:
                res = ToolResult(call_id, name, False, {}, str(exc).strip("'\""))
            except Exception as exc:  # tool bugs shouldn't kill the run
                log.exception("tool crashed", extra=kv(tool=name))
                res = ToolResult(call_id, name, False, {}, f"{type(exc).__name__}: {exc}")
        res.ms = (time.perf_counter() - t0) * 1e3
        self.state.decide(
            "tool_call", tool=name, args=args,
            result=_truncate(res.content) if res.ok else {"error": res.error},
            rationale=self.state.thought or None,
            candidate_id=(args or {}).get("candidate_id") or res.content.get("candidate_id"),
        )
        return res


def _truncate(content: dict[str, Any], limit: int = 600) -> dict[str, Any]:
    out = {}
    for k, v in content.items():
        if isinstance(v, str) and len(v) > limit:
            out[k] = v[:limit] + "..."
        else:
            out[k] = v
    return out
