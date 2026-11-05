"""An experiment in progress: evaluates candidates and records everything.

Both the autotuner and the agent go through Session, so every number in the
database was produced by the same compile -> validate -> benchmark pipeline.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from tritune.backends.base import Backend, BenchmarkResult, CompileResult, ProfileResult
from tritune.candidate import Candidate
from tritune.config import Settings
from tritune.correctness import ValidationResult
from tritune.env import Environment
from tritune.kernels import get_spec
from tritune.log import get_logger, kv
from tritune.problem import Problem
from tritune.stats import geomean
from tritune.store import Store

log = get_logger("session")


@dataclass
class Evaluation:
    candidate: Candidate
    compile: CompileResult | None = None
    validation: ValidationResult | None = None
    bench: BenchmarkResult | None = None
    profile: ProfileResult | None = None
    status: str = "pending"

    @property
    def measured(self) -> bool:
        return self.status == "measured"

    @property
    def p50(self) -> float | None:
        return self.bench.p50_ms if self.bench and self.bench.ok else None

    @property
    def p95(self) -> float | None:
        return self.bench.p95_ms if self.bench and self.bench.ok else None

    def brief(self) -> dict[str, Any]:
        c = self.candidate
        out: dict[str, Any] = {"candidate_id": c.id, "label": c.label, "status": self.status}
        if self.compile is not None:
            out["compile"] = {
                "ok": self.compile.ok,
                "error": self.compile.error,
                "kind": self.compile.error_kind,
                "ms": self.compile.compile_ms,
            }
        if self.validation is not None:
            out["validation"] = self.validation.summary()
        if self.bench is not None and self.bench.ok:
            out["p50_ms"] = round(self.bench.p50_ms or 0, 5)
            out["p95_ms"] = round(self.bench.p95_ms or 0, 5)
        return out


@dataclass
class ProblemSummary:
    problem: str
    baseline_id: str | None = None
    baseline_p50: float | None = None
    best_id: str | None = None
    best_label: str | None = None
    best_p50: float | None = None
    speedup: float | None = None
    evaluated: int = 0
    compile_failed: int = 0
    invalid: int = 0
    unverified: int = 0
    evals_to_best: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class Session:
    def __init__(
        self,
        store: Store,
        backend: Backend,
        settings: Settings,
        env: Environment,
        *,
        name: str,
        strategy: str,
        target: list[Problem],
        params: dict[str, Any] | None = None,
    ):
        self.store = store
        self.backend = backend
        self.settings = settings
        self.target = target
        self.strategy = strategy
        self.params = dict(params or {})
        self.params.setdefault("seed", settings.seed)
        self.params.setdefault("reps", settings.reps)
        self.params.setdefault("warmup", settings.warmup)
        self.exp_id = store.create_experiment(
            name=name,
            strategy=strategy,
            target=[p.key for p in target],
            params=self.params,
            environment=env.to_dict(),
            fingerprint=backend.fingerprint(),
            backend=backend.name,
            simulated=backend.simulated,
        )
        self.evals: dict[str, Evaluation] = {}
        self.order: list[str] = []
        self._by_hash: dict[tuple[str, str], str] = {}
        self._baselines: dict[str, Evaluation] = {}
        self._pinned: dict[str, str] = {}
        self._step = 0
        self.started = time.time()
        log.info(
            "experiment started", extra=kv(id=self.exp_id, strategy=strategy, backend=backend.name)
        )

    @property
    def seed(self) -> int:
        return int(self.params["seed"])

    # -- bookkeeping -----------------------------------------------------------

    def decide(self, kind: str, **kw: Any) -> int:
        self._step += 1
        return self.store.add_decision(self.exp_id, self._step, kind, **kw)

    def find_duplicate(self, cand: Candidate) -> Evaluation | None:
        cid = self._by_hash.get((cand.problem.key, cand.source_hash))
        return self.evals.get(cid) if cid else None

    def register(self, cand: Candidate) -> tuple[Evaluation, bool]:
        """Add a candidate. Returns (evaluation, is_duplicate)."""
        dup = self.find_duplicate(cand)
        if dup is not None:
            return dup, True
        self.store.add_candidate(self.exp_id, cand)
        ev = Evaluation(cand)
        self.evals[cand.id] = ev
        self.order.append(cand.id)
        self._by_hash[(cand.problem.key, cand.source_hash)] = cand.id
        return ev, False

    def get(self, cand_id: str) -> Evaluation:
        try:
            return self.evals[cand_id]
        except KeyError:
            raise KeyError(f"no candidate {cand_id!r} in this experiment") from None

    def _set_status(self, ev: Evaluation, status: str) -> None:
        ev.status = status
        speedup = self.speedup(ev) if status == "measured" else None
        self.store.set_status(ev.candidate.id, status, speedup)

    # -- pipeline stages -----------------------------------------------------

    def compile(self, ev: Evaluation) -> CompileResult:
        if ev.compile is None:
            ev.compile = self.backend.compile(ev.candidate)
            self.store.record_compile(ev.candidate.id, ev.compile)
            if not ev.compile.ok:
                self._set_status(ev, "compile_failed")
            elif ev.status == "pending":
                self._set_status(ev, "compiled")
        return ev.compile

    def validate(self, ev: Evaluation) -> ValidationResult:
        if ev.validation is None:
            comp = self.compile(ev)
            if not comp.ok:
                ev.validation = ValidationResult(False, status="error", error="did not compile")
            else:
                ev.validation = self.backend.validate(ev.candidate, self.seed)
                self.store.record_validation(ev.candidate.id, ev.validation)
                if ev.validation.status == "unverified":
                    self._set_status(ev, "unverified")
                elif not ev.validation.ok:
                    self._set_status(ev, "invalid")
                else:
                    self._set_status(ev, "validated")
        return ev.validation

    def benchmark(self, ev: Evaluation, reps: int | None = None) -> BenchmarkResult:
        reps = reps or int(self.params["reps"])
        if ev.bench is not None and ev.bench.ok and ev.bench.n >= reps:
            return ev.bench
        val = self.validate(ev)
        if not val.ok:
            # never time something we couldn't show is correct
            return BenchmarkResult(ok=False, error=f"not benchmarked: validation {val.status}")
        ev.bench = self.backend.benchmark(ev.candidate, int(self.params["warmup"]), reps)
        self.store.record_benchmark(ev.candidate.id, ev.bench)
        self._set_status(ev, "measured" if ev.bench.ok else "bench_failed")
        return ev.bench

    def profile(self, ev: Evaluation) -> ProfileResult:
        if ev.profile is None:
            if not self.compile(ev).ok:
                return ProfileResult(ok=False, error="did not compile")
            ev.profile = self.backend.profile(ev.candidate)
            self.store.record_profile(ev.candidate.id, ev.profile)
        return ev.profile

    def evaluate(
        self, cand: Candidate, reps: int | None = None, profile: bool = False
    ) -> Evaluation:
        ev, dup = self.register(cand)
        if dup:
            return ev
        self.benchmark(ev, reps)
        if profile and ev.measured:
            self.profile(ev)
        log.info(
            "evaluated", extra=kv(cand=cand.id, label=cand.label, status=ev.status, p50=ev.p50)
        )
        return ev

    # -- baselines and bests -------------------------------------------------

    def baseline(self, problem: Problem) -> Evaluation:
        if problem.key not in self._baselines:
            spec = get_spec(problem.op)
            cand = Candidate.from_config(
                problem, spec.baseline(problem), origin="baseline", note="untuned Triton baseline"
            )
            self._baselines[problem.key] = self.evaluate(cand)
        return self._baselines[problem.key]

    def register_baseline(self, problem: Problem) -> Evaluation:
        """Register the baseline without evaluating it (the agent drives that itself)."""
        if problem.key not in self._baselines:
            spec = get_spec(problem.op)
            cand = Candidate.from_config(
                problem, spec.baseline(problem), origin="baseline", note="untuned Triton baseline"
            )
            self._baselines[problem.key], _ = self.register(cand)
        return self._baselines[problem.key]

    def speedup(self, ev: Evaluation) -> float | None:
        base = self._baselines.get(ev.candidate.problem.key)
        if base is None or base.p50 is None or ev.p50 is None:
            return None
        return base.p50 / ev.p50

    def evaluations(self, problem: Problem | None = None) -> list[Evaluation]:
        evs = [self.evals[cid] for cid in self.order]
        if problem is not None:
            evs = [e for e in evs if e.candidate.problem.key == problem.key]
        return evs

    def pin(self, problem: Problem, cand_id: str) -> None:
        """Make an explicit choice the 'best' (the agent keeps its incumbent
        even if a rejected candidate was faster by less than the noise)."""
        self._pinned[problem.key] = cand_id

    def best(self, problem: Problem) -> Evaluation | None:
        pinned = self.evals.get(self._pinned.get(problem.key, ""))
        if pinned is not None and pinned.measured:
            return pinned
        measured = [
            e for e in self.evaluations(problem) if e.measured and e.validation and e.validation.ok
        ]
        return min(measured, key=lambda e: (e.p50, e.p95 or e.p50), default=None)

    # -- wrap up ---------------------------------------------------------------

    def summarize(self, problem: Problem) -> ProblemSummary:
        evs = self.evaluations(problem)
        s = ProblemSummary(problem.key, evaluated=len(evs))
        s.compile_failed = sum(e.status == "compile_failed" for e in evs)
        s.invalid = sum(e.status == "invalid" for e in evs)
        s.unverified = sum(e.status == "unverified" for e in evs)
        base = self._baselines.get(problem.key)
        if base is not None:
            s.baseline_id = base.candidate.id
            s.baseline_p50 = base.p50
        best = self.best(problem)
        if best is not None:
            s.best_id, s.best_label, s.best_p50 = best.candidate.id, best.candidate.label, best.p50
            if base is not None and base.measured:
                s.speedup = base.p50 / best.p50
            # how many evaluations until we were within 2% of the final best
            for i, e in enumerate(evs, 1):
                if e.p50 is not None and e.measured and e.p50 <= best.p50 * 1.02:
                    s.evals_to_best = i
                    break
        return s

    def finish(self, status: str = "done", extra: dict[str, Any] | None = None) -> dict[str, Any]:
        per_problem = []
        for p in self.target:
            s = self.summarize(p)
            if s.best_id:
                self.store.set_retained(self.exp_id, p.key, s.best_id)
            per_problem.append(s.__dict__)
        speedups = [s["speedup"] for s in per_problem if s["speedup"]]
        summary: dict[str, Any] = {
            "problems": per_problem,
            "geomean_speedup": geomean(speedups),
            "geomean_over": len(speedups),
            "problems_total": len(per_problem),
            "candidates": len(self.order),
            "compile_failed": sum(s["compile_failed"] for s in per_problem),
            "invalid": sum(s["invalid"] for s in per_problem),
            "unverified": sum(s["unverified"] for s in per_problem),
            "wall_s": round(time.time() - self.started, 2),
            "simulated": self.backend.simulated,
            "backend": self.backend.describe(),
        }
        cache = getattr(self.backend, "cache", None)
        if cache is not None:
            summary["cache"] = cache.stats()
        if extra:
            summary.update(extra)
        self.store.finish_experiment(self.exp_id, status, summary)
        log.info(
            "experiment finished", extra=kv(id=self.exp_id, geomean=summary["geomean_speedup"])
        )
        return summary
