"""The agent loop: planner proposes tool calls, runner executes them under a budget."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from kernelpilot.agent.planner import Planner
from kernelpilot.agent.tools import AgentState, ToolRegistry, ToolResult
from kernelpilot.log import get_logger, kv
from kernelpilot.problem import Problem
from kernelpilot.session import Session

log = get_logger("agent")


@dataclass
class AgentBudget:
    max_candidates: int = 12
    max_steps: int = 150
    max_seconds: float = 1800.0
    max_consecutive_errors: int = 5
    grace_steps: int = 8  # steps allowed after the candidate budget runs out


@dataclass
class AgentOutcome:
    problem: Problem
    stop_reason: str
    steps: int
    tool_calls: int
    candidates: int
    best_id: str | None
    speedup: float | None


BRIEFING = """\
Optimize the Triton kernel for {problem}.

Baseline candidate: {baseline_id} (untuned Triton: {baseline_label}).
Backend: {backend}.
Budget: {max_candidates} new candidates, {max_steps} turns.

Work in a loop: observe (metadata, profile, PTX), form one hypothesis, generate
a candidate, compile it, validate it, benchmark it, then decide what to try next
from what you measured. Fix compile or numerical failures before optimizing
further. benchmark_candidate decides whether a candidate beats the incumbent.
Call save_experiment with a short summary when you're done or out of budget.
"""


class AgentRunner:
    def __init__(self, session: Session, planner: Planner, budget: AgentBudget | None = None):
        self.session = session
        self.planner = planner
        self.budget = budget or AgentBudget()

    def run(self, problem: Problem) -> AgentOutcome:
        b = self.budget
        base = self.session.register_baseline(problem)
        state = AgentState(self.session, problem, b.max_candidates, baseline_id=base.candidate.id)
        registry = ToolRegistry(state)
        briefing = BRIEFING.format(
            problem=problem.key,
            baseline_id=base.candidate.id,
            baseline_label=base.candidate.label,
            backend=self.session.backend.describe(),
            max_candidates=b.max_candidates,
            max_steps=b.max_steps,
        )
        context: dict[str, Any] = {"baseline_id": base.candidate.id, "problem": problem.key}
        self.session.decide(
            "start",
            problem=problem.key,
            candidate_id=base.candidate.id,
            rationale=f"agent run with planner={self.planner.name}",
            args={"budget": b.__dict__},
        )
        self.planner.start(briefing, registry.schemas(), context)

        started = time.time()
        results: list[ToolResult] = []
        errors = idle = steps = 0
        grace = b.grace_steps
        reason = "max steps reached"
        while steps < b.max_steps:
            steps += 1
            if time.time() - started > b.max_seconds:
                reason = f"time budget of {b.max_seconds:.0f}s exhausted"
                break
            if state.created >= b.max_candidates:
                grace -= 1
                if grace < 0:
                    reason = "candidate budget exhausted"
                    break
            try:
                turn = self.planner.next_turn(results)
            except Exception as exc:
                log.exception("planner failed", extra=kv(planner=self.planner.name))
                self.session.decide("error", problem=problem.key, rationale=f"planner error: {exc}")
                reason = f"planner error: {type(exc).__name__}"
                break

            state.thought = turn.text.strip()
            if state.thought:
                self.session.decide("thought", problem=problem.key, rationale=state.thought)
            if not turn.calls:
                if turn.done:
                    reason = "planner finished"
                    break
                idle += 1
                if idle >= 3:
                    reason = "planner stopped calling tools"
                    break
                results = []
                continue
            idle = 0
            results = [registry.call(c.name, c.args, c.id) for c in turn.calls]
            errors = errors + 1 if all(not r.ok for r in results) else 0
            if errors >= b.max_consecutive_errors:
                reason = f"{errors} consecutive failed tool calls"
                self.session.decide("error", problem=problem.key, rationale=reason)
                break
            if state.done:
                reason = "saved by planner"
                break

        if not state.done:
            registry.call(
                "save_experiment",
                {"summary": f"Run stopped ({reason}); kept the best verified candidate."},
            )
        # the baseline is the reference for every speedup, make sure it got measured
        if base.bench is None:
            self.session.benchmark(base)

        best = self.session.best(problem)
        outcome = AgentOutcome(
            problem=problem,
            stop_reason=reason,
            steps=steps,
            tool_calls=registry.calls,
            candidates=state.created,
            best_id=best.candidate.id if best else None,
            speedup=self.session.speedup(best) if best else None,
        )
        log.info(
            "agent done", extra=kv(problem=problem.key, reason=reason, speedup=outcome.speedup)
        )
        return outcome
