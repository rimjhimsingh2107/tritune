"""Top-level entry points shared by the CLI, the API server and the demo."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from kernelpilot.agent.llm import LLMPlanner, llm_available, make_provider
from kernelpilot.agent.planner import HeuristicPlanner, Planner
from kernelpilot.agent.runner import AgentBudget, AgentRunner
from kernelpilot.backends import Backend, make_backend
from kernelpilot.config import Settings
from kernelpilot.env import Environment, detect
from kernelpilot.problem import Problem
from kernelpilot.session import Evaluation, Session
from kernelpilot.store import Store
from kernelpilot.tuner import STRATEGIES, run_search

ALL_STRATEGIES = ("agent", *STRATEGIES)


@dataclass
class Context:
    settings: Settings
    env: Environment
    store: Store
    backend: Backend

    @classmethod
    def create(cls, settings: Settings, env: Environment | None = None) -> Context:
        settings.ensure_home()
        env = env or detect()
        backend = make_backend(settings, env)
        return cls(settings, env, Store(settings.db_path), backend)


def make_planner(settings: Settings, kind: str = "auto", opening: list[dict[str, Any]] | None = None) -> Planner:
    if kind not in ("auto", "heuristic", "llm"):
        raise ValueError(f"planner must be auto, heuristic or llm (got {kind!r})")
    ok, why = llm_available(settings.llm_provider)
    if kind == "llm" and not ok:
        raise RuntimeError(f"LLM planner requested but unavailable: {why}")
    if kind == "llm" or (kind == "auto" and ok and not opening):
        return LLMPlanner(make_provider(settings.llm_provider, settings.llm_model or None))
    return HeuristicPlanner(opening=opening)


def run_experiment(
    ctx: Context,
    problems: list[Problem],
    strategy: str,
    budget: int,
    *,
    name: str | None = None,
    planner: str = "auto",
    opening: list[dict[str, Any]] | None = None,
    on_trial: Callable[[Evaluation], None] | None = None,
    extra_params: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    if strategy not in ALL_STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; options: {', '.join(ALL_STRATEGIES)}")
    if not problems:
        raise ValueError("no problems to optimize")
    params: dict[str, Any] = {"budget": budget, **(extra_params or {})}
    planner_obj = make_planner(ctx.settings, planner, opening) if strategy == "agent" else None
    if planner_obj is not None:
        params["planner"] = planner_obj.name

    label = name or f"{strategy} on {problems[0].key}" + (f" (+{len(problems) - 1})" if len(problems) > 1 else "")
    session = Session(ctx.store, ctx.backend, ctx.settings, ctx.env, name=label, strategy=strategy,
                      target=problems, params=params)
    status = "done"
    outcomes = []
    try:
        for p in problems:
            if strategy == "agent":
                # fresh planner state per problem; scripted openings only apply to the first
                pl = planner_obj if p is problems[0] else make_planner(ctx.settings, planner, None)
                out = AgentRunner(session, pl, AgentBudget(max_candidates=budget)).run(p)
                outcomes.append({"problem": p.key, "stop_reason": out.stop_reason, "steps": out.steps,
                                 "tool_calls": out.tool_calls})
            else:
                run_search(session, p, strategy, budget, on_trial=on_trial)
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    except Exception as exc:
        status = "failed"
        session.decide("error", rationale=f"experiment aborted: {type(exc).__name__}: {exc}")
        raise
    finally:
        summary = session.finish(status, {"agent_runs": outcomes} if outcomes else None)
    return session.exp_id, summary
