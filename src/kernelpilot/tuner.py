"""Deterministic (non-LLM) search over the kernel config space.

This is the baseline the agent has to beat. It only knows the config grid and
the measured numbers -- no compiler feedback, no hypotheses.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from kernelpilot.candidate import Candidate
from kernelpilot.kernels import get_spec
from kernelpilot.log import get_logger, kv
from kernelpilot.problem import KernelConfig, Problem
from kernelpilot.session import Evaluation, Session

log = get_logger("tuner")

STRATEGIES = ("grid", "random", "halving")


def _space(problem: Problem, exclude: set[str]) -> list[KernelConfig]:
    return [c for c in get_spec(problem.op).configs(problem) if c.key not in exclude]


def grid_configs(
    problem: Problem, budget: int, seed: int = 0, exclude: set[str] | None = None
) -> list[KernelConfig]:
    """Evenly strided walk over the full cartesian grid.

    Truncating the grid would only ever visit the first variant / smallest
    tiles; striding keeps coverage when the budget is smaller than the grid.
    """
    space = _space(problem, exclude or set())
    if budget >= len(space):
        return space
    idx = np.unique(np.linspace(0, len(space) - 1, budget).round().astype(int))
    return [space[i] for i in idx]


def random_configs(
    problem: Problem, budget: int, seed: int = 0, exclude: set[str] | None = None
) -> list[KernelConfig]:
    space = _space(problem, exclude or set())
    rng = np.random.default_rng([seed, len(space), len(problem.op)])
    if budget >= len(space):
        order = rng.permutation(len(space))
    else:
        order = rng.choice(len(space), size=budget, replace=False)
    return [space[i] for i in order]


@dataclass
class SearchResult:
    problem: Problem
    strategy: str
    baseline: Evaluation
    best: Evaluation | None
    trials: list[Evaluation]

    @property
    def speedup(self) -> float | None:
        if self.best is None or self.baseline.p50 is None or self.best.p50 is None:
            return None
        return self.baseline.p50 / self.best.p50


def _trial(
    session: Session,
    problem: Problem,
    cfg: KernelConfig,
    origin: str,
    why: str,
    reps: int | None = None,
) -> Evaluation:
    cand = Candidate.from_config(problem, cfg, origin=origin, note=why)
    ev = session.evaluate(cand, reps=reps)
    session.decide(
        "trial",
        problem=problem.key,
        tool="evaluate",
        candidate_id=cand.id,
        args={"config": cfg.to_dict()},
        result=ev.brief(),
        rationale=why,
    )
    return ev


def successive_halving(
    session: Session,
    problem: Problem,
    configs: list[KernelConfig],
    eta: int = 3,
    max_reps: int | None = None,
) -> list[Evaluation]:
    """Cheap timing for everyone, more reps for the survivors.

    Rung r benchmarks with min_reps * eta**r reps and keeps the best 1/eta.
    Compile + validation happen once per config regardless of rung.
    """
    max_reps = max_reps or int(session.params["reps"])
    rungs = max(1, math.ceil(math.log(max(len(configs), 1), eta)))
    reps = max(3, max_reps // eta ** (rungs - 1))
    alive: list[Evaluation] = []
    all_evals = []
    for i, cfg in enumerate(configs):
        ev = _trial(
            session, problem, cfg, "halving", f"rung 0: {i + 1}/{len(configs)} at {reps} reps", reps
        )
        all_evals.append(ev)
        if ev.measured:
            alive.append(ev)

    rung = 0
    while len(alive) > 1 and reps < max_reps:
        alive.sort(key=lambda e: e.p50)
        keep = max(1, math.ceil(len(alive) / eta))
        dropped = alive[keep:]
        alive = alive[:keep]
        rung += 1
        reps = min(max_reps, reps * eta)
        session.decide(
            "prune",
            problem=problem.key,
            rationale=(
                f"rung {rung}: kept {keep}, dropped {len(dropped)}; re-timing survivors at {reps} reps"
            ),
            result={
                "kept": [e.candidate.id for e in alive],
                "dropped": [e.candidate.id for e in dropped],
            },
        )
        for ev in alive:
            session.benchmark(ev, reps)
    return all_evals


def run_search(
    session: Session,
    problem: Problem,
    strategy: str,
    budget: int,
    on_trial: Callable[[Evaluation], None] | None = None,
) -> SearchResult:
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; pick one of {STRATEGIES}")
    base = session.baseline(problem)
    session.decide(
        "baseline",
        problem=problem.key,
        candidate_id=base.candidate.id,
        result=base.brief(),
        rationale="untuned Triton baseline (reference point for speedup)",
    )
    exclude = {base.candidate.config.key} if base.candidate.config else set()
    seed = session.seed

    if strategy == "grid":
        configs = grid_configs(problem, budget, seed, exclude)
    else:
        configs = random_configs(problem, budget, seed, exclude)
    log.info("search", extra=kv(problem=problem.key, strategy=strategy, configs=len(configs)))

    if strategy == "halving":
        trials = successive_halving(session, problem, configs)
        if on_trial:
            for ev in trials:
                on_trial(ev)
    else:
        trials = []
        for i, cfg in enumerate(configs):
            ev = _trial(session, problem, cfg, strategy, f"{strategy} point {i + 1}/{len(configs)}")
            trials.append(ev)
            if on_trial:
                on_trial(ev)

    best = session.best(problem)
    return SearchResult(problem, strategy, base, best, trials)
