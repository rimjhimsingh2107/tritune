"""Planners decide which tool to call next.

HeuristicPlanner needs no API key. It walks the observe -> hypothesize ->
modify -> compile -> validate -> benchmark -> analyze loop and takes its
hypotheses from the rule-based recommender. It talks to the runner in the
same shape an LLM does (text + tool calls per turn), so the runner, the
budget enforcement and the decision log are identical for both.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Protocol

from kernelpilot.agent.tools import ToolResult


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]
    id: str = ""


@dataclass
class Turn:
    text: str
    calls: list[ToolCall] = field(default_factory=list)
    done: bool = False


class Planner(Protocol):
    name: str

    def start(
        self, briefing: str, tools: list[dict[str, Any]], context: dict[str, Any]
    ) -> None: ...

    def next_turn(self, results: list[ToolResult]) -> Turn: ...


def _fmt_ms(v: float | None) -> str:
    if v is None:
        return "?"
    return f"{v * 1e3:.1f}us" if v < 0.1 else f"{v:.3f}ms"


class HeuristicPlanner:
    name = "heuristic"

    def __init__(self, opening: list[dict[str, Any]] | None = None, profile: bool = True):
        # opening: generate_candidate args to try before consulting the recommender.
        # the demo scenarios use this to steer the agent into a specific mistake.
        self.opening = list(opening or [])
        self.profile = profile
        self._ids = itertools.count(1)
        self.queue: list[tuple[str, ToolCall]] = []
        self.baseline_id = ""
        self.current = ""
        self.incumbent = ""
        self.recs: list[dict[str, Any]] | None = None  # None means: ask the advisor
        self.finished = False

    def _call(self, name: str, **args: Any) -> ToolCall:
        return ToolCall(name, args, f"h{next(self._ids)}")

    def start(self, briefing: str, tools: list[dict[str, Any]], context: dict[str, Any]) -> None:
        self.baseline_id = context["baseline_id"]
        self.incumbent = self.baseline_id
        b = self.baseline_id
        self.queue = [
            (
                "Start by looking at the baseline kernel and its launch config.",
                self._call("inspect_kernel", candidate_id=b),
            ),
            (
                "Compile the baseline to get register / shared memory numbers.",
                self._call("compile_candidate", candidate_id=b),
            ),
            (
                "Check the baseline is numerically correct before trusting any timing.",
                self._call("validate_correctness", candidate_id=b),
            ),
            (
                "Measure the baseline; every later candidate is compared against this.",
                self._call("benchmark_candidate", candidate_id=b),
            ),
        ]
        if self.profile:
            self.queue.append(
                (
                    "Profile the baseline to see what it's bound by.",
                    self._call("profile_candidate", candidate_id=b),
                )
            )

    # -- turn logic --------------------------------------------------------

    def next_turn(self, results: list[ToolResult]) -> Turn:
        for r in results:
            self._absorb(r)
        if self.finished:
            return Turn("Done.", done=True)
        if self.queue:
            text, call = self.queue.pop(0)
            return Turn(text, [call])
        return self._propose()

    def _absorb(self, r: ToolResult) -> None:
        c = r.content
        if r.name == "generate_candidate":
            if not r.ok:
                if "budget" in (r.error or ""):
                    self._finish("Candidate budget is used up.")
                return
            if c.get("duplicate_of"):
                return  # already evaluated; _propose will pick something else
            cid = c["candidate_id"]
            self.current = cid
            self.queue = [
                ("Compile the new candidate.", self._call("compile_candidate", candidate_id=cid)),
            ]
        elif r.name == "compile_candidate" and r.ok and c.get("candidate_id") == self.current:
            if c["ok"]:
                self.queue.insert(
                    0,
                    (
                        "It compiles. Validate before timing it.",
                        self._call("validate_correctness", candidate_id=self.current),
                    ),
                )
            else:
                self.queue.clear()  # back to proposing; the recommender sees the failure
        elif r.name == "validate_correctness" and r.ok and c.get("candidate_id") == self.current:
            if c["ok"]:
                self.queue.insert(
                    0,
                    (
                        "Numerics check out. Benchmark it against the incumbent.",
                        self._call("benchmark_candidate", candidate_id=self.current),
                    ),
                )
            else:
                self.queue.clear()
        elif r.name == "benchmark_candidate" and r.ok:
            if c.get("accepted") and c["candidate_id"] != self.baseline_id:
                self.incumbent = c["candidate_id"]
                if self.profile:
                    self.queue.insert(
                        0,
                        (
                            "New incumbent. Re-profile it so the next hypothesis uses fresh numbers.",
                            self._call("profile_candidate", candidate_id=self.incumbent),
                        ),
                    )
            self.incumbent = c.get("incumbent_id") or self.incumbent
        elif r.name == "recommend_optimization":
            self.recs = c.get("recommendations", []) if r.ok else []
        elif r.name == "save_experiment":
            self.finished = True

    def _finish(self, why: str) -> None:
        self.queue = [
            (
                f"{why} Wrapping up.",
                self._call(
                    "save_experiment", summary=why + " Kept the fastest verified candidate."
                ),
            )
        ]

    def _propose(self) -> Turn:
        if self.opening:
            args = self.opening.pop(0)
            return Turn(
                f"Trying an idea first: {args['hypothesis']}",
                [self._call("generate_candidate", **args)],
            )
        if self.recs is None:
            return Turn(
                "Ask the advisor what the current numbers point to.",
                [self._call("recommend_optimization", candidate_id=self.incumbent, limit=4)],
            )
        recs, self.recs = self.recs, None
        if not recs:
            return Turn(
                "No untried changes left that the observations justify.",
                [self._call("save_experiment", summary=self._summary_text())],
            )
        top = recs[0]
        cites = ", ".join(e["kind"] for e in top.get("evidence", [])) or "local search"
        text = f"{top['rationale']}. Hypothesis: {top['hypothesis']} [{cites}]"
        args = dict(top["generate_args"])
        args["evidence"] = top.get("evidence", [])
        return Turn(text, [self._call("generate_candidate", **args)])

    def _summary_text(self) -> str:
        return "Stopped when the advisor ran out of justified changes; kept the fastest verified candidate."
