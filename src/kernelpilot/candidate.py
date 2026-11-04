from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from kernelpilot.kernels import get_spec
from kernelpilot.problem import KernelConfig, Problem, sha256


def new_id() -> str:
    return "c" + uuid.uuid4().hex[:11]


@dataclass
class Candidate:
    problem: Problem
    source: str
    config: KernelConfig | None = None
    origin: str = "manual"  # baseline | grid | random | halving | agent | manual
    parent_id: str | None = None
    note: str = ""
    id: str = field(default_factory=new_id)

    @property
    def source_hash(self) -> str:
        return sha256(self.source)

    @property
    def label(self) -> str:
        return self.config.short() if self.config else f"custom@{self.source_hash[:8]}"

    @classmethod
    def from_config(
        cls,
        problem: Problem,
        config: KernelConfig,
        origin: str = "manual",
        parent_id: str | None = None,
        note: str = "",
    ) -> Candidate:
        source = get_spec(problem.op).render(config)
        return cls(problem, source, config, origin, parent_id, note)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "problem": self.problem.to_dict(),
            "config": self.config.to_dict() if self.config else None,
            "origin": self.origin,
            "parent_id": self.parent_id,
            "note": self.note,
            "source_hash": self.source_hash,
        }
