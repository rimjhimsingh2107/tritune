from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, fields
from typing import Any

import numpy as np

from tritune.candidate import Candidate
from tritune.correctness import ValidationResult


def _from_dict(cls, d: dict[str, Any]):
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in names})


@dataclass
class CompileResult:
    ok: bool
    error: str | None = None
    # syntax | lint | resources | triton | timeout | crash | unsupported
    error_kind: str | None = None
    log: str = ""
    compile_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    ptx: str | None = None
    simulated: bool = False
    cached: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CompileResult:
        return _from_dict(cls, d)


@dataclass
class BenchmarkResult:
    ok: bool
    error: str | None = None
    samples_ms: list[float] = field(default_factory=list)
    p50_ms: float | None = None
    p95_ms: float | None = None
    mean_ms: float | None = None
    std_ms: float | None = None
    min_ms: float | None = None
    max_ms: float | None = None
    warmup: int = 0
    method: str = ""
    simulated: bool = False
    cached: bool = False

    @property
    def n(self) -> int:
        return len(self.samples_ms)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BenchmarkResult:
        return _from_dict(cls, d)

    @classmethod
    def from_samples(cls, samples: list[float], **kw: Any) -> BenchmarkResult:
        if not samples:
            return cls(ok=False, error="no timing samples", **kw)
        a = np.asarray(samples, dtype=np.float64)
        return cls(
            ok=True,
            samples_ms=[float(x) for x in a],
            p50_ms=float(np.percentile(a, 50)),
            p95_ms=float(np.percentile(a, 95)),
            mean_ms=float(a.mean()),
            std_ms=float(a.std(ddof=1)) if a.size > 1 else 0.0,
            min_ms=float(a.min()),
            max_ms=float(a.max()),
            **kw,
        )


@dataclass
class ProfileResult:
    ok: bool
    error: str | None = None
    tool: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    units: dict[str, str] = field(default_factory=dict)
    simulated: bool = False
    raw: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ProfileResult:
        return _from_dict(cls, d)


class Backend(ABC):
    name: str
    simulated: bool

    @abstractmethod
    def fingerprint(self) -> dict[str, Any]:
        """Hardware + software identity used in cache keys and experiment records."""

    @abstractmethod
    def compile(self, cand: Candidate) -> CompileResult: ...

    @abstractmethod
    def validate(self, cand: Candidate, seed: int = 0) -> ValidationResult: ...

    @abstractmethod
    def benchmark(self, cand: Candidate, warmup: int, reps: int) -> BenchmarkResult: ...

    @abstractmethod
    def profile(self, cand: Candidate) -> ProfileResult: ...

    def describe(self) -> str:
        return self.name
