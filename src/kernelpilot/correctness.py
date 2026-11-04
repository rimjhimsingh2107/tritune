"""Numerical comparison against the float64 reference."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from kernelpilot.kernels.base import Tolerance


@dataclass
class Check:
    problem: str
    ok: bool
    max_abs_err: float
    max_rel_err: float
    mismatch_frac: float
    nan_count: int
    inf_count: int
    atol: float
    rtol: float
    message: str = ""
    worst_index: list[int] | None = None


@dataclass
class ValidationResult:
    ok: bool
    checks: list[Check] = field(default_factory=list)
    error: str | None = None
    # "verified" | "failed" | "error" | "unverified"
    status: str = "verified"

    @property
    def worst(self) -> Check | None:
        failing = [c for c in self.checks if not c.ok] or self.checks
        return max(failing, key=lambda c: c.max_abs_err, default=None)

    def summary(self) -> str:
        if self.error:
            return f"{self.status}: {self.error}"
        w = self.worst
        if w is None:
            return self.status
        verdict = "pass" if self.ok else "FAIL"
        return f"{verdict} ({w.problem}: max_abs={w.max_abs_err:.3g}, max_rel={w.max_rel_err:.3g}{', ' + w.message if w.message else ''})"

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "status": self.status,
            "error": self.error,
            "checks": [asdict(c) for c in self.checks],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ValidationResult:
        return cls(
            ok=d["ok"],
            status=d.get("status", "verified"),
            error=d.get("error"),
            checks=[Check(**c) for c in d.get("checks", [])],
        )


def compare(out: np.ndarray, ref: np.ndarray, tol: Tolerance, problem: str) -> Check:
    out = np.asarray(out)
    ref = np.asarray(ref, dtype=np.float64)
    if out.shape != ref.shape:
        return Check(
            problem,
            False,
            float("inf"),
            float("inf"),
            1.0,
            0,
            0,
            tol.atol,
            tol.rtol,
            message=f"shape mismatch: got {out.shape}, want {ref.shape}",
        )

    o = out.astype(np.float64)
    nan = int(np.isnan(o).sum())
    inf = int(np.isinf(o).sum())
    with np.errstate(invalid="ignore"):
        diff = np.abs(o - ref)
    diff = np.where(np.isfinite(diff), diff, np.inf)
    rel = diff / np.maximum(np.abs(ref), 1e-12)
    bad = diff > (tol.atol + tol.rtol * np.abs(ref))

    msg = []
    if nan:
        msg.append(f"{nan} NaN")
    if inf:
        msg.append(f"{inf} Inf")
    worst = None
    if diff.size:
        worst = [int(i) for i in np.unravel_index(int(np.argmax(diff)), diff.shape)]
    return Check(
        problem=problem,
        ok=not bool(bad.any()),
        max_abs_err=float(diff.max()) if diff.size else 0.0,
        max_rel_err=float(rel.max()) if rel.size else 0.0,
        mismatch_frac=float(bad.mean()) if bad.size else 0.0,
        nan_count=nan,
        inf_count=inf,
        atol=tol.atol,
        rtol=tol.rtol,
        message=", ".join(msg),
        worst_index=worst,
    )
