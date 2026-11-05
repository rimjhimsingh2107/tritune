"""Side-by-side comparison of two experiments (usually agent vs autotuner)."""

from __future__ import annotations

from typing import Any

from tritune.stats import geomean
from tritune.store import Store


def best_so_far(cands: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Speedup of the best verified candidate after each evaluation."""
    curve = []
    base = next((c for c in cands if c["origin"] == "baseline" and c["p50_ms"]), None)
    best = None
    for i, c in enumerate(cands, 1):
        if c["status"] == "measured" and c["p50_ms"] and (best is None or c["p50_ms"] < best):
            best = c["p50_ms"]
        curve.append(
            {
                "n": i,
                "best_p50_ms": best,
                "speedup": (base["p50_ms"] / best) if (base and best) else None,
            }
        )
    return curve


def experiment_stats(store: Store, exp_id: str) -> dict[str, Any]:
    exp = store.experiment(exp_id)
    if exp is None:
        raise KeyError(f"no experiment {exp_id!r}")
    summary = exp.get("summary") or {}
    out: dict[str, Any] = {
        "id": exp["id"],
        "name": exp["name"],
        "strategy": exp["strategy"],
        "simulated": bool(exp["simulated"]),
        "geomean_speedup": summary.get("geomean_speedup"),
        "candidates": summary.get("candidates"),
        "problems": {},
    }
    for p in summary.get("problems", []):
        cands = store.candidates(exp["id"], p["problem"])
        out["problems"][p["problem"]] = {**p, "curve": best_so_far(cands)}
    return out


def compare_experiments(store: Store, a_id: str, b_id: str) -> dict[str, Any]:
    a = experiment_stats(store, a_id)
    b = experiment_stats(store, b_id)
    shared = [k for k in a["problems"] if k in b["problems"]]
    rows = []
    for key in shared:
        pa, pb = a["problems"][key], b["problems"][key]
        sa, sb = pa.get("speedup"), pb.get("speedup")
        if sa and sb:
            winner = (
                "tie" if abs(sa - sb) / max(sa, sb) < 0.01 else (a["id"] if sa > sb else b["id"])
            )
        else:
            winner = a["id"] if sa else (b["id"] if sb else None)
        rows.append(
            {
                "problem": key,
                "a_speedup": sa,
                "b_speedup": sb,
                "a_best": pa.get("best_label"),
                "b_best": pb.get("best_label"),
                "a_evaluated": pa.get("evaluated"),
                "b_evaluated": pb.get("evaluated"),
                "a_evals_to_best": pa.get("evals_to_best"),
                "b_evals_to_best": pb.get("evals_to_best"),
                "a_failures": (pa.get("compile_failed") or 0) + (pa.get("invalid") or 0),
                "b_failures": (pb.get("compile_failed") or 0) + (pb.get("invalid") or 0),
                "a_curve": pa["curve"],
                "b_curve": pb["curve"],
                "winner": winner,
            }
        )
    return {
        "a": {k: v for k, v in a.items() if k != "problems"},
        "b": {k: v for k, v in b.items() if k != "problems"},
        "shared_problems": len(shared),
        "a_geomean_shared": geomean(r["a_speedup"] for r in rows),
        "b_geomean_shared": geomean(r["b_speedup"] for r in rows),
        "a_wins": sum(r["winner"] == a["id"] for r in rows),
        "b_wins": sum(r["winner"] == b["id"] for r in rows),
        "ties": sum(r["winner"] == "tie" for r in rows),
        "rows": rows,
        "mixed_backends": a["simulated"] != b["simulated"],
    }
