"""Assemble the optimization trace: one entry per candidate, in the order the
optimizer produced them, with the reasoning that led to it and what happened."""

from __future__ import annotations

from typing import Any

from kernelpilot.store import Store


def build_trace(store: Store, exp_id: str) -> dict[str, Any]:
    exp = store.experiment(exp_id)
    if exp is None:
        raise KeyError(exp_id)
    cands = store.candidates(exp["id"])
    decisions = store.decisions(exp["id"])

    by_cand: dict[str, list[dict[str, Any]]] = {}
    thoughts: list[dict[str, Any]] = []
    for d in decisions:
        if d["kind"] == "thought":
            thoughts.append({"step": d["step"], "problem": d["problem"], "text": d["rationale"]})
        elif d["candidate_id"] and d["kind"] != "tool_call":
            by_cand.setdefault(d["candidate_id"], []).append(d)

    tool_calls = [d for d in decisions if d["kind"] == "tool_call"]
    entries = []
    for c in cands:
        ds = by_cand.get(c["id"], [])
        hyp = next((d for d in ds if d["kind"] == "hypothesis"), None)
        trial = next((d for d in ds if d["kind"] == "trial"), None)
        verdicts = [d for d in ds if d["kind"] in ("accept", "reject")]
        verdict = verdicts[-1] if verdicts else None
        val = c.get("validation") or {}
        worst = None
        if val.get("checks"):
            worst = max(val["checks"], key=lambda x: x["max_abs_err"])
        entries.append(
            {
                "id": c["id"],
                "seq": c["seq"],
                "problem": c["problem"],
                "label": c["label"],
                "origin": c["origin"],
                "parent_id": c["parent_id"],
                "config": c["config"],
                "hypothesis": (hyp or {}).get("rationale")
                or (trial or {}).get("rationale")
                or c["note"],
                "rationale": ((hyp or {}).get("args") or {}).get("rationale"),
                "changes": ((hyp or {}).get("args") or {}).get("changes"),
                "evidence": (hyp or {}).get("evidence"),
                "step": (hyp or trial or {}).get("step"),
                "compile": {
                    "ok": None if c["compile_ok"] is None else bool(c["compile_ok"]),
                    "error": c["compile_error"],
                    "kind": c["compile_error_kind"],
                    "ms": c["compile_ms"],
                    "cached": bool(c["compile_cached"])
                    if c["compile_cached"] is not None
                    else None,
                },
                "validation": {
                    "status": c["valid_status"],
                    "error": val.get("error"),
                    "worst": worst,
                },
                "benchmark": {
                    "p50_ms": c["p50_ms"],
                    "p95_ms": c["p95_ms"],
                    "min_ms": c["min_ms"],
                    "max_ms": c["max_ms"],
                    "samples": c["samples"],
                },
                "metadata": c["metadata"],
                "profile": (c.get("profile") or {}).get("metrics"),
                "status": c["status"],
                "speedup": c["speedup"],
                "retained": bool(c["retained"]),
                "verdict": {
                    "kind": verdict["kind"],
                    "reason": verdict["rationale"],
                    "evidence": verdict["evidence"],
                }
                if verdict
                else None,
            }
        )
    return {
        "experiment": {
            k: exp[k] for k in ("id", "name", "strategy", "status", "simulated", "backend")
        },
        "entries": entries,
        "thoughts": thoughts,
        "tool_calls": len(tool_calls),
        "final": [
            {"problem": d["problem"], "summary": d["rationale"], "candidate_id": d["candidate_id"]}
            for d in decisions
            if d["kind"] == "final"
        ],
    }
