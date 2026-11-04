"""Rule-based optimization advisor.

Maps observations to concrete config changes with a hypothesis attached.
The heuristic planner follows these directly; the LLM planner gets them as
one tool among many and is free to ignore them.

Rules are deliberately boring and explainable. Each one says what it saw,
what it wants to change, and what should happen if the hypothesis is right.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kernelpilot.analysis.observations import Observation
from kernelpilot.kernels import get_spec
from kernelpilot.problem import KernelConfig, Problem, next_pow2


@dataclass
class Recommendation:
    config: KernelConfig
    changes: dict[str, Any]
    hypothesis: str
    rationale: str
    priority: float
    based_on: str  # "incumbent" | "failed"
    evidence: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "changes": self.changes,
            "config": self.config.to_dict(),
            "label": self.config.short(),
            "hypothesis": self.hypothesis,
            "rationale": self.rationale,
            "priority": round(self.priority, 2),
            "based_on": self.based_on,
            "evidence": self.evidence,
        }


def _by_kind(obs: list[Observation]) -> dict[str, Observation]:
    return {o.kind: o for o in obs}


class _Builder:
    def __init__(self, problem: Problem, tried: set[str], avoid: list[dict[str, Any]] | None = None):
        self.avoid = avoid or []
        self.problem = problem
        self.spec = get_spec(problem.op)
        self.tried = tried
        self.out: list[Recommendation] = []
        self._seen: set[str] = set()

    def add(self, base: KernelConfig, changes: dict[str, Any], hypothesis: str, rationale: str,
            priority: float, based_on: str, cites: list[Observation] | None = None) -> None:
        cfg = base.replace(**changes)
        if cfg.key == base.key or cfg.key in self.tried or cfg.key in self._seen:
            return
        if based_on == "incumbent" and changes in self.avoid:
            return  # same edit already measured slower once
        if self.spec.check(cfg, self.problem):
            return
        self._seen.add(cfg.key)
        self.out.append(Recommendation(
            cfg, changes, hypothesis, rationale, priority, based_on,
            [{"kind": o.kind, **o.evidence} for o in (cites or [])],
        ))


def _repairs(b: _Builder, failed: KernelConfig, obs: dict[str, Observation]) -> None:
    p, n = failed.params, b.problem
    if o := obs.get("compile_out_of_resources"):
        req, lim = o.evidence.get("required"), o.evidence.get("limit")
        over = f" ({req} B needed vs {lim} B available)" if req and lim else ""
        if failed.num_stages > 2:
            b.add(failed, {"num_stages": failed.num_stages - 1},
                  "One fewer pipeline stage should fit in shared memory and keep the tile size",
                  f"compile failed on shared memory{over}; each stage holds a full A and B tile", 0.98, "failed", [o])
        if p.get("BLOCK_K", 0) >= 64:
            b.add(failed, {"BLOCK_K": p["BLOCK_K"] // 2},
                  "Halving BLOCK_K halves the per-stage shared memory footprint",
                  f"compile failed on shared memory{over}", 0.95, "failed", [o])
        big = max(("BLOCK_M", "BLOCK_N"), key=lambda k: p.get(k, 0))
        if p.get(big, 0) > 64:
            b.add(failed, {big: p[big] // 2},
                  f"Shrinking {big} reduces both shared memory and accumulator registers",
                  f"compile failed on shared memory{over}", 0.9, "failed", [o])

    if o := obs.get("compile_non_pow2"):
        for k, v in p.items():
            if k.startswith("BLOCK") and v & (v - 1):
                b.add(failed, {k: next_pow2(v)}, f"Round {k}={v} up to {next_pow2(v)}",
                      "tl.arange extents must be powers of two", 0.97, "failed", [o])
                b.add(failed, {k: next_pow2(v) // 2}, f"Round {k}={v} down to {next_pow2(v) // 2}",
                      "tl.arange extents must be powers of two", 0.9, "failed", [o])

    if (o := obs.get("launch_constraint")) and n.op == "softmax":
        cols = n.shape[1]
        b.add(failed, {"variant": "online", "BLOCK_N": min(4096, next_pow2(cols))},
              "The chunked online variant handles rows wider than one block",
              f"single-pass needs BLOCK_N >= {cols}", 0.95, "failed", [o])
        b.add(failed, {"BLOCK_N": next_pow2(cols)}, "Make the block cover the whole row",
              f"single-pass needs BLOCK_N >= {cols}", 0.9, "failed", [o])

    if o := obs.get("numerical_nonfinite"):
        if failed.variant == "naive":
            b.add(failed, {"variant": "single_pass"},
                  "Restore the max subtraction and fp32 intermediates; exp() of raw fp16 logits overflows",
                  f"validation produced {o.evidence.get('nan', 0)} NaN / {o.evidence.get('inf', 0)} Inf",
                  0.99, "failed", [o])
            b.add(failed, {"variant": "online", "BLOCK_N": min(failed.params["BLOCK_N"], 2048)},
                  "Online softmax keeps a running max, so it is overflow-safe and works for any width",
                  "validation produced non-finite values", 0.9, "failed", [o])

    if o := obs.get("numerical_tolerance"):
        if failed.variant == "fp16acc":
            fix = "grouped" if p.get("GROUP_M", 1) > 1 else "tiled"
            b.add(failed, {"variant": fix},
                  "Accumulate in fp32; fp16 running sums lose bits once |acc| grows with K",
                  f"max abs error {o.evidence.get('max_abs_err', 0):.3g} vs atol {o.evidence.get('atol', 0):.3g}",
                  0.99, "failed", [o])
        if failed.variant == "atomic_fp16acc":
            b.add(failed, {"variant": "atomic"}, "Keep the per-lane accumulator in fp32",
                  f"relative error {o.evidence.get('max_rel_err', 0):.2e} exceeds rtol "
                  f"{o.evidence.get('rtol', 0):.0e}", 0.99, "failed", [o])

    if (o := obs.get("numerical_tail")) and failed.variant == "unmasked":
        b.add(failed, {"variant": "masked"}, "Put the bounds mask back; the tail block reads past the end",
              f"{o.evidence.get('mismatch_frac', 0):.2%} of elements wrong, all in the last block",
              0.99, "failed", [o])


def _improvements(b: _Builder, cfg: KernelConfig, obs: dict[str, Observation]) -> None:
    op, p, prob = b.problem.op, cfg.params, b.problem
    w, s = cfg.num_warps, cfg.num_stages

    pressure = obs.get("spills") or obs.get("register_pressure")
    low_occ = obs.get("low_occupancy")
    if pressure or (low_occ and low_occ.evidence.get("limiter") == "registers"):
        why = pressure or low_occ
        regs = why.evidence.get("n_regs")
        detail = f"n_regs={regs}" + (f", n_spills={why.evidence['n_spills']}" if "n_spills" in why.evidence else "")
        if op == "matmul":
            big = max(("BLOCK_M", "BLOCK_N"), key=lambda k: p[k])
            b.add(cfg, {big: p[big] // 2}, f"A smaller {big} shrinks the fp32 accumulator per thread",
                  f"register pressure ({detail})", 0.9, "incumbent", [why])
            b.add(cfg, {"num_warps": min(8, w * 2)}, "More warps split the accumulator across more threads",
                  f"register pressure ({detail})", 0.85, "incumbent", [why])
        else:
            b.add(cfg, {"num_warps": min(32, w * 2)},
                  "Doubling num_warps halves the elements each thread holds, which should remove the spills",
                  f"register pressure ({detail})", 0.95, "incumbent", [why])
            if op == "softmax" and cfg.variant == "single_pass":
                chunk = min(4096, max(256, next_pow2(prob.shape[1]) // 4))
                b.add(cfg, {"variant": "online", "BLOCK_N": chunk},
                      "Stream the row in chunks with online softmax instead of holding it all in registers",
                      f"whole {prob.shape[1]}-wide row lives in registers ({detail})", 0.8, "incumbent", [why])
    elif low_occ and low_occ.evidence.get("limiter") == "shared memory" and s > 1:
        b.add(cfg, {"num_stages": s - 1}, "Fewer stages frees shared memory so more blocks fit per SM",
              f"occupancy {low_occ.evidence['occupancy_pct']}% limited by shared memory", 0.8, "incumbent", [low_occ])

    if op == "matmul":
        if o := obs.get("small_tiles"):
            grow = {k: 128 for k in ("BLOCK_M", "BLOCK_N") if p[k] < 128}
            b.add(cfg, {**grow, "num_warps": 8},
                  "128x128 tiles double operand reuse per load; 8 warps keep the MMA units fed",
                  f"{p['BLOCK_M']}x{p['BLOCK_N']} tile reads each A/B element too many times", 0.85, "incumbent", [o])
            k = min(grow, key=lambda k: p[k]) if grow else None
            if k:
                b.add(cfg, {k: p[k] * 2}, f"Grow {k} only, a smaller step in arithmetic intensity",
                      "small output tile", 0.6, "incumbent", [o])
        if o := obs.get("row_major_order"):
            b.add(cfg, {"variant": "grouped", "GROUP_M": 8},
                  "Grouped tile order keeps a band of B resident in L2 across consecutive programs",
                  f"row-major order with N={prob.shape[1]}", 0.7, "incumbent", [o])
        if (o := obs.get("shallow_pipeline")) and s < 4:
            b.add(cfg, {"num_stages": s + 1}, "One more stage of cp.async prefetch to hide global load latency",
                  f"num_stages={s}", 0.65, "incumbent", [o])
        if (o := obs.get("latency_bound")) and s < 4:
            b.add(cfg, {"num_stages": s + 1}, "Neither memory nor compute is saturated; deepen the pipeline",
                  f"DRAM {o.evidence['dram_pct']:.0f}%, SM {o.evidence['sm_pct']:.0f}%", 0.6, "incumbent", [o])
    else:
        if (o := obs.get("atomic_contention")) and op == "reduction" and cfg.variant == "atomic":
            tiles = p.get("TILES", 1)
            if tiles < 16:
                b.add(cfg, {"TILES": min(16, tiles * 4)},
                      "Fold more tiles per program so there are 4x fewer same-address atomics",
                      f"{o.evidence['atomics']} atomics on one address per call", 0.88, "incumbent", [o])
            b.add(cfg, {"variant": "two_pass"}, "Write partials and reduce them in a second pass, no atomics at all",
                  f"{o.evidence['atomics']} atomics per call", 0.5, "incumbent", [o])
        if (o := obs.get("thin_threads")) and w > 1:
            b.add(cfg, {"num_warps": w // 2}, "Fewer warps so each thread gets enough elements to vectorize",
                  f"{o.evidence['elements_per_thread']:.1f} elements per thread", 0.6, "incumbent", [o])
        if (o := obs.get("fat_threads")) and w < 16:
            b.add(cfg, {"num_warps": w * 2}, "More warps per program for more memory-level parallelism",
                  f"{o.evidence['elements_per_thread']:.0f} elements per thread", 0.6, "incumbent", [o])
        if (o := obs.get("latency_bound")) and w < 16:
            b.add(cfg, {"num_warps": w * 2}, "More warps in flight to cover DRAM latency",
                  f"DRAM {o.evidence['dram_pct']:.0f}%, SM {o.evidence['sm_pct']:.0f}%", 0.55, "incumbent", [o])


def _neighbors(b: _Builder, cfg: KernelConfig, hot: list[str]) -> None:
    """Local search: one step along each axis of the config space."""
    space = b.spec.space(b.problem).get(cfg.variant, {})
    order = list(space)
    # parameters that already paid off go first
    axes = sorted(order, key=lambda a: (a not in hot, order.index(a)))
    for axis in axes:
        values = sorted(space[axis])
        cur = cfg.num_warps if axis == "num_warps" else cfg.num_stages if axis == "num_stages" else cfg.params.get(axis)
        if cur not in values:
            continue
        i = values.index(cur)
        for j in (i + 1, i - 1):
            if 0 <= j < len(values):
                b.add(cfg, {axis: values[j]}, f"Local refinement: {axis} {cur} -> {values[j]}",
                      "no specific bottleneck left to target; probing a neighbouring config",
                      0.3 if axis in hot else 0.2, "incumbent")
    for other in b.spec.space(b.problem):
        if other != cfg.variant:
            grid = b.spec.space(b.problem)[other]
            changes: dict[str, Any] = {"variant": other}
            for k, vals in grid.items():
                if k in ("num_warps", "num_stages"):
                    continue
                if cfg.params.get(k) not in vals:
                    changes[k] = vals[len(vals) // 2]
            b.add(cfg, changes, f"Try the {other} variant with similar parameters",
                  "structural alternative not explored yet", 0.25, "incumbent")


def recommend(
    problem: Problem,
    incumbent: KernelConfig | None,
    incumbent_obs: list[Observation],
    failed: KernelConfig | None = None,
    failed_obs: list[Observation] | None = None,
    tried: set[str] | None = None,
    hot_params: list[str] | None = None,
    avoid: list[dict[str, Any]] | None = None,
    limit: int = 6,
) -> list[Recommendation]:
    b = _Builder(problem, tried or set(), avoid)
    if failed is not None and failed_obs:
        _repairs(b, failed, _by_kind(failed_obs))
    if incumbent is not None:
        _improvements(b, incumbent, _by_kind(incumbent_obs))
        _neighbors(b, incumbent, hot_params or [])
    b.out.sort(key=lambda r: -r.priority)
    return b.out[:limit]
