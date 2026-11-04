"""Turn compiler / validator / profiler output into structured observations.

The agent (heuristic or LLM) reasons over these rather than raw logs. Each
observation carries the numbers that triggered it, so a later decision can
cite exactly what motivated it.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from kernelpilot.analysis.ptx import summarize_ptx
from kernelpilot.kernels import get_spec
from kernelpilot.problem import KernelConfig, Problem

LAUNCH_FLOOR_MS = 0.010  # under ~10us we're mostly paying launch latency


@dataclass
class Observation:
    kind: str
    message: str
    severity: str = "info"  # info | warn | error
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _smem_numbers(error: str) -> dict[str, int]:
    m = re.search(r"Required:\s*(\d+),\s*Hardware limit:\s*(\d+)", error or "")
    return {"required": int(m.group(1)), "limit": int(m.group(2))} if m else {}


def observe_failure(problem: Problem, ev) -> list[Observation]:
    """What went wrong with a candidate that didn't make it to benchmarking."""
    obs: list[Observation] = []
    comp = ev.compile
    if comp is not None and not comp.ok:
        err = comp.error or ""
        kind = comp.error_kind or "unknown"
        ev_data: dict[str, Any] = {"error_kind": kind, "error": err[:400]}
        if kind == "resources" or "out of resource" in err:
            ev_data.update(_smem_numbers(err))
            obs.append(
                Observation(
                    "compile_out_of_resources",
                    "shared memory demand exceeds the per-block limit",
                    "error",
                    ev_data,
                )
            )
        elif "power of 2" in err:
            obs.append(
                Observation(
                    "compile_non_pow2", "a block extent is not a power of two", "error", ev_data
                )
            )
        elif kind == "launch":
            obs.append(
                Observation(
                    "launch_constraint", "config violates a launch constraint", "error", ev_data
                )
            )
        elif kind == "syntax":
            obs.append(
                Observation("compile_syntax", "candidate source does not parse", "error", ev_data)
            )
        elif kind in ("timeout", "crash"):
            obs.append(Observation("compile_crash", f"compile step {kind}", "error", ev_data))
        else:
            obs.append(Observation("compile_error", "Triton rejected the kernel", "error", ev_data))
        return obs

    val = ev.validation
    if val is not None and not val.ok:
        if val.status == "unverified":
            obs.append(
                Observation(
                    "unverified",
                    "correctness could not be established",
                    "error",
                    {"error": val.error},
                )
            )
            return obs
        worst = val.worst
        data = {"status": val.status}
        if worst is not None:
            data.update(
                problem=worst.problem,
                max_abs_err=worst.max_abs_err,
                max_rel_err=worst.max_rel_err,
                nan=worst.nan_count,
                inf=worst.inf_count,
                mismatch_frac=worst.mismatch_frac,
                atol=worst.atol,
                rtol=worst.rtol,
            )
        if worst is not None and (worst.nan_count or worst.inf_count):
            obs.append(
                Observation(
                    "numerical_nonfinite",
                    "output contains NaN/Inf (overflow or 0/0)",
                    "error",
                    data,
                )
            )
        elif (
            worst is not None
            and worst.mismatch_frac < 0.05
            and worst.max_abs_err > 10 * (worst.atol + 1)
        ):
            obs.append(
                Observation(
                    "numerical_tail",
                    "a small slice of the output is garbage (bounds/masking?)",
                    "error",
                    data,
                )
            )
        else:
            obs.append(
                Observation(
                    "numerical_tolerance",
                    "error exceeds tolerance (precision loss?)",
                    "error",
                    data,
                )
            )
    return obs


def observe(problem: Problem, ev, config: KernelConfig | None) -> list[Observation]:
    """Observations for a measured candidate."""
    obs = observe_failure(problem, ev)
    if obs:
        return obs
    spec = get_spec(problem.op)
    md = (ev.compile.metadata if ev.compile else None) or {}
    prof = (ev.profile.metrics if ev.profile and ev.profile.ok else None) or {}

    regs = md.get("n_regs") or prof.get("registers_per_thread")
    spills = md.get("n_spills") or 0
    if spills:
        obs.append(
            Observation(
                "spills",
                f"{spills} registers spilled to local memory",
                "warn",
                {"n_spills": spills, "n_regs": regs},
            )
        )
    elif regs and regs >= 200:
        obs.append(
            Observation(
                "register_pressure",
                f"{regs} registers/thread, close to the 255 cap",
                "warn",
                {"n_regs": regs},
            )
        )

    occ = prof.get("achieved_occupancy_pct")
    occ_src = "profiler"
    if occ is None and md.get("est_occupancy") is not None:
        occ, occ_src = 100 * md["est_occupancy"], "estimate"
    if occ is not None and occ < 25:
        obs.append(
            Observation(
                "low_occupancy",
                f"occupancy {occ:.0f}% ({occ_src})",
                "warn",
                {
                    "occupancy_pct": round(occ, 1),
                    "source": occ_src,
                    "limiter": md.get("occupancy_limiter"),
                },
            )
        )

    dram, sm = prof.get("dram_throughput_pct"), prof.get("sm_throughput_pct")
    if dram is not None and sm is not None:
        if dram >= 60 and dram > sm:
            obs.append(
                Observation(
                    "memory_bound",
                    f"DRAM {dram:.0f}% of peak vs SM {sm:.0f}%",
                    "info",
                    {"dram_pct": dram, "sm_pct": sm},
                )
            )
        elif sm >= 60 and sm > dram:
            obs.append(
                Observation(
                    "compute_bound",
                    f"SM {sm:.0f}% of peak vs DRAM {dram:.0f}%",
                    "info",
                    {"dram_pct": dram, "sm_pct": sm},
                )
            )
        elif dram < 50 and sm < 50:
            obs.append(
                Observation(
                    "latency_bound",
                    f"neither DRAM ({dram:.0f}%) nor SM ({sm:.0f}%) is saturated",
                    "warn",
                    {"dram_pct": dram, "sm_pct": sm},
                )
            )

    if ev.p50 is not None and ev.p50 < LAUNCH_FLOOR_MS:
        obs.append(
            Observation(
                "launch_latency",
                f"p50 {ev.p50 * 1e3:.1f}us is near launch overhead",
                "info",
                {"p50_ms": ev.p50},
            )
        )

    if config is not None:
        cost = spec.cost(config, problem)
        if cost.lane_efficiency < 0.8:
            obs.append(
                Observation(
                    "padding_waste",
                    f"only {cost.lane_efficiency:.0%} of lanes do useful work",
                    "info",
                    {"lane_efficiency": round(cost.lane_efficiency, 3)},
                )
            )
        if cost.atomics > 2048:
            obs.append(
                Observation(
                    "atomic_contention",
                    f"{cost.atomics} same-address atomics per call",
                    "warn",
                    {"atomics": cost.atomics},
                )
            )
        if problem.op == "matmul":
            p = config.params
            if p["BLOCK_M"] * p["BLOCK_N"] < 128 * 64:
                obs.append(
                    Observation(
                        "small_tiles",
                        f"{p['BLOCK_M']}x{p['BLOCK_N']} output tile: low data reuse",
                        "info",
                        {"BLOCK_M": p["BLOCK_M"], "BLOCK_N": p["BLOCK_N"]},
                    )
                )
            if p.get("GROUP_M", 1) == 1 and problem.shape[1] >= 1024:
                obs.append(
                    Observation(
                        "row_major_order",
                        "row-major tile order re-streams B through L2",
                        "info",
                        {"GROUP_M": 1, "N": problem.shape[1]},
                    )
                )
            if config.num_stages <= 2:
                obs.append(
                    Observation(
                        "shallow_pipeline",
                        f"num_stages={config.num_stages}: little latency hiding",
                        "info",
                        {"num_stages": config.num_stages},
                    )
                )
        else:
            threads = 32 * config.num_warps
            block = next((v for k, v in config.params.items() if k.startswith("BLOCK")), None)
            if block:
                per_thread = block / threads
                if per_thread < 2:
                    obs.append(
                        Observation(
                            "thin_threads",
                            f"{per_thread:.1f} elements/thread: too many warps for the block",
                            "info",
                            {"elements_per_thread": per_thread},
                        )
                    )
                elif per_thread > 32 and not spills:
                    obs.append(
                        Observation(
                            "fat_threads",
                            f"{per_thread:.0f} elements/thread",
                            "info",
                            {"elements_per_thread": per_thread},
                        )
                    )

    if ev.compile is not None and ev.compile.ptx:
        ptx = summarize_ptx(ev.compile.ptx)
        if ptx.local_loads or ptx.local_stores:
            obs.append(
                Observation(
                    "ptx_local_memory",
                    "PTX has ld/st.local (spills)",
                    "warn",
                    {"ld_local": ptx.local_loads, "st_local": ptx.local_stores},
                )
            )
        if ptx.global_loads and ptx.vectorized_load_ratio < 0.5:
            obs.append(
                Observation(
                    "ptx_scalar_loads",
                    "most global loads are scalar",
                    "info",
                    {"vectorized_ratio": round(ptx.vectorized_load_ratio, 2)},
                )
            )
    return obs
