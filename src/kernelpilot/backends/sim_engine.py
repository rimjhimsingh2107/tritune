"""Simulated compile / validate / benchmark / profile.

What's real here:
  * parsing + static checks on the candidate source (analysis.static_check)
  * mapping the source back to a known kernel body and launch config
  * resource-limit checks (shared memory, registers) using the spec's estimates
  * numerical validation: the variant's arithmetic is emulated in numpy and
    compared against the fp64 reference, so NaNs/overflow/precision loss are
    actually computed, not scripted
What's not:
  * timings and profiler counters come from perf_model, plus seeded noise
"""

from __future__ import annotations

import ast
from functools import lru_cache
from typing import Any

from kernelpilot.analysis.static_check import kernel_fingerprint, lint
from kernelpilot.backends import perf_model
from kernelpilot.correctness import Check, ValidationResult, compare
from kernelpilot.kernels import REGISTRY, get_spec
from kernelpilot.problem import KernelConfig, Problem

DEVICE = perf_model.SimDevice()
SKIP_PARAMS = {"NUM_WARPS", "NUM_STAGES", "FINAL_BLOCK"}


@lru_cache(maxsize=1)
def _fingerprints() -> dict[str, list[tuple[str, str]]]:
    table: dict[str, list[tuple[str, str]]] = {}
    for op, spec in REGISTRY.items():
        problem = Problem(op, spec.default_shapes[0])
        for variant in spec.variants:
            src = spec.render(spec.baseline(problem).replace(variant=variant))
            fp = kernel_fingerprint(ast.parse(src))
            if fp:
                table.setdefault(fp, []).append((op, variant))
    return table


def identify(source: str, problem: Problem) -> KernelConfig | None:
    """Recover (variant, params) from candidate source, or None if unknown."""
    report = lint(source)
    if report.fingerprint is None:
        return None
    matches = [v for op, v in _fingerprints().get(report.fingerprint, []) if op == problem.op]
    if not matches:
        return None
    consts = report.constants
    params = {k: v for k, v in consts.items() if k.isupper() and k not in SKIP_PARAMS}
    variant = matches[0]
    if problem.op == "matmul" and set(matches) >= {"tiled", "grouped"}:
        variant = "grouped" if params.get("GROUP_M", 1) > 1 else "tiled"
    return KernelConfig(
        variant=variant,
        params=params,
        num_warps=consts.get("NUM_WARPS", 4),
        num_stages=consts.get("NUM_STAGES", 2),
    )


def validation_problem(problem: Problem) -> Problem:
    """Shrink huge shapes for CPU emulation; keep the ragged remainders."""
    if problem.op == "matmul":
        dims = tuple(d if d <= 1024 else 1024 + d % 64 for d in problem.shape)
        return Problem(problem.op, dims, problem.dtype)
    if problem.op == "softmax":
        m, n = problem.shape
        return Problem(problem.op, (min(m, 2048), n), problem.dtype)
    return problem


def _compile(source: str, problem: Problem) -> dict[str, Any]:
    report = lint(source)
    warnings = [w.render() for w in report.warnings]
    if not report.ok:
        kind = "syntax" if report.syntax_error else "lint"
        return {
            "ok": False,
            "error": report.error_text(),
            "error_kind": kind,
            "log": "\n".join(warnings),
            "simulated": True,
        }

    spec = get_spec(problem.op)
    cfg = identify(source, problem)
    if cfg is None:
        return {
            "ok": True,
            "simulated": True,
            "log": "\n".join(
                [
                    *warnings,
                    "kernel body not recognised by the simulator; "
                    "resource/timing model unavailable",
                ]
            ),
            "metadata": {"model": perf_model.MODEL_VERSION, "recognised": False},
            "compile_ms": None,
        }

    problems = spec.check(cfg, problem)
    if problems:
        return {
            "ok": False,
            "error_kind": "launch",
            "simulated": True,
            "error": "launch constraint violated: " + "; ".join(problems),
            "log": "\n".join(warnings),
        }

    res = spec.resources(cfg, problem)
    if res.smem_bytes > DEVICE.smem_per_block:
        return {
            "ok": False,
            "error_kind": "resources",
            "simulated": True,
            "error": (
                f"triton.runtime.errors.OutOfResources: out of resource: shared memory, "
                f"Required: {res.smem_bytes}, Hardware limit: {DEVICE.smem_per_block}. "
                "Reducing block sizes or `num_stages` may help."
            ),
            "log": "\n".join(warnings),
        }

    blocks, occ, limiter = perf_model.occupancy(res, DEVICE)
    spilled = max(0, res.regs_per_thread - DEVICE.max_regs_per_thread)
    if spilled:
        warnings.append(f"ptxas warning: {spilled} registers spilled to local memory (estimated)")
    key = source + problem.key
    return {
        "ok": True,
        "simulated": True,
        "compile_ms": perf_model.compile_ms(spec, cfg, len(report.kernels), key),
        "log": "\n".join(warnings),
        "metadata": {
            "model": perf_model.MODEL_VERSION,
            "recognised": True,
            "variant": cfg.variant,
            "config": cfg.to_dict(),
            "kernels": report.kernels,
            "n_regs": min(res.regs_per_thread, DEVICE.max_regs_per_thread),
            "n_spills": spilled,
            "shared": res.smem_bytes,
            "num_warps": cfg.num_warps,
            "num_stages": cfg.num_stages,
            "threads": res.threads,
            "blocks_per_sm": blocks,
            "occupancy_limiter": limiter,
            "est_occupancy": round(occ, 4),
        },
    }


@lru_cache(maxsize=256)
def _reference(op: str, pkey: str, seed: int):
    spec = get_spec(op)
    p = Problem.parse(pkey)
    inputs = spec.make_inputs(p, seed)
    return inputs, spec.reference(inputs, p)


_CHECKS: dict[tuple[str, str, str, int], Check] = {}


def _emulated_check(cfg: KernelConfig, p: Problem, seed: int) -> Check:
    # every config with the same numerics key computes the same thing in
    # emulation, so one emulation per (numerics, shape, seed) is enough
    spec = get_spec(p.op)
    key = (p.op, spec.numerics_key(cfg, p), p.key, seed)
    if key not in _CHECKS:
        if len(_CHECKS) > 512:
            _CHECKS.clear()
        inputs, ref = _reference(p.op, p.key, seed)
        _CHECKS[key] = compare(spec.emulate(cfg, inputs, p), ref, spec.tolerance(p), p.key)
    return _CHECKS[key]


def _validate(source: str, problem: Problem, seed: int) -> dict[str, Any]:
    comp = _compile(source, problem)
    if not comp["ok"]:
        return ValidationResult(
            False, status="error", error=f"does not compile: {comp['error']}"
        ).to_dict()
    cfg = identify(source, problem)
    if cfg is None:
        return ValidationResult(
            False,
            status="unverified",
            error="simulated backend can't emulate an unrecognised kernel body",
        ).to_dict()

    spec = get_spec(problem.op)
    checks = []
    targets = [validation_problem(problem), *spec.edge_problems(problem)]
    for p in targets:
        issues = spec.check(cfg, p)
        if issues:
            checks.append(
                Check(
                    p.key,
                    False,
                    float("inf"),
                    float("inf"),
                    1.0,
                    0,
                    0,
                    0.0,
                    0.0,
                    message="launch fails: " + "; ".join(issues),
                )
            )
            continue
        checks.append(_emulated_check(cfg, p, seed))
    ok = all(c.ok for c in checks)
    return ValidationResult(ok, checks=checks, status="verified" if ok else "failed").to_dict()


def _benchmark(source: str, problem: Problem, warmup: int, reps: int) -> dict[str, Any]:
    comp = _compile(source, problem)
    if not comp["ok"]:
        return {"ok": False, "error": f"does not compile: {comp['error']}", "simulated": True}
    cfg = identify(source, problem)
    if cfg is None:
        return {
            "ok": False,
            "simulated": True,
            "error": "no timing model for an unrecognised kernel body",
        }
    spec = get_spec(problem.op)
    pred = perf_model.predict(spec, cfg, problem, DEVICE)
    samples = perf_model.noisy_samples(pred.time_s, source + problem.key, reps)
    return {
        "ok": True,
        "samples_ms": samples,
        "warmup": warmup,
        "simulated": True,
        "method": f"simulated ({perf_model.MODEL_VERSION}, seeded noise)",
    }


def _profile(source: str, problem: Problem) -> dict[str, Any]:
    comp = _compile(source, problem)
    if not comp["ok"]:
        return {"ok": False, "error": f"does not compile: {comp['error']}", "simulated": True}
    cfg = identify(source, problem)
    if cfg is None:
        return {"ok": False, "simulated": True, "error": "no model for an unrecognised kernel body"}
    spec = get_spec(problem.op)
    pred = perf_model.predict(spec, cfg, problem, DEVICE)
    res = spec.resources(cfg, problem)
    metrics = pred.as_metrics(DEVICE, res)
    metrics["bound_memory"] = 1.0 if pred.bound == "memory" else 0.0
    units = {k: ("%" if k.endswith("_pct") else "us" if k.endswith("_us") else "") for k in metrics}
    units["shared_mem_per_block"] = "bytes"
    return {
        "ok": True,
        "simulated": True,
        "tool": "simulated-profiler",
        "metrics": metrics,
        "units": units,
        "raw": f"bound={pred.bound} limiter={pred.limiter} blocks/sm={pred.blocks_per_sm}",
    }


def handle(req: dict[str, Any]) -> dict[str, Any]:
    action = req["action"]
    problem = Problem.from_dict(req["problem"])
    source = req["source"]
    if action == "compile":
        return _compile(source, problem)
    if action == "validate":
        return _validate(source, problem, int(req.get("seed", 0)))
    if action == "benchmark":
        return _benchmark(source, problem, int(req.get("warmup", 10)), int(req.get("reps", 50)))
    if action == "profile":
        return _profile(source, problem)
    raise ValueError(f"unknown action {action!r}")
