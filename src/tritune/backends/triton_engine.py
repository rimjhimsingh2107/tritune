"""Real compile / validate / benchmark / profile on a CUDA device.

Always runs inside the worker process (see worker.py), never in the main
process: a bad candidate can hang the GPU, raise an illegal memory access
that poisons the CUDA context, or segfault the interpreter.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from tritune.correctness import Check, ValidationResult, compare
from tritune.kernels import get_spec
from tritune.problem import Problem

TORCH_DTYPES = {"fp16": "float16", "fp32": "float32"}


def _torch():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device not available in worker")
    return torch


def load_candidate(source: str, workdir: Path):
    path = workdir / "candidate.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("tt_candidate", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "launch"):
        raise AttributeError("candidate module has no launch()")
    return module


def to_device(arrays: list[np.ndarray]):
    torch = _torch()
    return [torch.from_numpy(np.ascontiguousarray(a)).cuda() for a in arrays]


def torch_reference(problem: Problem, tensors) -> Any:
    """PyTorch reference computed in float64 on the device."""
    torch = _torch()
    t = [x.to(torch.float64) for x in tensors]
    if problem.op == "vector_add":
        return t[0] + t[1]
    if problem.op == "softmax":
        return torch.softmax(t[0], dim=-1)
    if problem.op == "reduction":
        return t[0].sum().reshape(1)
    if problem.op == "matmul":
        return t[0] @ t[1]
    raise ValueError(problem.op)


def _metadata(handles: list[Any]) -> dict[str, Any]:
    """Pull what we can out of CompiledKernel objects. Field names moved
    around between Triton 2.x and 3.x, so probe instead of assuming."""
    kernels = []
    for h in handles:
        if h is None:
            continue
        md = getattr(h, "metadata", None)

        def field(name: str, md=md, h=h):
            if hasattr(h, name):
                return getattr(h, name)
            if isinstance(md, dict):
                return md.get(name)
            return getattr(md, name, None)

        kernels.append(
            {
                "name": field("name"),
                "n_regs": field("n_regs"),
                "n_spills": field("n_spills"),
                "shared": field("shared"),
                "num_warps": field("num_warps"),
                "num_stages": field("num_stages"),
            }
        )
    if not kernels:
        return {}
    main = kernels[0]
    return {**main, "kernels": kernels}


def _ptx(handles: list[Any]) -> str | None:
    for h in handles:
        asm = getattr(h, "asm", None)
        if isinstance(asm, dict) and asm.get("ptx"):
            ptx = asm["ptx"]
            return ptx if isinstance(ptx, str) else ptx.decode()
    return None


def _compile(source: str, problem: Problem, seed: int) -> dict[str, Any]:
    torch = _torch()
    spec = get_spec(problem.op)
    with tempfile.TemporaryDirectory(prefix="tt-") as tmp:
        work = Path(tmp)
        # fresh triton cache dir so compile time is a real compile every time;
        # our own content-hash cache sits above this
        os.environ["TRITON_CACHE_DIR"] = str(work / "triton-cache")
        try:
            module = load_candidate(source, work)
        except SyntaxError as exc:
            return {"ok": False, "error_kind": "syntax", "error": f"SyntaxError: {exc}"}
        except Exception as exc:
            return {
                "ok": False,
                "error_kind": "triton",
                "error": f"{type(exc).__name__}: {exc}",
                "log": traceback.format_exc(limit=4),
            }

        inputs = to_device(spec.make_inputs(problem, seed))
        try:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            module.launch(*inputs)
            torch.cuda.synchronize()
            first = time.perf_counter() - t0
            t0 = time.perf_counter()
            module.launch(*inputs)
            torch.cuda.synchronize()
            second = time.perf_counter() - t0
        except Exception as exc:
            name = type(exc).__name__
            kind = (
                "resources"
                if "OutOfResources" in name or "out of resource" in str(exc)
                else "triton"
            )
            return {
                "ok": False,
                "error_kind": kind,
                "error": f"{name}: {exc}",
                "log": traceback.format_exc(limit=6),
            }

        handles = list(getattr(module, "_compiled", []) or [])
        return {
            "ok": True,
            "compile_ms": max(0.0, (first - second) * 1e3),
            "metadata": _metadata(handles),
            "ptx": _ptx(handles),
        }


def _validate(source: str, problem: Problem, seed: int) -> dict[str, Any]:
    torch = _torch()
    spec = get_spec(problem.op)
    checks: list[Check] = []
    with tempfile.TemporaryDirectory(prefix="tt-") as tmp:
        try:
            module = load_candidate(source, Path(tmp))
        except Exception as exc:
            return ValidationResult(
                False, status="error", error=f"{type(exc).__name__}: {exc}"
            ).to_dict()
        for p in [problem, *spec.edge_problems(problem)]:
            tensors = to_device(spec.make_inputs(p, seed))
            try:
                out = module.launch(*tensors)
                torch.cuda.synchronize()
            except Exception as exc:
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
                        message=f"launch raised {type(exc).__name__}: {exc}",
                    )
                )
                continue
            ref = torch_reference(p, tensors)
            checks.append(
                compare(out.float().cpu().numpy(), ref.cpu().numpy(), spec.tolerance(p), p.key)
            )
    ok = all(c.ok for c in checks)
    return ValidationResult(ok, checks=checks, status="verified" if ok else "failed").to_dict()


def _benchmark(source: str, problem: Problem, seed: int, warmup: int, reps: int) -> dict[str, Any]:
    """CUDA-event timing with an L2 flush before every rep (same idea as
    triton.testing.do_bench), so we time cold-cache kernels consistently."""
    torch = _torch()
    spec = get_spec(problem.op)
    with tempfile.TemporaryDirectory(prefix="tt-") as tmp:
        module = load_candidate(source, Path(tmp))
        inputs = to_device(spec.make_inputs(problem, seed))
        flush = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.int32, device="cuda")
        for _ in range(warmup):
            module.launch(*inputs)
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
        for i in range(reps):
            flush.zero_()
            starts[i].record()
            module.launch(*inputs)
            ends[i].record()
        torch.cuda.synchronize()
        samples = [s.elapsed_time(e) for s, e in zip(starts, ends, strict=True)]
    return {
        "ok": True,
        "samples_ms": samples,
        "warmup": warmup,
        "method": "cuda events, L2 flushed between reps",
    }


NCU_METRICS = {
    "gpu__time_duration.sum": "duration_ns",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed": "dram_throughput_pct",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "sm_throughput_pct",
    "sm__warps_active.avg.pct_of_peak_sustained_active": "achieved_occupancy_pct",
    "launch__registers_per_thread": "registers_per_thread",
    "launch__shared_mem_per_block_static": "static_smem_per_block",
    "launch__shared_mem_per_block_dynamic": "shared_mem_per_block",
    "launch__occupancy_limit_registers": "occupancy_limit_registers",
    "launch__occupancy_limit_shared_mem": "occupancy_limit_shared_mem",
}


def _profile(source: str, problem: Problem, seed: int, ncu: str | None) -> dict[str, Any]:
    from tritune.analysis.ncu import parse_ncu_csv

    if not ncu:
        return {
            "ok": False,
            "error": "Nsight Compute (ncu) not found; set TRITUNE_NCU or add it to PATH",
        }
    with tempfile.TemporaryDirectory(prefix="tt-prof-") as tmp:
        src = Path(tmp) / "candidate_src.py"
        src.write_text(source)
        cmd = [
            ncu,
            "--csv",
            "--page",
            "raw",
            "--launch-skip",
            "3",
            "--launch-count",
            "1",
            "--metrics",
            ",".join(NCU_METRICS),
            sys.executable,
            "-m",
            "tritune.backends.worker",
            "--profile-target",
            str(src),
            problem.key,
            str(seed),
        ]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=600, check=False)
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "ncu timed out"}
        if out.returncode != 0:
            tail = (out.stderr or out.stdout)[-2000:]
            hint = (
                " (ERR_NVGPUCTRPERM: needs profiling permissions)"
                if "ERR_NVGPUCTRPERM" in tail
                else ""
            )
            return {"ok": False, "error": f"ncu exited {out.returncode}{hint}", "raw": tail}
        metrics, units = parse_ncu_csv(out.stdout, NCU_METRICS)
        if not metrics:
            return {
                "ok": False,
                "error": "ncu produced no metrics for the kernel",
                "raw": out.stdout[-2000:],
            }
        if "duration_ns" in metrics:
            metrics["duration_us"] = metrics.pop("duration_ns") / 1e3
            units["duration_us"] = "us"
        return {
            "ok": True,
            "tool": "ncu",
            "metrics": metrics,
            "units": units,
            "raw": out.stdout[-4000:],
        }


def profile_target(src_path: str, problem_key: str, seed: int) -> None:
    """Entry point ncu wraps: a few warm launches then the one that gets profiled."""
    torch = _torch()
    problem = Problem.parse(problem_key)
    spec = get_spec(problem.op)
    module = load_candidate(Path(src_path).read_text(), Path(tempfile.mkdtemp(prefix="tt-")))
    inputs = to_device(spec.make_inputs(problem, seed))
    for _ in range(4):
        module.launch(*inputs)
    torch.cuda.synchronize()


def handle(req: dict[str, Any]) -> dict[str, Any]:
    action = req["action"]
    problem = Problem.from_dict(req["problem"])
    source = req["source"]
    seed = int(req.get("seed", 0))
    if action == "compile":
        return _compile(source, problem, seed)
    if action == "validate":
        return _validate(source, problem, seed)
    if action == "benchmark":
        return _benchmark(
            source, problem, seed, int(req.get("warmup", 10)), int(req.get("reps", 50))
        )
    if action == "profile":
        return _profile(source, problem, seed, req.get("ncu"))
    raise ValueError(f"unknown action {action!r}")
