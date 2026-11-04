from __future__ import annotations

import numpy as np

from kernelpilot.kernels.base import (
    Cost,
    KernelSpec,
    Resources,
    Tolerance,
    Variant,
    render_template,
    rng_for,
)
from kernelpilot.problem import KernelConfig, Problem, next_pow2


class Softmax(KernelSpec):
    name = "softmax"
    summary = "Row-wise softmax over the last dim of an (M, N) matrix."
    inputs = ("x",)
    dtypes = ("fp16", "fp32")
    default_shapes = ((4096, 1024), (8192, 512), (2048, 4096), (1823, 781))
    variants = {
        "single_pass": Variant(
            "single_pass",
            "softmax_single_pass.py.tmpl",
            "whole row in registers, one read + one write. needs BLOCK_N >= N",
        ),
        "online": Variant(
            "online",
            "softmax_online.py.tmpl",
            "chunked online softmax, any N, two reads + one write",
        ),
        "naive": Variant(
            "naive",
            "softmax_naive.py.tmpl",
            "no max subtraction, fp16 intermediates. overflows on large logits",
            searchable=False,
        ),
    }

    def make_inputs(self, problem: Problem, seed: int = 0) -> list[np.ndarray]:
        rng = rng_for(problem, seed)
        # logit scale ~6 is deliberately "real-model-ish": big enough that skipping
        # the max subtraction overflows fp16 exp(), small enough for fp32.
        x = rng.standard_normal(problem.shape) * 6.0
        return [x.astype(problem.np_dtype)]

    def reference(self, inputs: list[np.ndarray], problem: Problem) -> np.ndarray:
        x = inputs[0].astype(np.float64)
        x = x - x.max(axis=1, keepdims=True)
        e = np.exp(x)
        return e / e.sum(axis=1, keepdims=True)

    def emulate(
        self, config: KernelConfig, inputs: list[np.ndarray], problem: Problem
    ) -> np.ndarray:
        x = inputs[0]
        if config.variant == "naive":
            with np.errstate(over="ignore", invalid="ignore"):
                num = np.exp(x.astype(np.float32)).astype(x.dtype)
                den = num.astype(np.float32).sum(axis=1, keepdims=True)
                y = num.astype(np.float32) / den
            return y.astype(problem.np_dtype)
        x32 = x.astype(np.float32)
        e = np.exp(x32 - x32.max(axis=1, keepdims=True))
        return (e / e.sum(axis=1, keepdims=True)).astype(problem.np_dtype)

    def numerics_key(self, config: KernelConfig, problem: Problem) -> str:
        return "naive" if config.variant == "naive" else "stable"

    def tolerance(self, problem: Problem) -> Tolerance:
        if problem.dtype == "fp16":
            return Tolerance(atol=1e-3, rtol=1e-2)
        return Tolerance(atol=1e-6, rtol=1e-5)

    def edge_problems(self, problem: Problem) -> list[Problem]:
        _, n = problem.shape
        return [
            Problem(self.name, (5, max(1, n - 3)), problem.dtype),
            Problem(self.name, (1, n), problem.dtype),
        ]

    def baseline(self, problem: Problem) -> KernelConfig:
        _, n = problem.shape
        return KernelConfig("single_pass", {"BLOCK_N": next_pow2(n)}, num_warps=4, num_stages=2)

    def space(self, problem: Problem) -> dict[str, dict[str, list[int]]]:
        _, n = problem.shape
        full = next_pow2(n)
        chunks = [b for b in (128, 256, 512, 1024, 2048, 4096) if b <= full]
        return {
            "single_pass": {
                "BLOCK_N": [full],
                "num_warps": [1, 2, 4, 8, 16, 32],
                "num_stages": [1, 2],
            },
            "online": {
                "BLOCK_N": chunks or [full],
                "num_warps": [1, 2, 4, 8, 16],
                "num_stages": [1, 2, 3],
            },
        }

    def check(self, config: KernelConfig, problem: Problem) -> list[str]:
        errs = self.common_checks(config)
        block = config.params.get("BLOCK_N")
        _, n = problem.shape
        if block is None:
            return [*errs, "missing BLOCK_N"]
        if config.variant in ("single_pass", "naive") and block < n:
            errs.append(f"{config.variant} needs BLOCK_N >= n_cols ({block} < {n})")
        if block > 65536:
            errs.append("BLOCK_N above 65536 exceeds Triton's max tensor numel")
        return errs

    def resources(self, config: KernelConfig, problem: Problem) -> Resources:
        threads = 32 * config.num_warps
        per_thread = max(1.0, config.params["BLOCK_N"] / threads)
        if config.variant == "online":
            regs = 28 + per_thread * 4  # m, s, x, scratch
        else:
            regs = 20 + per_thread * 3  # x, num, y
        smem = 8 * config.num_warps  # cross-warp reduction scratch
        return Resources(threads, int(regs), smem)

    def cost(self, config: KernelConfig, problem: Problem) -> Cost:
        m, n = problem.shape
        block = config.params["BLOCK_N"]
        e = problem.elem_bytes
        if config.variant == "online":
            chunks = -(-n // block)
            return Cost(
                flops=10.0 * m * n,
                dram_bytes=3.0 * m * n * e,
                programs=m,
                lane_efficiency=n / (chunks * block),
            )
        return Cost(
            flops=6.0 * m * n,
            dram_bytes=2.0 * m * n * e,
            programs=m,
            lane_efficiency=n / block,
        )

    def render(self, config: KernelConfig) -> str:
        return render_template(
            self.variant(config.variant).template,
            BLOCK_N=config.params["BLOCK_N"],
            num_warps=config.num_warps,
            num_stages=config.num_stages,
        )
