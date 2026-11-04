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
from kernelpilot.problem import KernelConfig, Problem


class Reduction(KernelSpec):
    name = "reduction"
    summary = "Full sum of a 1-D tensor into an fp32 scalar."
    inputs = ("x",)
    default_shapes = ((1 << 20,), (1 << 24,), ((1 << 22) + 7,))
    variants = {
        "atomic": Variant(
            "atomic",
            "reduction_atomic.py.tmpl",
            "per-program register accumulation, one fp32 atomic per program",
        ),
        "two_pass": Variant(
            "two_pass",
            "reduction_two_pass.py.tmpl",
            "partials to global memory, then a single-program final pass (deterministic)",
        ),
        "atomic_fp16acc": Variant(
            "atomic_fp16acc",
            "reduction_atomic.py.tmpl",
            "atomic variant but the accumulator stays in the input dtype",
            searchable=False,
        ),
    }

    def make_inputs(self, problem: Problem, seed: int = 0) -> list[np.ndarray]:
        rng = rng_for(problem, seed)
        # positive values so relative error of the sum is well conditioned
        return [rng.uniform(0.0, 1.0, problem.shape).astype(problem.np_dtype)]

    def reference(self, inputs: list[np.ndarray], problem: Problem) -> np.ndarray:
        return np.array([inputs[0].astype(np.float64).sum()])

    def emulate(
        self, config: KernelConfig, inputs: list[np.ndarray], problem: Problem
    ) -> np.ndarray:
        x = inputs[0]
        if config.variant != "atomic_fp16acc" or problem.dtype == "fp32":
            return np.array([x.astype(np.float32).sum(dtype=np.float32)], dtype=np.float32)

        block = config.params["BLOCK_SIZE"]
        tiles = config.params.get("TILES", 1)
        span = block * tiles
        programs = -(-x.size // span)
        padded = np.zeros(programs * span, dtype=x.dtype)
        padded[: x.size] = x
        lanes = padded.reshape(programs, tiles, block)
        acc = np.zeros((programs, block), dtype=np.float16)
        for t in range(tiles):
            acc = (acc + lanes[:, t, :]).astype(np.float16)
        partial = acc.sum(axis=1, dtype=np.float16)
        return np.array([partial.astype(np.float32).sum(dtype=np.float32)], dtype=np.float32)

    def numerics_key(self, config: KernelConfig, problem: Problem) -> str:
        if config.variant == "atomic_fp16acc" and problem.dtype == "fp16":
            return f"fp16acc/{config.params['BLOCK_SIZE']}/{config.params.get('TILES', 1)}"
        return "fp32acc"

    def tolerance(self, problem: Problem) -> Tolerance:
        # output is fp32 and accumulation must be fp32 regardless of input
        # dtype, so the tolerance is tight for both.
        return Tolerance(atol=1e-3, rtol=2e-5)

    def edge_problems(self, problem: Problem) -> list[Problem]:
        return [
            Problem(self.name, (1,), problem.dtype),
            Problem(self.name, (10007,), problem.dtype),
        ]

    def baseline(self, problem: Problem) -> KernelConfig:
        return KernelConfig("atomic", {"BLOCK_SIZE": 1024, "TILES": 1}, num_warps=4, num_stages=2)

    def space(self, problem: Problem) -> dict[str, dict[str, list[int]]]:
        blocks = [512, 1024, 2048, 4096, 8192]
        return {
            "atomic": {
                "BLOCK_SIZE": blocks,
                "TILES": [1, 2, 4, 8, 16],
                "num_warps": [2, 4, 8, 16],
                "num_stages": [1, 2],
            },
            "two_pass": {
                "BLOCK_SIZE": blocks,
                "num_warps": [2, 4, 8, 16],
                "num_stages": [1, 2],
            },
        }

    def check(self, config: KernelConfig, problem: Problem) -> list[str]:
        errs = self.common_checks(config)
        block = config.params.get("BLOCK_SIZE")
        if block is None:
            return [*errs, "missing BLOCK_SIZE"]
        if block < 32 * config.num_warps:
            errs.append(f"BLOCK_SIZE={block} smaller than one element per thread")
        if config.variant != "two_pass":
            tiles = config.params.get("TILES")
            if tiles is None or tiles < 1 or tiles > 64:
                errs.append("TILES must be in [1, 64] for atomic variants")
        return errs

    def resources(self, config: KernelConfig, problem: Problem) -> Resources:
        threads = 32 * config.num_warps
        per_thread = config.params["BLOCK_SIZE"] / threads
        regs = 18 + per_thread * 2
        return Resources(threads, int(regs), 8 * config.num_warps)

    def cost(self, config: KernelConfig, problem: Problem) -> Cost:
        (n,) = problem.shape
        block = config.params["BLOCK_SIZE"]
        if config.variant == "two_pass":
            parts = -(-n // block)
            return Cost(
                flops=n,
                dram_bytes=n * problem.elem_bytes + 8.0 * parts,
                programs=parts,
                lane_efficiency=n / (parts * block),
                launches=2,
            )
        span = block * config.params.get("TILES", 1)
        programs = -(-n // span)
        return Cost(
            flops=n,
            dram_bytes=float(n * problem.elem_bytes),
            programs=programs,
            lane_efficiency=n / (programs * span),
            atomics=programs,
        )

    def render(self, config: KernelConfig) -> str:
        variant = self.variant(config.variant)
        values: dict[str, object] = {
            "BLOCK_SIZE": config.params["BLOCK_SIZE"],
            "num_warps": config.num_warps,
            "num_stages": config.num_stages,
        }
        if config.variant != "two_pass":
            values["TILES"] = config.params.get("TILES", 1)
            values["ACC_DTYPE"] = (
                "x_ptr.dtype.element_ty" if config.variant == "atomic_fp16acc" else "tl.float32"
            )
        return render_template(variant.template, **values)
