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


class VectorAdd(KernelSpec):
    name = "vector_add"
    summary = "out = x + y, elementwise. Pure bandwidth; mostly a sanity check."
    inputs = ("x", "y")
    default_shapes = ((1 << 20,), (1 << 24,), ((1 << 20) + 37,))
    variants = {
        "masked": Variant("masked", "vector_add.py.tmpl", "bounds-masked loads/stores"),
        "unmasked": Variant(
            "unmasked",
            "vector_add_unmasked.py.tmpl",
            "drops the bounds mask; reads past the end when N % BLOCK != 0",
            searchable=False,
        ),
    }

    def make_inputs(self, problem: Problem, seed: int = 0) -> list[np.ndarray]:
        rng = rng_for(problem, seed)
        (n,) = problem.shape
        x = rng.standard_normal(n).astype(problem.np_dtype)
        y = rng.standard_normal(n).astype(problem.np_dtype)
        return [x, y]

    def reference(self, inputs: list[np.ndarray], problem: Problem) -> np.ndarray:
        x, y = inputs
        return x.astype(np.float64) + y.astype(np.float64)

    def emulate(
        self, config: KernelConfig, inputs: list[np.ndarray], problem: Problem
    ) -> np.ndarray:
        x, y = inputs
        out = (x + y).astype(problem.np_dtype)
        if config.variant == "unmasked":
            block = config.params["BLOCK_SIZE"]
            n = x.shape[0]
            tail = n % block
            if tail:
                # the last program reads past the allocation. on hardware that's
                # whatever lives next door; model it as garbage in the tail block.
                start = n - tail
                rng = np.random.default_rng(block)
                out[start:] = rng.standard_normal(tail).astype(problem.np_dtype) * 1e3
        return out

    def numerics_key(self, config: KernelConfig, problem: Problem) -> str:
        if config.variant == "unmasked":
            return f"unmasked/{config.params['BLOCK_SIZE']}"
        return config.variant

    def tolerance(self, problem: Problem) -> Tolerance:
        return Tolerance(atol=1e-3, rtol=1e-3) if problem.dtype == "fp16" else Tolerance(1e-6, 1e-6)

    def edge_problems(self, problem: Problem) -> list[Problem]:
        return [Problem(self.name, (1,), problem.dtype), Problem(self.name, (4099,), problem.dtype)]

    def baseline(self, problem: Problem) -> KernelConfig:
        return KernelConfig("masked", {"BLOCK_SIZE": 1024}, num_warps=4, num_stages=2)

    def space(self, problem: Problem) -> dict[str, dict[str, list[int]]]:
        return {
            "masked": {
                "BLOCK_SIZE": [256, 512, 1024, 2048, 4096, 8192],
                "num_warps": [2, 4, 8, 16],
                "num_stages": [1, 2],
            }
        }

    def check(self, config: KernelConfig, problem: Problem) -> list[str]:
        errs = self.common_checks(config)
        block = config.params.get("BLOCK_SIZE")
        if block is None:
            errs.append("missing BLOCK_SIZE")
        elif block < 32 * config.num_warps:
            errs.append(f"BLOCK_SIZE={block} smaller than one element per thread")
        elif block > 65536:
            errs.append("BLOCK_SIZE above 65536 exceeds Triton's max tensor numel per program")
        return errs

    def resources(self, config: KernelConfig, problem: Problem) -> Resources:
        threads = 32 * config.num_warps
        per_thread = config.params["BLOCK_SIZE"] / threads
        # x, y, out per element plus offsets/masks, ~16 baseline regs
        regs = int(16 + per_thread * (3 * problem.elem_bytes / 4 + 2))
        return Resources(threads, regs, 0)

    def cost(self, config: KernelConfig, problem: Problem) -> Cost:
        (n,) = problem.shape
        block = config.params["BLOCK_SIZE"]
        programs = -(-n // block)
        return Cost(
            flops=n,
            dram_bytes=3.0 * n * problem.elem_bytes,
            programs=programs,
            lane_efficiency=n / (programs * block),
        )

    def render(self, config: KernelConfig) -> str:
        return render_template(
            self.variant(config.variant).template,
            BLOCK_SIZE=config.params["BLOCK_SIZE"],
            num_warps=config.num_warps,
            num_stages=config.num_stages,
        )
