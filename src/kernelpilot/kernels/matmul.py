from __future__ import annotations

import math

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


def _round_tf32(x: np.ndarray) -> np.ndarray:
    """Round fp32 to tf32 (10 explicit mantissa bits), like tl.dot does by default."""
    bits = x.astype(np.float32).view(np.uint32)
    bits = (bits + np.uint32(0x1000)) & np.uint32(0xFFFFE000)
    return bits.view(np.float32)


class Matmul(KernelSpec):
    name = "matmul"
    summary = "C = A @ B with A (M, K), B (K, N). Tiled, tensor-core path via tl.dot."
    inputs = ("a", "b")
    default_shapes = (
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 1024),
        (1000, 1000, 1000),
    )
    dtypes = ("fp16", "fp32")
    variants = {
        "tiled": Variant("tiled", "matmul.py.tmpl", "one program per output tile, row-major order"),
        "grouped": Variant(
            "grouped", "matmul.py.tmpl", "tiles visited in GROUP_M bands for L2 reuse of B"
        ),
        "fp16acc": Variant(
            "fp16acc",
            "matmul.py.tmpl",
            "fp16 accumulator: faster MMA on some parts, loses precision as K grows",
            searchable=False,
        ),
    }

    # problems are (M, N, K)
    def make_inputs(self, problem: Problem, seed: int = 0) -> list[np.ndarray]:
        m, n, k = problem.shape
        rng = rng_for(problem, seed)
        a = rng.standard_normal((m, k)).astype(problem.np_dtype)
        b = rng.standard_normal((k, n)).astype(problem.np_dtype)
        return [a, b]

    def reference(self, inputs: list[np.ndarray], problem: Problem) -> np.ndarray:
        a, b = inputs
        return a.astype(np.float64) @ b.astype(np.float64)

    def emulate(
        self, config: KernelConfig, inputs: list[np.ndarray], problem: Problem
    ) -> np.ndarray:
        a, b = inputs
        if config.variant == "fp16acc":
            # mma.sync with f16 accumulators: model as k=16 chunks rounded into
            # an fp16 running sum
            k = a.shape[1]
            acc = np.zeros((a.shape[0], b.shape[1]), dtype=np.float16)
            a32, b32 = a.astype(np.float32), b.astype(np.float32)
            for k0 in range(0, k, 16):
                part = a32[:, k0 : k0 + 16] @ b32[k0 : k0 + 16, :]
                acc = (acc.astype(np.float32) + part.astype(np.float16)).astype(np.float16)
            return acc.astype(problem.np_dtype)
        if problem.dtype == "fp32":
            return (_round_tf32(a) @ _round_tf32(b)).astype(np.float32)
        return (a.astype(np.float32) @ b.astype(np.float32)).astype(problem.np_dtype)

    def numerics_key(self, config: KernelConfig, problem: Problem) -> str:
        return "fp16acc" if config.variant == "fp16acc" else "fp32acc"

    def tolerance(self, problem: Problem) -> Tolerance:
        k = problem.shape[2]
        if problem.dtype == "fp16":
            return Tolerance(atol=1e-3 * math.sqrt(k), rtol=5e-3)
        # tf32 inputs: ~2^-11 relative error per product
        return Tolerance(atol=2e-3 * math.sqrt(k), rtol=1e-3)

    def edge_problems(self, problem: Problem) -> list[Problem]:
        return [
            Problem(self.name, (17, 33, 47), problem.dtype),
            Problem(self.name, (130, 70, 300), problem.dtype),
        ]

    def baseline(self, problem: Problem) -> KernelConfig:
        return KernelConfig(
            "tiled",
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 1},
            num_warps=4,
            num_stages=2,
        )

    def space(self, problem: Problem) -> dict[str, dict[str, list[int]]]:
        tiles = [32, 64, 128, 256]
        common = {
            "BLOCK_M": tiles,
            "BLOCK_N": tiles,
            "BLOCK_K": [32, 64],
            "num_warps": [4, 8],
            "num_stages": [2, 3, 4],
        }
        return {
            "tiled": {**common, "GROUP_M": [1]},
            "grouped": {**common, "GROUP_M": [4, 8]},
        }

    def check(self, config: KernelConfig, problem: Problem) -> list[str]:
        errs = self.common_checks(config)
        need = ("BLOCK_M", "BLOCK_N", "BLOCK_K", "GROUP_M")
        missing = [p for p in need if p not in config.params]
        if missing:
            return [*errs, f"missing params: {', '.join(missing)}"]
        p = config.params
        if min(p["BLOCK_M"], p["BLOCK_N"], p["BLOCK_K"]) < 16:
            errs.append("tl.dot needs every block dim >= 16")
        if config.variant == "tiled" and p["GROUP_M"] != 1:
            errs.append("tiled variant uses GROUP_M=1 by definition")
        if config.variant == "grouped" and p["GROUP_M"] < 2:
            errs.append("grouped variant needs GROUP_M >= 2")
        return errs

    def resources(self, config: KernelConfig, problem: Problem) -> Resources:
        p = config.params
        e = problem.elem_bytes
        threads = 32 * config.num_warps
        tile_a = p["BLOCK_M"] * p["BLOCK_K"]
        tile_b = p["BLOCK_K"] * p["BLOCK_N"]
        smem = (tile_a + tile_b) * e * config.num_stages
        acc_regs = p["BLOCK_M"] * p["BLOCK_N"] / threads  # fp32 accumulator
        if config.variant == "fp16acc":
            acc_regs /= 2
        operand_regs = (tile_a + tile_b) * e / 4 / threads
        return Resources(threads, int(40 + acc_regs + operand_regs), smem)

    def cost(self, config: KernelConfig, problem: Problem) -> Cost:
        m, n, k = problem.shape
        p = config.params
        e = problem.elem_bytes
        tm = -(-m // p["BLOCK_M"])
        tn = -(-n // p["BLOCK_N"])
        # traffic with no inter-program reuse: every tile row re-reads A, every
        # tile column re-reads B. the perf model credits L2 for part of this.
        traffic = (m * k * tn + k * n * tm) * e + m * n * e
        return Cost(
            flops=2.0 * m * n * k,
            dram_bytes=float(traffic),
            programs=tm * tn,
            lane_efficiency=(m * n) / (tm * p["BLOCK_M"] * tn * p["BLOCK_N"]),
        )

    def render(self, config: KernelConfig) -> str:
        p = config.params
        return render_template(
            self.variant(config.variant).template,
            BLOCK_M=p["BLOCK_M"],
            BLOCK_N=p["BLOCK_N"],
            BLOCK_K=p["BLOCK_K"],
            GROUP_M=p["GROUP_M"],
            ACC_DTYPE="tl.float16" if config.variant == "fp16acc" else "tl.float32",
            num_warps=config.num_warps,
            num_stages=config.num_stages,
        )
