from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass
from importlib import resources
from string import Template

import numpy as np

from kernelpilot.problem import KernelConfig, Problem, is_pow2


@dataclass(frozen=True)
class Variant:
    name: str
    template: str
    summary: str
    # Buggy variants exist so the agent/demo can hit real numerical failures.
    # They never show up in autotuner search spaces.
    searchable: bool = True


@dataclass(frozen=True)
class Resources:
    """Static estimate of per-program resources.

    This is a back-of-envelope model, not what ptxas will actually allocate.
    The real backend replaces it with compiler metadata whenever it can.
    """

    threads: int
    regs_per_thread: int
    smem_bytes: int


@dataclass(frozen=True)
class Cost:
    flops: float
    dram_bytes: float
    programs: int
    # fraction of lanes doing useful work (padding/masking waste)
    lane_efficiency: float = 1.0
    atomics: int = 0
    launches: int = 1


@dataclass(frozen=True)
class Tolerance:
    atol: float
    rtol: float


def render_template(name: str, **values: object) -> str:
    text = resources.files("kernelpilot.kernels.templates").joinpath(name).read_text()
    return Template(text).substitute({k: str(v) for k, v in values.items()})


class KernelSpec(ABC):
    name: str
    summary: str
    variants: dict[str, Variant]
    # names of the input tensors, in the order launch() takes them
    inputs: tuple[str, ...]
    default_shapes: tuple[tuple[int, ...], ...]
    dtypes: tuple[str, ...] = ("fp32", "fp16")

    # -- inputs / references -------------------------------------------------

    @abstractmethod
    def make_inputs(self, problem: Problem, seed: int = 0) -> list[np.ndarray]: ...

    @abstractmethod
    def reference(self, inputs: list[np.ndarray], problem: Problem) -> np.ndarray:
        """float64 numpy reference. The ground truth for everything else."""

    @abstractmethod
    def emulate(
        self, config: KernelConfig, inputs: list[np.ndarray], problem: Problem
    ) -> np.ndarray:
        """Reproduce the *numerics* of a variant on the CPU.

        Used by the simulated backend. It models accumulation dtype, masking
        and the order of operations that matter for rounding; it does not care
        about tile sizes unless they change the arithmetic.
        """

    def numerics_key(self, config: KernelConfig, problem: Problem) -> str:
        """Configs with the same key produce bit-identical emulated output."""
        return config.variant

    @abstractmethod
    def tolerance(self, problem: Problem) -> Tolerance: ...

    def edge_problems(self, problem: Problem) -> list[Problem]:
        """Small awkward shapes that exercise masking/tails."""
        return []

    # -- configs ---------------------------------------------------------------

    @abstractmethod
    def baseline(self, problem: Problem) -> KernelConfig:
        """The 'untuned Triton' config: what you'd write from the tutorial."""

    @abstractmethod
    def space(self, problem: Problem) -> dict[str, dict[str, list[int]]]:
        """Per-variant parameter grid. num_warps/num_stages included."""

    @abstractmethod
    def check(self, config: KernelConfig, problem: Problem) -> list[str]:
        """Launch-level constraint violations (empty list means OK)."""

    def configs(self, problem: Problem) -> list[KernelConfig]:
        out: list[KernelConfig] = []
        for variant, grid in self.space(problem).items():
            names = sorted(grid)
            for combo in itertools.product(*(grid[n] for n in names)):
                values = dict(zip(names, combo, strict=True))
                warps = values.pop("num_warps", 4)
                stages = values.pop("num_stages", 2)
                cfg = KernelConfig(variant, values, warps, stages)
                if not self.check(cfg, problem):
                    out.append(cfg)
        return out

    # -- performance-model hooks ---------------------------------------------

    @abstractmethod
    def resources(self, config: KernelConfig, problem: Problem) -> Resources: ...

    @abstractmethod
    def cost(self, config: KernelConfig, problem: Problem) -> Cost: ...

    # -- source ----------------------------------------------------------------

    @abstractmethod
    def render(self, config: KernelConfig) -> str: ...

    def variant(self, name: str) -> Variant:
        try:
            return self.variants[name]
        except KeyError:
            raise ValueError(
                f"{self.name} has no variant {name!r}; options: {sorted(self.variants)}"
            ) from None

    def common_checks(self, config: KernelConfig) -> list[str]:
        errs = []
        if config.variant not in self.variants:
            errs.append(f"unknown variant {config.variant!r}")
        if not is_pow2(config.num_warps) or config.num_warps > 32:
            errs.append(f"num_warps={config.num_warps} must be a power of two <= 32")
        if config.num_stages < 1 or config.num_stages > 8:
            errs.append(f"num_stages={config.num_stages} out of range [1, 8]")
        for k, v in config.params.items():
            if k.startswith("BLOCK") and not is_pow2(v):
                errs.append(f"{k}={v} is not a power of two (tl.arange needs pow2 extents)")
        return errs


def rng_for(problem: Problem, seed: int) -> np.random.Generator:
    # stable across runs and python versions, unlike hash()
    mix = sum((i + 1) * d for i, d in enumerate(problem.shape)) + len(problem.op) * 7919
    return np.random.default_rng([seed, mix, problem.elem_bytes])
