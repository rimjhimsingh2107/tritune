"""Deterministic demo scenarios. Always run on the simulated backend.

These go through the real orchestration: planner, tools, session, store,
comparison and reporting. The only scripted part is the *opening* move of
some scenarios, which forces the agent to try a specific bad idea first so
you can watch it detect and recover from the failure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kernelpilot.kernels import get_spec
from kernelpilot.problem import Problem


@dataclass
class Scenario:
    name: str
    title: str
    problem: str
    budget: int
    opening: list[dict[str, Any]] = field(default_factory=list)
    compare_with: str | None = None  # also run this autotuner strategy for comparison
    blurb: str = ""


def _bad_matmul_source() -> str:
    p = Problem.parse("matmul:1000x1000x1000:fp16")
    spec = get_spec("matmul")
    src = spec.render(spec.baseline(p))
    return src.replace("BLOCK_M = 64", "BLOCK_M = 96")


SCENARIOS: dict[str, Scenario] = {
    "success": Scenario(
        "success",
        "Successful optimization: register spills in a wide softmax",
        "softmax:1024x16384:fp32",
        budget=6,
        compare_with="grid",
        blurb="The untuned kernel holds a 16K-wide row in registers with 4 warps and spills badly. "
              "The agent reads the spill count from compiler metadata and goes after it.",
    ),
    "invalid-kernel": Scenario(
        "invalid-kernel",
        "Invalid kernel: an 'optimization' that breaks numerics",
        "softmax:2048x1024:fp16",
        budget=4,
        opening=[{
            "variant": "naive",
            "hypothesis": "Drop the max-subtraction pass and keep exp() in fp16 to save a reduction and registers",
            "rationale": "baseline spends a full tl.max reduction per row",
        }],
        blurb="The first idea looks reasonable on paper but overflows fp16. Validation catches NaNs "
              "before it is ever timed, and the agent repairs it.",
    ),
    "compile-failure": Scenario(
        "compile-failure",
        "Compilation failure: tiles that don't fit in shared memory",
        "matmul:2048x2048x2048:fp16",
        budget=6,
        opening=[{
            "params": {"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 64},
            "num_warps": 8,
            "num_stages": 4,
            "hypothesis": "Max out the output tile to maximize arithmetic intensity",
            "rationale": "baseline 64x64 tile is memory bound",
        }],
        blurb="A 256x256x64 fp16 tile with 4 stages needs 256 KB of shared memory. The compiler error is "
              "parsed into an observation and the agent shrinks the footprint step by step.",
    ),
    "bad-source": Scenario(
        "bad-source",
        "Hand-edited source with an illegal block size",
        "matmul:1000x1000x1000:fp16",
        budget=5,
        opening=[{
            "source": _bad_matmul_source(),
            "hypothesis": "BLOCK_M=96 wastes fewer rows on a 1000-row problem than 64 or 128",
            "rationale": "1000 rows / 64 leaves a mostly empty last tile row",
        }],
        blurb="A source-level edit picks a non-power-of-two block. Static checks flag tl.arange(0, 96); "
              "the agent rounds it to a legal size and keeps going.",
    ),
    "regression": Scenario(
        "regression",
        "Performance regression: a change that makes things slower",
        "reduction:16777216:fp32",
        budget=4,
        opening=[{
            "params": {"BLOCK_SIZE": 8192},
            "num_warps": 2,
            "num_stages": 1,
            "hypothesis": "Bigger blocks with fewer warps should cut launch and atomic overhead",
            "rationale": "baseline launches 16K programs",
        }],
        blurb="The opening move compiles and validates fine but measures much slower. It is rejected and "
              "the agent continues from the previous incumbent.",
    ),
}
