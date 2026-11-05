"""Named benchmark suites.

'standard' is the one the README numbers refer to: 13 shapes across matmul,
softmax and reduction, mixing power-of-two and ragged sizes, both dtypes.
"""

from __future__ import annotations

from tritune.kernels import get_spec
from tritune.problem import Problem

SUITES: dict[str, list[str]] = {
    "quick": [
        "matmul:512x512x512:fp16",
        "softmax:1024x1024:fp16",
        "reduction:1048576:fp32",
        "vector_add:1048576:fp32",
    ],
    "standard": [
        "matmul:1024x1024x1024:fp16",
        "matmul:2048x2048x2048:fp16",
        "matmul:4096x4096x1024:fp16",
        "matmul:1000x1000x1000:fp16",
        "matmul:2048x512x2048:fp32",
        "softmax:4096x1024:fp16",
        "softmax:8192x512:fp16",
        "softmax:2048x4096:fp32",
        "softmax:1823x781:fp16",
        "softmax:1024x16384:fp32",
        "reduction:1048576:fp32",
        "reduction:16777216:fp32",
        "reduction:4194311:fp16",
    ],
}
SUITES["full"] = SUITES["standard"] + ["vector_add:16777216:fp32", "vector_add:1048613:fp16"]


def suite_problems(name: str) -> list[Problem]:
    try:
        keys = SUITES[name]
    except KeyError:
        raise KeyError(f"unknown suite {name!r}; known: {', '.join(SUITES)}") from None
    return [Problem.parse(k) for k in keys]


def default_problems(op: str) -> list[Problem]:
    spec = get_spec(op)
    return [
        Problem(op, shape, dtype) for shape in spec.default_shapes[:2] for dtype in spec.dtypes[:1]
    ]
