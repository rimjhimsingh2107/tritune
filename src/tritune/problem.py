"""Problem + config types shared by every part of the system.

A Problem is "which operator, which shape, which dtype". A KernelConfig is
"which kernel variant, with which launch/tile parameters". Candidates are
the cross product of the two plus rendered source.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

DTYPES = {"fp16": np.float16, "fp32": np.float32}
DTYPE_BYTES = {"fp16": 2, "fp32": 4}


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def next_pow2(n: int) -> int:
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


@dataclass(frozen=True)
class Problem:
    op: str
    shape: tuple[int, ...]
    dtype: str = "fp32"

    def __post_init__(self) -> None:
        if self.dtype not in DTYPES:
            raise ValueError(f"unsupported dtype {self.dtype!r} (want one of {list(DTYPES)})")
        if not self.shape or any(int(d) <= 0 for d in self.shape):
            raise ValueError(f"bad shape {self.shape!r}")
        object.__setattr__(self, "shape", tuple(int(d) for d in self.shape))

    @property
    def key(self) -> str:
        dims = "x".join(str(d) for d in self.shape)
        return f"{self.op}:{dims}:{self.dtype}"

    @property
    def np_dtype(self) -> type:
        return DTYPES[self.dtype]

    @property
    def elem_bytes(self) -> int:
        return DTYPE_BYTES[self.dtype]

    def to_dict(self) -> dict[str, Any]:
        return {"op": self.op, "shape": list(self.shape), "dtype": self.dtype}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Problem:
        return cls(op=d["op"], shape=tuple(d["shape"]), dtype=d.get("dtype", "fp32"))

    @classmethod
    def parse(cls, text: str) -> Problem:
        """Parse 'matmul:1024x1024x512:fp16' style keys."""
        parts = text.split(":")
        if len(parts) not in (2, 3):
            raise ValueError(f"can't parse problem {text!r}")
        shape = tuple(int(x) for x in parts[1].lower().split("x"))
        return cls(parts[0], shape, parts[2] if len(parts) == 3 else "fp32")


@dataclass(frozen=True)
class KernelConfig:
    variant: str
    params: dict[str, int] = field(default_factory=dict)
    num_warps: int = 4
    num_stages: int = 2

    def __hash__(self) -> int:
        return hash(self.key)

    @property
    def key(self) -> str:
        return canonical_json(self.to_dict())

    def get(self, name: str, default: int | None = None) -> int | None:
        return self.params.get(name, default)

    def replace(self, **changes: Any) -> KernelConfig:
        params = dict(self.params)
        variant = changes.pop("variant", self.variant)
        warps = changes.pop("num_warps", self.num_warps)
        stages = changes.pop("num_stages", self.num_stages)
        params.update(changes)
        return KernelConfig(variant, params, warps, stages)

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "params": dict(sorted(self.params.items())),
            "num_warps": self.num_warps,
            "num_stages": self.num_stages,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> KernelConfig:
        return cls(
            variant=d["variant"],
            params={k: int(v) for k, v in d.get("params", {}).items()},
            num_warps=int(d.get("num_warps", 4)),
            num_stages=int(d.get("num_stages", 2)),
        )

    def short(self) -> str:
        p = " ".join(f"{k}={v}" for k, v in sorted(self.params.items()))
        return f"{self.variant} {p} w{self.num_warps} s{self.num_stages}".strip()
