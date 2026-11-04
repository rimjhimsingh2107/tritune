"""Lightweight PTX inspection.

Not a PTX parser in any serious sense -- regexes over the text Triton hands
back in CompiledKernel.asm['ptx']. Enough to answer the questions that matter
when deciding what to try next: are loads vectorised, are tensor cores used,
is there a cp.async pipeline, is anything going to local memory.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass, field

_INSTR = re.compile(r"^\s*(?:@!?%p\d+\s+)?([a-z][a-z0-9_]*(?:\.[a-z0-9_]+)*)", re.M)
_REG_DECL = re.compile(r"\.reg\s+\.(\w+)\s+%\w+<(\d+)>")
_ENTRY = re.compile(r"\.(?:visible\s+)?\.entry\s+(\w+)")
_TARGET = re.compile(r"\.target\s+(sm_\w+)")


@dataclass
class PtxSummary:
    entries: list[str] = field(default_factory=list)
    target: str | None = None
    instructions: int = 0
    global_loads: int = 0
    global_stores: int = 0
    vector_global_loads: int = 0
    shared_loads: int = 0
    shared_stores: int = 0
    local_loads: int = 0
    local_stores: int = 0
    mma: int = 0
    cp_async: int = 0
    barriers: int = 0
    atomics: int = 0
    ex2: int = 0
    declared_regs: dict[str, int] = field(default_factory=dict)
    top_opcodes: list[tuple[str, int]] = field(default_factory=list)

    @property
    def vectorized_load_ratio(self) -> float:
        return self.vector_global_loads / self.global_loads if self.global_loads else 0.0

    def observations(self) -> list[str]:
        notes = []
        if self.local_loads or self.local_stores:
            notes.append(
                f"local memory traffic ({self.local_loads} ld.local / {self.local_stores} st.local): "
                "likely register spills"
            )
        if self.global_loads and self.vectorized_load_ratio < 0.5:
            notes.append(
                f"only {self.vectorized_load_ratio:.0%} of global loads are vectorised; "
                "check alignment / contiguity hints"
            )
        if self.mma:
            notes.append(f"tensor cores in use ({self.mma} mma instructions)")
        if self.cp_async:
            notes.append(f"async copy pipeline present ({self.cp_async} cp.async)")
        if self.atomics:
            notes.append(f"{self.atomics} atomic instructions")
        return notes

    def to_dict(self) -> dict:
        d = asdict(self)
        d["vectorized_load_ratio"] = round(self.vectorized_load_ratio, 3)
        d["observations"] = self.observations()
        return d


def summarize_ptx(ptx: str) -> PtxSummary:
    s = PtxSummary()
    s.entries = _ENTRY.findall(ptx)
    m = _TARGET.search(ptx)
    s.target = m.group(1) if m else None
    for kind, count in _REG_DECL.findall(ptx):
        s.declared_regs[kind] = s.declared_regs.get(kind, 0) + int(count)

    ops: Counter[str] = Counter()
    for line in ptx.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("//", ".", "{", "}", "$")) or stripped.endswith(":"):
            continue
        mm = _INSTR.match(line)
        if not mm:
            continue
        op = mm.group(1)
        ops[op] += 1
        if op.startswith("ld.global"):
            s.global_loads += 1
            if ".v2" in op or ".v4" in op:
                s.vector_global_loads += 1
        elif op.startswith("st.global"):
            s.global_stores += 1
        elif op.startswith("ld.shared"):
            s.shared_loads += 1
        elif op.startswith("st.shared"):
            s.shared_stores += 1
        elif op.startswith("ld.local"):
            s.local_loads += 1
        elif op.startswith("st.local"):
            s.local_stores += 1
        elif op.startswith(("mma", "wgmma")):
            s.mma += 1
        elif op.startswith("cp.async"):
            s.cp_async += 1
        elif op.startswith(("bar.", "barrier")):
            s.barriers += 1
        elif op.startswith(("atom", "red.")):
            s.atomics += 1
        elif op.startswith("ex2"):
            s.ex2 += 1
    s.instructions = sum(ops.values())
    s.top_opcodes = ops.most_common(12)
    return s
