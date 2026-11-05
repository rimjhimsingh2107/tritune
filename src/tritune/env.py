"""Figure out what hardware/toolchain we're actually running on.

Every probe is defensive: a missing driver or a broken torch install should
produce a readable 'unavailable because ...' line, never a traceback.
"""

from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from typing import Any

from tritune import __version__


@dataclass
class Probe:
    name: str
    ok: bool
    detail: str


@dataclass
class Environment:
    python: str
    platform: str
    machine: str
    tritune: str
    numpy: str | None = None
    torch: str | None = None
    triton: str | None = None
    cuda_available: bool = False
    cuda_runtime: str | None = None
    driver: str | None = None
    gpu_name: str | None = None
    compute_capability: str | None = None
    gpu_count: int = 0
    ncu: str | None = None
    ncu_version: str | None = None
    probes: list[Probe] = field(default_factory=list)

    @property
    def can_run_triton(self) -> bool:
        return bool(self.cuda_available and self.torch and self.triton)

    def fingerprint(self, backend: str) -> dict[str, Any]:
        """The parts of the environment that can change a compile or a timing."""
        return {
            "backend": backend,
            "gpu": self.gpu_name,
            "cc": self.compute_capability,
            "driver": self.driver,
            "cuda": self.cuda_runtime,
            "torch": self.torch,
            "triton": self.triton,
            "tritune": self.tritune,
        }

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["can_run_triton"] = self.can_run_triton
        return d


def _version(module: str) -> str | None:
    if importlib.util.find_spec(module) is None:
        return None
    try:
        mod = __import__(module)
        return str(getattr(mod, "__version__", "unknown"))
    except Exception as exc:  # broken installs happen (ABI mismatch etc.)
        return f"broken ({type(exc).__name__}: {exc})"


def _run(cmd: list[str], timeout: float = 10.0) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def find_ncu() -> str | None:
    explicit = os.environ.get("TRITUNE_NCU")
    if explicit:
        return explicit if os.path.exists(explicit) else None
    found = shutil.which("ncu")
    if found:
        return found
    for guess in ("/usr/local/cuda/bin/ncu", "/opt/nvidia/nsight-compute/ncu"):
        if os.path.exists(guess):
            return guess
    return None


def detect() -> Environment:
    env = Environment(
        python=sys.version.split()[0],
        platform=f"{platform.system()} {platform.release()}",
        machine=platform.machine(),
        tritune=__version__,
        numpy=_version("numpy"),
        torch=_version("torch"),
        triton=_version("triton"),
    )
    probes = env.probes
    probes.append(Probe("numpy", env.numpy is not None, env.numpy or "not installed"))

    if env.torch and not env.torch.startswith("broken"):
        try:
            import torch

            env.cuda_available = bool(torch.cuda.is_available())
            env.cuda_runtime = torch.version.cuda
            if env.cuda_available:
                env.gpu_count = torch.cuda.device_count()
                env.gpu_name = torch.cuda.get_device_name(0)
                major, minor = torch.cuda.get_device_capability(0)
                env.compute_capability = f"{major}.{minor}"
            probes.append(
                Probe(
                    "torch.cuda",
                    env.cuda_available,
                    f"{env.gpu_name} (sm_{env.compute_capability})"
                    if env.cuda_available
                    else "torch installed but no CUDA device visible",
                )
            )
        except Exception as exc:
            probes.append(Probe("torch.cuda", False, f"probe failed: {exc}"))
    else:
        probes.append(
            Probe("torch", False, env.torch or "not installed (pip install tritune[gpu])")
        )

    probes.append(
        Probe(
            "triton",
            bool(env.triton and not env.triton.startswith("broken")),
            env.triton or "not installed (no wheels for macOS / Windows)",
        )
    )

    smi = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    if smi:
        env.driver = smi.splitlines()[0].strip()
    probes.append(Probe("nvidia driver", smi is not None, env.driver or "nvidia-smi not found"))

    env.ncu = find_ncu()
    if env.ncu:
        out = _run([env.ncu, "--version"])
        env.ncu_version = out.splitlines()[-1] if out else "unknown"
    probes.append(
        Probe(
            "nsight compute",
            env.ncu is not None,
            env.ncu_version or "ncu not on PATH (profiling disabled)",
        )
    )
    return env
