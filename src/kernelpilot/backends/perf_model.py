"""Analytic performance model used by the simulated backend.

None of the numbers this produces are measurements. The model exists so the
whole pipeline (search, agent, storage, reports) can be exercised on a laptop,
and so the simulated results respond to config changes in roughly the right
direction: bigger matmul tiles raise arithmetic intensity until registers or
shared memory run out, too few warps starve the memory system, padding wastes
lanes, atomics serialize, spills hurt.

The device numbers are loosely A100-40GB shaped. Treat everything as
"plausible", not "predictive".
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from kernelpilot.kernels.base import Cost, KernelSpec, Resources
from kernelpilot.problem import KernelConfig, Problem, sha256

MODEL_VERSION = "sim-roofline-1"


@dataclass(frozen=True)
class SimDevice:
    name: str = "SIM-A100-40GB (simulated)"
    sms: int = 108
    dram_bw: float = 1.555e12  # bytes/s
    l2_bytes: int = 40 * 1024 * 1024
    fp16_tc_flops: float = 312e12
    tf32_tc_flops: float = 156e12
    fp32_flops: float = 19.5e12
    smem_per_sm: int = 164 * 1024
    smem_per_block: int = 163 * 1024
    regs_per_sm: int = 65536
    max_regs_per_thread: int = 255
    max_threads_per_sm: int = 2048
    max_blocks_per_sm: int = 32
    launch_overhead_s: float = 3.0e-6
    atomic_same_addr_s: float = 2.0e-9

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Prediction:
    time_s: float
    t_mem_s: float
    t_compute_s: float
    occupancy: float
    blocks_per_sm: int
    limiter: str  # what capped blocks/SM
    bound: str  # memory | compute | latency
    achieved_bw: float
    dram_bytes: float
    spilled_regs: int
    wave_efficiency: float

    def as_metrics(self, dev: SimDevice, res: Resources) -> dict[str, float]:
        busy = max(self.time_s, 1e-12)
        return {
            "duration_us": self.time_s * 1e6,
            "dram_throughput_pct": 100.0 * min(1.0, self.dram_bytes / busy / dev.dram_bw),
            "sm_throughput_pct": 100.0 * min(1.0, self.t_compute_s / busy),
            "achieved_occupancy_pct": 100.0 * self.occupancy,
            "registers_per_thread": float(min(res.regs_per_thread, dev.max_regs_per_thread)),
            "shared_mem_per_block": float(res.smem_bytes),
            "spilled_registers": float(self.spilled_regs),
            "wave_efficiency_pct": 100.0 * self.wave_efficiency,
        }


def occupancy(res: Resources, dev: SimDevice) -> tuple[int, float, str]:
    regs = min(res.regs_per_thread, dev.max_regs_per_thread)
    regs_alloc = int(math.ceil(regs / 8) * 8)  # register allocation granularity
    limits = {
        "threads": dev.max_threads_per_sm // res.threads,
        "registers": dev.regs_per_sm // max(1, regs_alloc * res.threads),
        "shared memory": dev.smem_per_sm // res.smem_bytes
        if res.smem_bytes
        else dev.max_blocks_per_sm,
        "block slots": dev.max_blocks_per_sm,
    }
    limiter = min(limits, key=lambda k: limits[k])
    blocks = max(0, limits[limiter])
    return blocks, blocks * res.threads / dev.max_threads_per_sm, limiter


def _l2_miss_rate(
    spec: KernelSpec, cfg: KernelConfig, problem: Problem, concurrent: int, dev: SimDevice
) -> float:
    """Fraction of re-read matmul operand traffic that actually goes to DRAM."""
    _, n, k = problem.shape
    e = problem.elem_bytes
    p = cfg.params
    tn = -(-n // p["BLOCK_N"])
    group = p.get("GROUP_M", 1)
    if group <= 1:
        # row-major: in-flight programs cover concurrent/tn tile rows, but each
        # needs all of B
        working = (max(1, concurrent // max(tn, 1)) * p["BLOCK_M"] * k + k * n) * e
    else:
        cols = max(1, concurrent // group)
        working = (group * p["BLOCK_M"] * k + min(cols, tn) * p["BLOCK_N"] * k) * e
    usable = 0.6 * dev.l2_bytes
    return float(np.clip(working / usable, 0.04, 1.0))


def predict(spec: KernelSpec, cfg: KernelConfig, problem: Problem, dev: SimDevice) -> Prediction:
    res = spec.resources(cfg, problem)
    cost: Cost = spec.cost(cfg, problem)
    blocks, occ, limiter = occupancy(res, dev)
    blocks = max(blocks, 1)
    concurrent = blocks * dev.sms
    waves = math.ceil(cost.programs / concurrent)
    wave_eff = cost.programs / (waves * concurrent)
    spilled = max(0, res.regs_per_thread - dev.max_regs_per_thread)

    # --- memory side ------------------------------------------------------
    dram = cost.dram_bytes
    if spec.name == "matmul":
        m, n, k = problem.shape
        unique = (m * k + k * n + m * n) * problem.elem_bytes
        dram = unique + (cost.dram_bytes - unique) * _l2_miss_rate(
            spec, cfg, problem, concurrent, dev
        )
    elif spec.name == "softmax" and cfg.variant == "online":
        # the second read of a row usually hits L2 when rows are small
        m, ncols = problem.shape
        row_bytes = ncols * problem.elem_bytes
        reread_hit = 0.85 if row_bytes * concurrent < 0.5 * dev.l2_bytes else 0.3
        dram -= reread_hit * m * row_bytes
    if spilled:
        dram += spilled * 4.0 * res.threads * cost.programs * 2  # spill store + reload

    # need enough warps in flight to cover DRAM latency (Little's law-ish)
    warps_per_sm = blocks * res.threads / 32
    latency_hiding = min(1.0, 0.25 + warps_per_sm / 40.0)
    if cfg.num_stages == 1 and spec.name in ("matmul",):
        latency_hiding *= 0.75
    elif cfg.num_stages >= 3 and spec.name == "matmul":
        latency_hiding = min(1.0, latency_hiding * 1.1)
    # tail programs leave SMs idle
    sm_util = min(1.0, cost.programs / dev.sms) if cost.programs < dev.sms else 1.0
    achieved_bw = dev.dram_bw * 0.92 * latency_hiding * max(sm_util, 0.02)
    t_mem = dram / achieved_bw

    # --- compute side -----------------------------------------------------
    if spec.name == "matmul":
        peak = dev.fp16_tc_flops if problem.dtype == "fp16" else dev.tf32_tc_flops
        tile = cfg.params["BLOCK_M"] * cfg.params["BLOCK_N"]
        mma_eff = min(1.0, (tile / (128 * 128)) ** 0.3)
        warp_fit = 1.0 if tile / (cfg.num_warps * 32) >= 32 else 0.8
        stage_eff = {1: 0.6, 2: 0.82, 3: 0.95}.get(cfg.num_stages, 1.0)
        eff = 0.85 * mma_eff * warp_fit * stage_eff * min(1.0, 0.45 + occ) * wave_eff
        if cfg.variant == "fp16acc":
            eff *= 1.15
    else:
        peak = dev.fp32_flops
        eff = 0.5 * min(1.0, 0.3 + occ)
        if spec.name == "softmax":
            eff *= 0.35  # exp() goes through the SFUs
    eff *= cost.lane_efficiency
    t_compute = cost.flops / (peak * max(eff, 1e-3))

    t = max(t_mem, t_compute) + 0.12 * min(t_mem, t_compute)
    t += cost.launches * dev.launch_overhead_s
    t += cost.atomics * dev.atomic_same_addr_s
    if spilled:
        t *= 1.0 + min(1.0, spilled / 64)

    if t_mem >= t_compute:
        bound = "memory"
    else:
        bound = "compute"
    if t < 4 * dev.launch_overhead_s * cost.launches:
        bound = "latency"
    return Prediction(
        time_s=t,
        t_mem_s=t_mem,
        t_compute_s=t_compute,
        occupancy=occ,
        blocks_per_sm=blocks,
        limiter=limiter,
        bound=bound,
        achieved_bw=dram / t,
        dram_bytes=dram,
        spilled_regs=spilled,
        wave_efficiency=wave_eff,
    )


def noisy_samples(base_s: float, key: str, n: int, sigma: float = 0.015) -> list[float]:
    """Deterministic 'measurement noise' seeded from the candidate identity."""
    seed = int(sha256(key)[:12], 16)
    rng = np.random.default_rng(seed)
    noise = rng.lognormal(mean=0.0, sigma=sigma, size=n)
    # occasional slow outlier, like a clock dip or a context switch
    spikes = rng.random(n) < 0.03
    noise[spikes] *= rng.uniform(1.1, 1.4, int(spikes.sum()))
    return [float(base_s * x * 1e3) for x in noise]


def compile_ms(spec: KernelSpec, cfg: KernelConfig, n_kernels: int, key: str) -> float:
    """Rough Triton JIT time: frontend + LLVM + ptxas, grows with unrolled work."""
    seed = int(sha256("compile" + key)[:8], 16)
    rng = np.random.default_rng(seed)
    base = 160.0 + 70.0 * n_kernels
    work = 0.0
    if spec.name == "matmul":
        p = cfg.params
        work = (p["BLOCK_M"] * p["BLOCK_N"]) / 256 + 25 * cfg.num_stages
    elif spec.name == "reduction":
        work = 6.0 * cfg.params.get("TILES", 1)
    elif spec.name == "softmax":
        work = 40.0 if cfg.variant == "online" else 10.0
    return float((base + work) * rng.uniform(0.9, 1.15))
