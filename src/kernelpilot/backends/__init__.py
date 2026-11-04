from __future__ import annotations

import time
from typing import Any

from kernelpilot.backends.base import Backend, BenchmarkResult, CompileResult, ProfileResult
from kernelpilot.backends.isolation import Executor, make_executor
from kernelpilot.cache import ResultCache, cache_key
from kernelpilot.candidate import Candidate
from kernelpilot.config import Settings
from kernelpilot.correctness import ValidationResult
from kernelpilot.env import Environment, detect
from kernelpilot.log import get_logger, kv

log = get_logger("backend")

__all__ = [
    "Backend",
    "BackendUnavailable",
    "BenchmarkResult",
    "CachedBackend",
    "CompileResult",
    "ProfileResult",
    "WorkerBackend",
    "make_backend",
]

TRANSIENT = {"timeout", "crash"}


class BackendUnavailable(RuntimeError):
    pass


class WorkerBackend(Backend):
    """Ships each request to an engine ('sim' or 'triton') through an executor."""

    def __init__(self, engine: str, executor: Executor, env: Environment, timeout_s: float = 120.0):
        self.engine = engine
        self.name = engine
        self.simulated = engine == "sim"
        self.executor = executor
        self.env = env
        self.timeout_s = timeout_s

    def fingerprint(self) -> dict[str, Any]:
        fp = self.env.fingerprint(self.engine)
        if self.simulated:
            from kernelpilot.backends import perf_model

            fp.update(
                gpu=perf_model.SimDevice().name,
                model=perf_model.MODEL_VERSION,
                cc=None,
                driver=None,
                cuda=None,
            )
        return fp

    def describe(self) -> str:
        if self.simulated:
            return "simulated (no GPU; analytic model, numerics emulated on CPU)"
        return f"triton on {self.env.gpu_name} (sm_{self.env.compute_capability})"

    def _call(self, action: str, cand: Candidate, **extra: Any) -> dict[str, Any]:
        req = {
            "engine": self.engine,
            "action": action,
            "source": cand.source,
            "problem": cand.problem.to_dict(),
            **extra,
        }
        t0 = time.perf_counter()
        res = self.executor.run(req, self.timeout_s)
        log.debug(
            "worker call",
            extra=kv(
                action=action,
                cand=cand.id,
                ok=res.get("ok"),
                ms=round((time.perf_counter() - t0) * 1e3, 1),
            ),
        )
        return res

    def compile(self, cand: Candidate) -> CompileResult:
        res = self._call("compile", cand)
        res["simulated"] = self.simulated
        return CompileResult.from_dict(res)

    def validate(self, cand: Candidate, seed: int = 0) -> ValidationResult:
        res = self._call("validate", cand, seed=seed)
        if "checks" not in res and "status" not in res:
            return ValidationResult(False, status="error", error=res.get("error", "worker failure"))
        return ValidationResult.from_dict(res)

    def benchmark(self, cand: Candidate, warmup: int, reps: int) -> BenchmarkResult:
        res = self._call("benchmark", cand, warmup=warmup, reps=reps)
        if not res.get("ok"):
            return BenchmarkResult(ok=False, error=res.get("error"), simulated=self.simulated)
        return BenchmarkResult.from_samples(
            res["samples_ms"], warmup=warmup, method=res.get("method", ""), simulated=self.simulated
        )

    def profile(self, cand: Candidate) -> ProfileResult:
        res = self._call("profile", cand, ncu=self.env.ncu)
        res["simulated"] = self.simulated
        return ProfileResult.from_dict(res)


class CachedBackend(Backend):
    def __init__(self, inner: Backend, cache: ResultCache):
        self.inner = inner
        self.cache = cache
        self.name = inner.name
        self.simulated = inner.simulated
        self._fp = inner.fingerprint()

    def fingerprint(self) -> dict[str, Any]:
        return self._fp

    def describe(self) -> str:
        return self.inner.describe()

    def _key(self, stage: str, cand: Candidate, **extra: Any) -> str:
        return cache_key(stage, cand.source_hash, cand.problem.key, self._fp, **extra)

    def compile(self, cand: Candidate) -> CompileResult:
        key = self._key("compile", cand)
        hit = self.cache.get(key)
        if hit is not None:
            res = CompileResult.from_dict(hit)
            res.cached = True
            return res
        res = self.inner.compile(cand)
        if res.error_kind not in TRANSIENT:
            self.cache.put(key, "compile", res.to_dict())
        return res

    def validate(self, cand: Candidate, seed: int = 0) -> ValidationResult:
        key = self._key("validate", cand, seed=seed)
        hit = self.cache.get(key)
        if hit is not None:
            return ValidationResult.from_dict(hit)
        res = self.inner.validate(cand, seed)
        if res.status != "error":
            self.cache.put(key, "validate", res.to_dict())
        return res

    def benchmark(self, cand: Candidate, warmup: int, reps: int) -> BenchmarkResult:
        key = self._key("benchmark", cand, warmup=warmup, reps=reps)
        hit = self.cache.get(key)
        if hit is not None:
            res = BenchmarkResult.from_dict(hit)
            res.cached = True
            return res
        res = self.inner.benchmark(cand, warmup, reps)
        if res.ok:
            self.cache.put(key, "benchmark", res.to_dict())
        return res

    def profile(self, cand: Candidate) -> ProfileResult:
        key = self._key("profile", cand)
        hit = self.cache.get(key)
        if hit is not None:
            return ProfileResult.from_dict(hit)
        res = self.inner.profile(cand)
        if res.ok:
            self.cache.put(key, "profile", res.to_dict())
        return res


def make_backend(
    settings: Settings, env: Environment | None = None, cache: ResultCache | None = None
) -> Backend:
    env = env or detect()
    choice = settings.backend
    if choice == "auto":
        choice = "triton" if env.can_run_triton else "sim"
    if choice == "triton" and not env.can_run_triton:
        reasons = [
            p.detail for p in env.probes if not p.ok and p.name in ("torch", "torch.cuda", "triton")
        ]
        raise BackendUnavailable(
            "triton backend needs torch + triton + a CUDA device: " + "; ".join(reasons)
        )

    isolation = settings.isolation or ("subprocess" if choice == "triton" else "inprocess")
    inner = WorkerBackend(choice, make_executor(isolation), env, settings.timeout_s)
    if not settings.use_cache:
        return inner
    if cache is None:
        settings.ensure_home()
        cache = ResultCache(settings.db_path)
    return CachedBackend(inner, cache)
