import pytest

from kernelpilot.backends import CachedBackend, WorkerBackend, make_backend
from kernelpilot.backends.isolation import InProcessExecutor, SubprocessExecutor
from kernelpilot.backends.perf_model import SimDevice, predict
from kernelpilot.backends.sim_engine import identify
from kernelpilot.cache import ResultCache
from kernelpilot.candidate import Candidate
from kernelpilot.config import load_settings
from kernelpilot.env import detect
from kernelpilot.kernels import get_spec
from kernelpilot.problem import KernelConfig, Problem


@pytest.fixture(scope="module")
def env():
    return detect()


@pytest.fixture
def backend(env):
    return CachedBackend(WorkerBackend("sim", InProcessExecutor(), env), ResultCache(":memory:"))


def cand(key, **changes):
    p = Problem.parse(key)
    spec = get_spec(p.op)
    cfg = spec.baseline(p)
    if changes:
        cfg = cfg.replace(**changes)
    return Candidate.from_config(p, cfg)


def test_identify_recovers_config_from_source():
    c = cand("matmul:512x512x512:fp16", variant="grouped", GROUP_M=8, BLOCK_M=128, num_warps=8)
    cfg = identify(c.source, c.problem)
    assert cfg is not None
    assert cfg.variant == "grouped"
    assert cfg.params["BLOCK_M"] == 128 and cfg.params["GROUP_M"] == 8
    assert cfg.num_warps == 8
    assert identify(c.source.replace("pid_n = (pid", "pid_n = 0 * (pid"), c.problem) is None


def test_full_pipeline_on_baseline(backend):
    c = cand("softmax:1024x1024:fp16")
    comp = backend.compile(c)
    assert comp.ok and comp.simulated
    assert comp.metadata["n_regs"] > 0
    val = backend.validate(c)
    assert val.ok and val.status == "verified"
    bench = backend.benchmark(c, warmup=2, reps=20)
    assert bench.ok and bench.n == 20 and bench.p50_ms <= bench.p95_ms
    prof = backend.profile(c)
    assert prof.ok and "dram_throughput_pct" in prof.metrics


def test_shared_memory_overflow_is_a_compile_error(backend):
    c = cand("matmul:2048x2048x2048:fp16", BLOCK_M=256, BLOCK_N=256, BLOCK_K=64, num_stages=4)
    comp = backend.compile(c)
    assert not comp.ok
    assert comp.error_kind == "resources"
    assert "out of resource: shared memory" in comp.error


def test_lint_failure_propagates(backend):
    c = cand("vector_add:4096:fp32")
    c.source = c.source.replace("BLOCK_SIZE = 1024", "BLOCK_SIZE = 1000")
    comp = backend.compile(c)
    assert not comp.ok and comp.error_kind == "lint"
    val = backend.validate(c)
    assert not val.ok and val.status == "error"
    assert not backend.benchmark(c, 1, 3).ok


def test_numerical_failure_is_detected(backend):
    c = cand("softmax:512x512:fp16", variant="naive")
    assert backend.compile(c).ok
    val = backend.validate(c)
    assert not val.ok and val.status == "failed"
    assert val.worst.nan_count > 0


def test_unrecognised_kernel_is_unverified_not_passed(backend):
    c = cand("softmax:512x512:fp32")
    c.source = c.source.replace("num = tl.exp(x)", "num = tl.exp(x) * 1.0")
    assert backend.compile(c).ok
    val = backend.validate(c)
    assert val.status == "unverified" and not val.ok
    assert not backend.benchmark(c, 1, 3).ok


def test_benchmark_is_deterministic_and_cached(backend):
    c = cand("reduction:1048576:fp32")
    a = backend.benchmark(c, 2, 10)
    b = backend.benchmark(c, 2, 10)
    assert a.samples_ms == b.samples_ms
    assert b.cached and not a.cached
    assert backend.cache.hits >= 1


def test_cache_key_changes_with_env(env):
    cache = ResultCache(":memory:")
    b1 = CachedBackend(WorkerBackend("sim", InProcessExecutor(), env), cache)
    c = cand("vector_add:4096:fp32")
    b1.compile(c)
    b2 = CachedBackend(WorkerBackend("sim", InProcessExecutor(), env), cache)
    b2._fp = {**b2._fp, "triton": "9.9.9"}
    assert not b2.compile(c).cached
    assert b1.compile(c).cached


def test_perf_model_spills_hurt():
    spec = get_spec("softmax")
    p = Problem("softmax", (2048, 16384), "fp32")
    dev = SimDevice()
    few = predict(spec, KernelConfig("single_pass", {"BLOCK_N": 16384}, num_warps=4), p, dev)
    many = predict(spec, KernelConfig("single_pass", {"BLOCK_N": 16384}, num_warps=16), p, dev)
    assert few.spilled_regs > 0 and many.spilled_regs == 0
    assert many.time_s < few.time_s


def test_make_backend_auto_falls_back_to_sim(env, tmp_path):
    if env.can_run_triton:
        pytest.skip("real GPU present")
    b = make_backend(load_settings(backend="auto", db_path=tmp_path / "kp.db"), env)
    assert b.simulated


def test_make_backend_explains_missing_gpu(env, tmp_path):
    if env.can_run_triton:
        pytest.skip("real GPU present")
    from kernelpilot.backends import BackendUnavailable

    with pytest.raises(BackendUnavailable, match="CUDA"):
        make_backend(load_settings(backend="triton", db_path=tmp_path / "kp.db"), env)


class TestSubprocessIsolation:
    def test_roundtrip(self, env):
        b = WorkerBackend("sim", SubprocessExecutor(), env, timeout_s=60)
        c = cand("vector_add:4096:fp16")
        assert b.compile(c).ok
        assert b.validate(c).ok

    def test_timeout_kills_worker(self):
        res = SubprocessExecutor().run({"_sleep": 30, "engine": "sim", "action": "compile"}, timeout=1.5)
        assert res["ok"] is False and res["error_kind"] == "timeout"

    def test_hard_crash_is_reported(self):
        res = SubprocessExecutor().run({"_crash": 3, "engine": "sim", "action": "compile"}, timeout=30)
        assert res["error_kind"] == "crash" and "code 3" in res["error"]

    def test_secrets_are_not_passed_down(self, monkeypatch):
        from kernelpilot.backends.isolation import scrubbed_env

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        monkeypatch.setenv("KP_SEED", "1")
        env = scrubbed_env()
        assert "ANTHROPIC_API_KEY" not in env and env["KP_SEED"] == "1"
