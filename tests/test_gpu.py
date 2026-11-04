"""Real-hardware tests. Skipped unless torch, triton and a CUDA device are present."""

import pytest

from kernelpilot.env import detect

pytestmark = pytest.mark.gpu

ENV = detect()
if not ENV.can_run_triton:
    pytest.skip("needs torch + triton + a CUDA device", allow_module_level=True)

from kernelpilot.backends import WorkerBackend  # noqa: E402
from kernelpilot.backends.isolation import SubprocessExecutor  # noqa: E402
from kernelpilot.candidate import Candidate  # noqa: E402
from kernelpilot.kernels import REGISTRY, get_spec  # noqa: E402
from kernelpilot.problem import Problem  # noqa: E402

SHAPES = {
    "vector_add": (1 << 20,),
    "softmax": (1024, 1000),
    "reduction": ((1 << 20) + 7,),
    "matmul": (512, 384, 640),
}


@pytest.fixture(scope="module")
def backend():
    return WorkerBackend("triton", SubprocessExecutor(), ENV, timeout_s=300)


@pytest.mark.parametrize("op", sorted(REGISTRY))
@pytest.mark.parametrize("dtype", ["fp16", "fp32"])
def test_searchable_variants_compile_validate_and_time(backend, op, dtype):
    p = Problem(op, SHAPES[op], dtype)
    spec = get_spec(op)
    seen = set()
    for cfg in spec.configs(p):
        if cfg.variant in seen:
            continue
        seen.add(cfg.variant)
        cand = Candidate.from_config(p, cfg)
        comp = backend.compile(cand)
        assert comp.ok, comp.error
        assert comp.metadata.get("n_regs") is not None
        assert comp.compile_ms and comp.compile_ms > 0
        val = backend.validate(cand)
        assert val.ok, val.summary()
        bench = backend.benchmark(cand, warmup=3, reps=10)
        assert bench.ok and bench.p50_ms > 0 and not bench.simulated


def test_broken_variants_fail_validation_on_hardware(backend):
    p = Problem("softmax", (512, 1024), "fp16")
    cand = Candidate.from_config(p, get_spec("softmax").baseline(p).replace(variant="naive"))
    assert backend.compile(cand).ok
    assert not backend.validate(cand).ok


def test_ptx_is_captured(backend):
    p = Problem("vector_add", (4096,), "fp32")
    comp = backend.compile(Candidate.from_config(p, get_spec("vector_add").baseline(p)))
    assert comp.ptx and ".entry" in comp.ptx


def test_shared_memory_overflow_is_reported(backend):
    p = Problem("matmul", (1024, 1024, 1024), "fp32")
    cfg = get_spec("matmul").baseline(p).replace(BLOCK_M=256, BLOCK_N=256, BLOCK_K=64, num_stages=5)
    comp = backend.compile(Candidate.from_config(p, cfg))
    assert not comp.ok and comp.error_kind == "resources"
