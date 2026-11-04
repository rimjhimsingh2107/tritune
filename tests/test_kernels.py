import ast

import numpy as np
import pytest

from kernelpilot.correctness import compare
from kernelpilot.kernels import REGISTRY, get_spec
from kernelpilot.problem import KernelConfig, Problem

SMALL = {
    "vector_add": (5000,),
    "softmax": (64, 300),
    "reduction": (100_000,),
    "matmul": (96, 80, 200),
}


def small(op, dtype="fp32"):
    return Problem(op, SMALL[op], dtype)


@pytest.mark.parametrize("op", sorted(REGISTRY))
@pytest.mark.parametrize("dtype", ["fp16", "fp32"])
def test_baseline_is_valid_and_correct(op, dtype):
    spec = get_spec(op)
    p = small(op, dtype)
    cfg = spec.baseline(p)
    assert spec.check(cfg, p) == []
    inputs = spec.make_inputs(p)
    check = compare(
        spec.emulate(cfg, inputs, p), spec.reference(inputs, p), spec.tolerance(p), p.key
    )
    assert check.ok, check


@pytest.mark.parametrize("op", sorted(REGISTRY))
def test_inputs_are_reproducible(op):
    spec = get_spec(op)
    p = small(op)
    a = spec.make_inputs(p, seed=3)
    b = spec.make_inputs(p, seed=3)
    c = spec.make_inputs(p, seed=4)
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x, y)
    assert not np.array_equal(a[0], c[0])


@pytest.mark.parametrize("op", sorted(REGISTRY))
def test_search_space_only_has_searchable_valid_configs(op):
    spec = get_spec(op)
    p = small(op, "fp16")
    configs = spec.configs(p)
    assert configs
    assert len({c.key for c in configs}) == len(configs)
    for cfg in configs:
        assert spec.variants[cfg.variant].searchable
        assert spec.check(cfg, p) == []


@pytest.mark.parametrize("op", sorted(REGISTRY))
def test_rendered_source_is_python_with_launch(op):
    spec = get_spec(op)
    p = small(op)
    for variant in spec.variants:
        cfg = spec.baseline(p).replace(variant=variant)
        if variant == "grouped":
            cfg = cfg.replace(GROUP_M=8)
        src = spec.render(cfg)
        tree = ast.parse(src)
        names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
        assert "launch" in names
        assert "$" not in src
        assert "@triton.jit" in src


def test_reference_values():
    sm = get_spec("softmax")
    p = Problem("softmax", (2, 3))
    out = sm.reference([np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])], p)
    np.testing.assert_allclose(out.sum(axis=1), [1.0, 1.0])
    np.testing.assert_allclose(out[0], [1 / 3] * 3)

    red = get_spec("reduction")
    assert red.reference([np.ones(10, np.float32)], Problem("reduction", (10,)))[0] == 10.0


def test_naive_softmax_overflows_in_fp16_but_not_fp32():
    spec = get_spec("softmax")
    for dtype, should_pass in [("fp16", False), ("fp32", True)]:
        p = Problem("softmax", (64, 512), dtype)
        inputs = spec.make_inputs(p)
        cfg = KernelConfig("naive", {"BLOCK_N": 512})
        check = compare(
            spec.emulate(cfg, inputs, p), spec.reference(inputs, p), spec.tolerance(p), p.key
        )
        assert check.ok is should_pass
        if not should_pass:
            assert check.nan_count > 0


def test_unmasked_add_only_breaks_on_ragged_sizes():
    spec = get_spec("vector_add")
    cfg = KernelConfig("unmasked", {"BLOCK_SIZE": 1024})
    for n, should_pass in [(4096, True), (4100, False)]:
        p = Problem("vector_add", (n,))
        inputs = spec.make_inputs(p)
        check = compare(
            spec.emulate(cfg, inputs, p), spec.reference(inputs, p), spec.tolerance(p), p.key
        )
        assert check.ok is should_pass


def test_fp16_accumulators_fail_tolerance():
    mm = get_spec("matmul")
    p = Problem("matmul", (128, 128, 2048), "fp16")
    inputs = mm.make_inputs(p)
    cfg = mm.baseline(p).replace(variant="fp16acc")
    check = compare(mm.emulate(cfg, inputs, p), mm.reference(inputs, p), mm.tolerance(p), p.key)
    assert not check.ok

    red = get_spec("reduction")
    p = Problem("reduction", (1 << 20,), "fp16")
    inputs = red.make_inputs(p)
    cfg = KernelConfig("atomic_fp16acc", {"BLOCK_SIZE": 1024, "TILES": 16})
    check = compare(red.emulate(cfg, inputs, p), red.reference(inputs, p), red.tolerance(p), p.key)
    assert not check.ok


def test_check_catches_bad_launch_params():
    mm = get_spec("matmul")
    p = Problem("matmul", (64, 64, 64), "fp16")
    bad = KernelConfig(
        "tiled", {"BLOCK_M": 96, "BLOCK_N": 8, "BLOCK_K": 32, "GROUP_M": 1}, num_warps=3
    )
    errs = " ".join(mm.check(bad, p))
    assert "BLOCK_M=96" in errs
    assert "num_warps=3" in errs
    assert ">= 16" in errs

    sm = get_spec("softmax")
    p = Problem("softmax", (8, 3000))
    assert sm.check(KernelConfig("single_pass", {"BLOCK_N": 2048}), p)


def test_compare_flags_shape_mismatch_and_nans():
    from kernelpilot.kernels.base import Tolerance

    tol = Tolerance(1e-6, 1e-6)
    c = compare(np.zeros(3), np.zeros(4), tol, "x")
    assert not c.ok and "shape" in c.message
    c = compare(np.array([np.nan, 1.0]), np.array([0.0, 1.0]), tol, "x")
    assert not c.ok and c.nan_count == 1
