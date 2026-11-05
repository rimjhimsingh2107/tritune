import pytest

from tritune.problem import KernelConfig, Problem, is_pow2, next_pow2


def test_problem_key_roundtrip():
    p = Problem("matmul", (128, 64, 32), "fp16")
    assert p.key == "matmul:128x64x32:fp16"
    assert Problem.parse(p.key) == p
    assert Problem.from_dict(p.to_dict()) == p


def test_problem_rejects_garbage():
    with pytest.raises(ValueError):
        Problem("softmax", (0, 4))
    with pytest.raises(ValueError):
        Problem("softmax", (4, 4), "bf16")
    with pytest.raises(ValueError):
        Problem.parse("softmax")


def test_config_key_is_order_independent():
    a = KernelConfig("tiled", {"BLOCK_M": 64, "BLOCK_N": 32})
    b = KernelConfig("tiled", {"BLOCK_N": 32, "BLOCK_M": 64})
    assert a.key == b.key
    assert hash(a) == hash(b)


def test_config_replace():
    c = KernelConfig("tiled", {"BLOCK_M": 64}, num_warps=4)
    d = c.replace(BLOCK_M=128, num_warps=8, variant="grouped")
    assert d.params["BLOCK_M"] == 128 and d.num_warps == 8 and d.variant == "grouped"
    assert c.params["BLOCK_M"] == 64  # original untouched


@pytest.mark.parametrize("n,want", [(1, 1), (2, 2), (3, 4), (1000, 1024), (1024, 1024)])
def test_next_pow2(n, want):
    assert next_pow2(n) == want
    assert is_pow2(want)
