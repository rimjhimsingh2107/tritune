from kernelpilot.analysis.static_check import lint
from kernelpilot.kernels import REGISTRY
from kernelpilot.problem import Problem


def _baseline_src(op, variant=None):
    spec = REGISTRY[op]
    p = Problem(op, spec.default_shapes[0])
    cfg = spec.baseline(p)
    if variant:
        cfg = cfg.replace(variant=variant)
    return spec.render(cfg)


def test_all_templates_lint_clean():
    for op, spec in REGISTRY.items():
        for variant in spec.variants:
            report = lint(_baseline_src(op, variant))
            assert report.ok, (op, variant, report.error_text())
            assert report.has_launch
            assert report.fingerprint


def test_syntax_error_reports_line():
    report = lint("import triton\ndef launch(x:\n    pass\n")
    assert not report.ok
    assert "SyntaxError" in report.error_text()
    assert "line 2" in report.error_text()


def test_non_pow2_arange_is_caught_via_constexpr_binding():
    src = _baseline_src("softmax").replace("BLOCK_N = 1024", "BLOCK_N = 1000")
    report = lint(src)
    assert not report.ok
    assert "power of 2" in report.error_text()


def test_hallucinated_tl_function_and_numpy_call():
    src = _baseline_src("softmax").replace("tl.exp(x)", "tl.fast_exp(x)")
    src = src.replace("den = tl.sum(num, axis=0)", "den = np.sum(num)")
    errors = lint(src).error_text()
    assert "has no attribute 'fast_exp'" in errors
    assert "non-jit function np.sum" in errors


def test_unsupported_constructs():
    src = '''
import triton
import triton.language as tl

@triton.jit
def k(x_ptr, n, BLOCK: tl.constexpr):
    for i in range(n):
        if i > 3:
            break
    vals = [i for i in range(4)]
    try:
        tl.store(x_ptr, undefined_thing)
    except Exception:
        pass
'''
    report = lint(src)
    text = report.error_text()
    assert "Break" in text
    assert "list comprehension" in text
    assert "try/except" in text
    assert "name 'undefined_thing' is not defined" in text
    assert "must define launch" in text


def test_print_is_a_warning_not_an_error():
    src = _baseline_src("vector_add").replace("    tl.store(", "    print(pid)\n    tl.store(")
    report = lint(src)
    assert report.ok
    assert any("device_print" in w.message for w in report.warnings)


def test_fingerprint_ignores_constants_but_not_kernel_body():
    a = _baseline_src("softmax")
    b = a.replace("NUM_WARPS = 4", "NUM_WARPS = 8")
    c = a.replace("x = x - tl.max(x, axis=0)", "x = x - tl.min(x, axis=0)")
    assert lint(a).fingerprint == lint(b).fingerprint
    assert lint(a).fingerprint != lint(c).fingerprint
