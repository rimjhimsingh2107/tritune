"""AST-level checks for Triton candidate modules.

This is what you'd catch by reading the source carefully: missing launch(),
Python constructs the Triton frontend rejects, non-power-of-two aranges,
calls into numpy/torch from inside a @triton.jit body, undefined names.

On the real backend these are only warnings -- the Triton compiler is the
authority. The simulated backend treats them as compile errors because it
has no compiler to defer to. The messages mimic Triton's wording where there
is an equivalent so the agent sees the same text either way.
"""

from __future__ import annotations

import ast
import builtins
from dataclasses import dataclass, field

from kernelpilot.problem import is_pow2

# triton.language names we accept. intentionally generous; this isn't trying
# to track every release, just to flag obvious hallucinations like tl.fast_softmax.
TL_NAMES = frozenset(
    """
    program_id num_programs arange load store zeros zeros_like full dot sum max min
    maximum minimum exp exp2 log log2 sqrt rsqrt abs where atomic_add atomic_max
    atomic_min atomic_cas atomic_xchg atomic_and atomic_or atomic_xor cdiv static_range
    range constexpr float16 float32 float64 bfloat16 int1 int8 int16 int32 int64 uint8
    uint16 uint32 uint64 float8e4nv float8e5 broadcast_to broadcast reshape view trans
    permute expand_dims cumsum cumprod argmax argmin sigmoid softmax fma multiple_of
    max_contiguous max_constancy debug_barrier device_print device_assert static_assert
    static_print umulhi xor_sum make_block_ptr advance split join interleave ravel sort
    flip clamp div_rn fdiv erf floor ceil cos sin associative_scan reduce math extra
    dtype tensor pointer_type void cast histogram inline_asm_elementwise rand randn
    randint philox swizzle2d assume make_tensor_descriptor
    """.split()
)

ALLOWED_BUILTINS = frozenset({"range", "min", "max", "float", "int", "len", "print", "isinstance", "True", "False", "None"})

UNSUPPORTED = {
    ast.Try: "try/except",
    ast.With: "with-statement",
    ast.Lambda: "lambda",
    ast.ListComp: "list comprehension",
    ast.SetComp: "set comprehension",
    ast.DictComp: "dict comprehension",
    ast.GeneratorExp: "generator expression",
    ast.Yield: "yield",
    ast.Global: "global",
    ast.Nonlocal: "nonlocal",
    ast.ClassDef: "class definition",
    ast.Delete: "del",
    ast.Break: "Break",
    ast.Continue: "Continue",
}


@dataclass
class Issue:
    line: int
    col: int
    message: str
    fatal: bool = True

    def render(self, kernel: str | None = None) -> str:
        where = f" in {kernel}" if kernel else ""
        return f"at {self.line}:{self.col}{where}: {self.message}"


@dataclass
class LintReport:
    syntax_error: str | None = None
    issues: list[Issue] = field(default_factory=list)
    kernels: list[str] = field(default_factory=list)
    constants: dict[str, int] = field(default_factory=dict)
    constexpr_bindings: dict[str, dict[str, int]] = field(default_factory=dict)
    has_launch: bool = False
    fingerprint: str | None = None

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.fatal]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if not i.fatal]

    @property
    def ok(self) -> bool:
        return self.syntax_error is None and not self.errors

    def error_text(self) -> str:
        if self.syntax_error:
            return self.syntax_error
        return "\n".join(f"CompilationError {i.render()}" for i in self.errors)


def _is_jit(fn: ast.FunctionDef) -> bool:
    for dec in fn.decorator_list:
        target = dec.func if isinstance(dec, ast.Call) else dec
        if isinstance(target, ast.Attribute) and target.attr in ("jit", "autotune", "heuristics"):
            return True
        if isinstance(target, ast.Name) and target.id == "jit":
            return True
    return False


def _const_eval(node: ast.AST, env: dict[str, int]) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.Name):
        return env.get(node.id)
    if isinstance(node, ast.BinOp):
        left, right = _const_eval(node.left, env), _const_eval(node.right, env)
        if left is None or right is None:
            return None
        ops = {
            ast.Add: lambda a, b: a + b,
            ast.Sub: lambda a, b: a - b,
            ast.Mult: lambda a, b: a * b,
            ast.FloorDiv: lambda a, b: a // b if b else None,
            ast.LShift: lambda a, b: a << b,
        }
        fn = ops.get(type(node.op))
        return fn(left, right) if fn else None
    return None


def _module_constants(tree: ast.Module) -> dict[str, int]:
    out: dict[str, int] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            val = _const_eval(node.value, out)
            if val is not None:
                out[node.targets[0].id] = val
    return out


def _module_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                for n in ast.walk(t):
                    if isinstance(n, ast.Name):
                        names.add(n.id)
    return names


def _launch_bindings(tree: ast.Module, constants: dict[str, int], kernels: set[str]) -> dict[str, dict[str, int]]:
    """Find `kernel[grid](..., NAME=VALUE)` calls and resolve the constexpr kwargs."""
    out: dict[str, dict[str, int]] = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Subscript)):
            continue
        target = node.func.value
        if isinstance(target, ast.Name) and target.id in kernels:
            bound = out.setdefault(target.id, {})
            for kw in node.keywords:
                if kw.arg:
                    val = _const_eval(kw.value, constants)
                    if val is not None:
                        bound[kw.arg] = val
    return out


def _tl_attr(node: ast.AST) -> str | None:
    """'arange' for tl.arange / triton.language.arange, else None."""
    if not isinstance(node, ast.Attribute):
        return None
    base = node.value
    if isinstance(base, ast.Name) and base.id == "tl":
        return node.attr
    if (
        isinstance(base, ast.Attribute)
        and base.attr == "language"
        and isinstance(base.value, ast.Name)
        and base.value.id == "triton"
    ):
        return node.attr
    return None


class _KernelChecker(ast.NodeVisitor):
    def __init__(self, fn: ast.FunctionDef, env: dict[str, int], module_names: set[str], jit_names: set[str]):
        self.fn = fn
        self.env = env
        self.module_names = module_names
        self.jit_names = jit_names
        self.issues: list[Issue] = []
        self.local_names = {a.arg for a in fn.args.args + fn.args.kwonlyargs}
        for node in ast.walk(fn):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                self.local_names.add(node.id)

    def err(self, node: ast.AST, msg: str, fatal: bool = True) -> None:
        self.issues.append(Issue(getattr(node, "lineno", 0), getattr(node, "col_offset", 0), msg, fatal))

    def generic_visit(self, node: ast.AST) -> None:
        for kind, label in UNSUPPORTED.items():
            if isinstance(node, kind):
                self.err(node, f"unsupported AST node type: {label}")
        if isinstance(node, ast.FunctionDef) and node is not self.fn:
            self.err(node, "nested function definitions are not supported in @triton.jit")
            return
        super().generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            known = (
                node.id in self.local_names
                or node.id in self.module_names
                or node.id in ALLOWED_BUILTINS
            )
            if not known:
                if hasattr(builtins, node.id):
                    self.err(node, f"builtin {node.id!r} is not supported inside @triton.jit")
                else:
                    self.err(node, f"NameError: name '{node.id}' is not defined")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        name = _tl_attr(node)
        if name is not None and name not in TL_NAMES:
            self.err(node, f"AttributeError: module 'triton.language' has no attribute '{name}'")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        tl_name = _tl_attr(func)
        if tl_name in ("arange",) and len(node.args) == 2:
            lo, hi = (_const_eval(a, self.env) for a in node.args)
            if lo is not None and hi is not None:
                extent = hi - lo
                if extent <= 0:
                    self.err(node, f"arange's end argument must be greater than start ({lo}, {hi})")
                elif not is_pow2(extent):
                    self.err(node, f"arange's range must be a power of 2 (got {extent})")
                elif extent > 1 << 20:
                    self.err(node, f"numel ({extent}) exceeds triton maximum tensor numel (1048576)")
        elif tl_name in ("zeros", "full") and node.args:
            shape = node.args[0]
            dims = shape.elts if isinstance(shape, (ast.Tuple, ast.List)) else []
            for i, d in enumerate(dims):
                v = _const_eval(d, self.env)
                if v is not None and not is_pow2(v):
                    self.err(node, f"Shape element {i} must be a power of 2 (got {v})")
        elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            owner = func.value.id
            if owner in ("np", "numpy", "torch", "math", "random", "os"):
                self.err(node, f"calling non-jit function {owner}.{func.attr} inside a @triton.jit kernel")
        elif isinstance(func, ast.Name):
            if func.id == "print":
                self.err(node, "print() inside a kernel; use tl.device_print", fatal=False)
            elif (
                func.id not in ALLOWED_BUILTINS
                and func.id not in self.jit_names
                and func.id in self.module_names
            ):
                self.err(node, f"calling non-jit function {func.id!r} from a @triton.jit kernel")
        self.generic_visit(node)


def kernel_fingerprint(tree: ast.Module) -> str | None:
    """Hash of the jit function bodies only. Constants/launch code don't count."""
    from kernelpilot.problem import sha256

    parts = [ast.dump(n) for n in tree.body if isinstance(n, ast.FunctionDef) and _is_jit(n)]
    return sha256("\n".join(parts))[:16] if parts else None


def lint(source: str) -> LintReport:
    report = LintReport()
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        report.syntax_error = f"SyntaxError: {exc.msg} (line {exc.lineno}, col {exc.offset})"
        return report

    report.constants = _module_constants(tree)
    module_names = _module_names(tree)
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    jit_fns = [f for f in funcs if _is_jit(f)]
    report.kernels = [f.name for f in jit_fns]
    report.has_launch = any(f.name == "launch" for f in funcs)
    report.fingerprint = kernel_fingerprint(tree)

    if not jit_fns:
        report.issues.append(Issue(1, 0, "no @triton.jit kernel found in candidate"))
    if not report.has_launch:
        report.issues.append(Issue(1, 0, "candidate must define launch(*inputs) -> output"))

    jit_names = set(report.kernels)
    report.constexpr_bindings = _launch_bindings(tree, report.constants, jit_names)
    for fn in jit_fns:
        env = dict(report.constants)
        env.update(report.constexpr_bindings.get(fn.name, {}))
        checker = _KernelChecker(fn, env, module_names, jit_names)
        for stmt in fn.body:
            checker.visit(stmt)
        report.issues.extend(checker.issues)
    report.issues.sort(key=lambda i: (i.line, i.col))
    return report
