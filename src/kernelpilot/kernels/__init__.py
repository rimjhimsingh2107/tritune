from kernelpilot.kernels.base import Cost, KernelSpec, Resources, Tolerance, Variant
from kernelpilot.kernels.matmul import Matmul
from kernelpilot.kernels.reduction import Reduction
from kernelpilot.kernels.softmax import Softmax
from kernelpilot.kernels.vector_add import VectorAdd

REGISTRY: dict[str, KernelSpec] = {
    spec.name: spec for spec in (VectorAdd(), Softmax(), Reduction(), Matmul())
}


def get_spec(op: str) -> KernelSpec:
    try:
        return REGISTRY[op]
    except KeyError:
        raise KeyError(f"unknown operator {op!r}; known: {', '.join(sorted(REGISTRY))}") from None


__all__ = ["REGISTRY", "Cost", "KernelSpec", "Resources", "Tolerance", "Variant", "get_spec"]
