# Pytest configuration: every backend test runs on each available backend
# Author: Ammar Mian
# Date: 21/10/2025

import pytest
import torch

# ── Backend availability predicates ───────────────────────────────────────────

def cupy_available() -> bool:
    """True if CuPy is installed and a CUDA device is present."""
    try:
        import cupy
        return cupy.cuda.is_available()
    except (ImportError, Exception):
        return False


def jax_available() -> bool:
    """True if JAX (CPU) is installed."""
    try:
        import jax  # noqa: F401
        return True
    except ImportError:
        return False


def jax_cuda_available() -> bool:
    """True if JAX is installed and a CUDA GPU is present."""
    try:
        import jax
        return len(jax.devices("gpu")) > 0
    except Exception:
        return False


def jax_metal_available() -> bool:
    """True if jax-metal is installed and a Metal device is present."""
    try:
        import jax
        return len(jax.devices("metal")) > 0
    except Exception:
        return False


# ── Shared parametrize list ───────────────────────────────────────────────────
#
# Use ALL_BACKEND_PARAMS in @pytest.mark.parametrize("backend_name", ALL_BACKEND_PARAMS)
# so new backends are exercised everywhere without touching individual tests.
# Unavailable backends are automatically skipped.

ALL_BACKEND_PARAMS = [
    "numpy",
    "torch-cpu",
    pytest.param(
        "torch-cuda",
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason="CUDA not available"
        ),
    ),
    pytest.param(
        "torch-mps",
        marks=pytest.mark.skipif(
            not torch.backends.mps.is_available(), reason="MPS not available"
        ),
    ),
    pytest.param(
        "cupy",
        marks=pytest.mark.skipif(
            not cupy_available(), reason="CuPy / CUDA not available"
        ),
    ),
    pytest.param(
        "jax-cpu",
        marks=pytest.mark.skipif(not jax_available(), reason="JAX not installed"),
    ),
    pytest.param(
        "jax-cuda",
        marks=pytest.mark.skipif(
            not jax_cuda_available(), reason="JAX GPU (CUDA) not available"
        ),
    ),
    pytest.param(
        "jax-metal",
        marks=pytest.mark.skipif(
            not jax_metal_available(), reason="jax-metal not available"
        ),
    ),
]

# Subset for tests that only make sense on CPU backends
CPU_BACKEND_PARAMS = [
    "numpy",
    "torch-cpu",
    pytest.param(
        "jax-cpu",
        marks=pytest.mark.skipif(not jax_available(), reason="JAX not installed"),
    ),
]


def backend_dtypes(backend_name):
    """Complex data dtypes a backend supports: complex64 and complex128, complex64 only on MPS."""
    if str(backend_name).endswith("mps"):
        return ["complex64"]
    return ["complex64", "complex128"]


BACKEND_DTYPE_PARAMS = []
for _p in ALL_BACKEND_PARAMS:
    _name = _p if isinstance(_p, str) else _p.values[0]
    _marks = () if isinstance(_p, str) else _p.marks
    for _dt in backend_dtypes(_name):
        BACKEND_DTYPE_PARAMS.append(pytest.param(_name, _dt, marks=_marks, id=f"{_name}-{_dt}"))
