"""Backend layer: the algorithms of closure_glrt run unchanged on numpy, torch (cpu, cuda, mps),
cupy and jax. They receive a backend name, get the array module with ``get_backend_module`` and
never import a compute library themselves; data enter with ``get_data_on_device`` and leave with
``to_numpy``. Author: Ammar Mian.
"""

from dataclasses import dataclass
from types import ModuleType
from typing import TYPE_CHECKING, Literal, Union

import numpy as np
import numpy.typing as npt
import torch

# ── Type annotations ──────────────────────────────────────────────────────────

# Runtime-safe aliases: only numpy and torch are guaranteed to be present.
type ArrayLike = npt.ArrayLike | torch.Tensor
type Array = npt.NDArray | torch.Tensor

if TYPE_CHECKING:
    # Extend aliases with optional backend types for static analysis.
    # Wrapped in try/except so type-checkers that have the packages installed
    # get the full union; those that don't fall back to the base aliases.
    try:
        import cupy as _cupy_t
        import jax as _jax_t

        type Array = npt.NDArray | torch.Tensor | _cupy_t.ndarray | _jax_t.Array  # type: ignore[no-redef]
        type ArrayLike = npt.ArrayLike | torch.Tensor | _cupy_t.ndarray | _jax_t.Array  # type: ignore[no-redef]
    except ImportError:
        pass


# ── Private helpers ───────────────────────────────────────────────────────────


def _is_cupy_array(x) -> bool:
    """True if *x* is a CuPy ndarray (duck-typed; no cupy import needed)."""
    return type(x).__module__.startswith("cupy")


def _is_jax_array(x) -> bool:
    """True if *x* is a JAX array (duck-typed; no jax import needed)."""
    return type(x).__module__.startswith("jax")


def _is_torch_module(m: ModuleType) -> bool:
    """True if *m* is the torch module (i.e. not an array-API-like module)."""
    return m is torch


# ── Backend dataclass ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Backend:
    """Hardware/library backend descriptor.

    Decouples *library* (numpy / torch / cupy / jax) from *device*
    (cpu / cuda / mps / metal).

    Parameters
    ----------
    lib : Literal["numpy", "torch", "cupy", "jax"]
        The compute library.
    device : str
        Target device: ``"cpu"``, ``"cuda"``, ``"mps"`` (Apple Silicon /
        torch), or ``"metal"`` (Apple Silicon / jax).  Ignored for numpy
        (always CPU).

    Raises
    ------
    ValueError
        If the lib/device combination is unsupported.
    """

    lib: Literal["numpy", "torch", "cupy", "jax"]
    device: str = "cpu"

    def __post_init__(self):
        valid = {
            "numpy": {"cpu"},
            "torch": {"cpu", "cuda", "mps"},
            "cupy": {"cuda"},
            "jax": {"cpu", "cuda", "metal"},
        }
        if self.lib not in valid:
            raise ValueError(f"Unknown lib {self.lib!r}. Valid choices: {sorted(valid)}")
        if self.device not in valid[self.lib]:
            raise ValueError(
                f"Device {self.device!r} is not valid for lib={self.lib!r}. "
                f"Valid choices: {valid[self.lib]}"
            )
        # Pin the JAX default device at construction time so that all
        # jnp.eye / jnp.ones / jnp.zeros etc. land on the right device
        # immediately, without requiring an explicit get_data_on_device call.
        # On Apple Silicon, JAX defaults to Metal; without this, any jnp call
        # with a complex dtype would crash before get_data_on_device can help.
        if self.lib == "jax":
            try:
                import os

                import jax

                # Prevent XLA from pre-allocating ~75% of GPU memory at init.
                os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
                platform = "gpu" if self.device == "cuda" else self.device
                try:
                    jax.config.update("jax_default_device", jax.devices(platform)[0])
                except RuntimeError as e:
                    available = [d.platform for d in jax.devices()]
                    raise RuntimeError(
                        f"JAX cannot use device '{self.device}' (platform='{platform}'). "
                        f"Available JAX devices: {available}. "
                        f"Install jax[cuda] for GPU support, or use 'jax-cpu' instead."
                    ) from e
            except ImportError:
                pass  # JAX not installed; will fail later when actually used

    # ── Factories ──────────────────────────────────────────────────────────

    @staticmethod
    def numpy() -> "Backend":
        return Backend("numpy", "cpu")

    @staticmethod
    def torch_cpu() -> "Backend":
        return Backend("torch", "cpu")

    @staticmethod
    def torch_cuda() -> "Backend":
        return Backend("torch", "cuda")

    @staticmethod
    def torch_mps() -> "Backend":
        return Backend("torch", "mps")

    @staticmethod
    def cupy_cuda() -> "Backend":
        return Backend("cupy", "cuda")

    @staticmethod
    def jax_cpu() -> "Backend":
        return Backend("jax", "cpu")

    @staticmethod
    def jax_cuda() -> "Backend":
        return Backend("jax", "cuda")

    @staticmethod
    def jax_metal() -> "Backend":
        return Backend("jax", "metal")

    # ── Properties ─────────────────────────────────────────────────────────

    @property
    def is_torch(self) -> bool:
        return self.lib == "torch"

    @property
    def is_cupy(self) -> bool:
        return self.lib == "cupy"

    @property
    def is_jax(self) -> bool:
        return self.lib == "jax"

    @property
    def is_cuda(self) -> bool:
        """True for any lib running on a CUDA device."""
        return self.device == "cuda"

    @property
    def is_mps(self) -> bool:
        """True for torch on Apple Silicon GPU."""
        return self.lib == "torch" and self.device == "mps"

    @property
    def is_gpu(self) -> bool:
        """True for any hardware accelerator (cuda, mps, metal)."""
        return self.device in {"cuda", "mps", "metal"}

    @property
    def torch_device(self) -> "torch.device":
        """Corresponding ``torch.device`` — only valid when ``is_torch``."""
        if not self.is_torch:
            raise AttributeError("torch_device is only valid for torch backends")
        return torch.device(self.device)

    @property
    def jax_device(self):
        """Corresponding JAX device object — only valid when ``is_jax``."""
        if not self.is_jax:
            raise AttributeError("jax_device is only valid for jax backends")
        import jax

        # JAX uses "gpu" as the platform name for CUDA devices.
        platform = "gpu" if self.device == "cuda" else self.device
        return jax.devices(platform)[0]

    # ── String round-trip ───────────────────────────────────────────────────

    @classmethod
    def from_str(cls, s: str) -> "Backend":
        """Parse a backend string into a :class:`Backend` object."""
        _map = {
            "numpy": cls.numpy,
            "torch-cpu": cls.torch_cpu,
            "torch-cuda": cls.torch_cuda,
            "torch-mps": cls.torch_mps,
            "cupy": cls.cupy_cuda,
            "cupy-cuda": cls.cupy_cuda,
            "jax-cpu": cls.jax_cpu,
            "jax-cuda": cls.jax_cuda,
            "jax-metal": cls.jax_metal,
        }
        if s not in _map:
            raise ValueError(f"Unknown backend string: {s!r}. Valid choices: {sorted(_map)}")
        return _map[s]()

    def __str__(self) -> str:
        if self.lib == "numpy":
            return "numpy"
        return f"{self.lib}-{self.device}"


# ── Internal normalisation helper ─────────────────────────────────────────────


def _normalize_backend(backend: Union[str, "Backend"]) -> "Backend":
    """Convert string or Backend to Backend object."""
    if isinstance(backend, Backend):
        return backend
    return Backend.from_str(backend)


# ── Module accessor ───────────────────────────────────────────────────────────


def get_backend_module(backend: Union[str, "Backend"]) -> ModuleType:
    """Return the compute module for a given backend.

    Parameters
    ----------
    backend : str or Backend
        Backend specification.

    Returns
    -------
    ModuleType
        ``numpy``, ``torch``, ``cupy``, or ``jax.numpy``.

    Raises
    ------
    ImportError
        If the requested optional library (cupy / jax) is not installed.
    ValueError
        If the backend string is not recognised.
    """
    b = _normalize_backend(backend)
    if b.lib == "numpy":
        return np
    if b.lib == "torch":
        return torch
    if b.lib == "cupy":
        try:
            import cupy

            return cupy
        except ImportError:
            raise ImportError(
                "Backend 'cupy' requires CuPy. Install it with: uv sync --extra cupy"
            ) from None
    if b.lib == "jax":
        try:
            import jax.numpy as jnp

            # Default device already pinned in Backend.__post_init__.
            return jnp
        except ImportError:
            raise ImportError(
                "Backend 'jax' requires JAX. "
                "Install it with: uv sync --extra jax  (CPU/CUDA) or "
                "uv sync --extra jax-metal  (Apple Silicon)"
            ) from None
    raise ValueError(f"backend {backend} not found.")


# ── Data movement ─────────────────────────────────────────────────────────────


_MPS_DOWNCAST = {torch.float64: torch.float32, torch.complex128: torch.complex64}


def get_data_on_device(data: Array, backend: Union[str, "Backend"]) -> Array:
    """Move *data* to the requested backend and device.

    Handles all pairwise conversions between numpy / torch / cupy / jax,
    including automatic float64→float32 downcasting when the target device
    does not support float64 (torch-mps, jax-metal), and complex128→complex64
    on torch-mps. jax-metal has no complex dtypes at all and raises.

    Parameters
    ----------
    data : Array
        Source array (numpy, torch, cupy, or jax).
    backend : str or Backend
        Target backend.

    Returns
    -------
    Array
        Array on the requested backend/device.
    """
    b = _normalize_backend(backend)

    # ── numpy target ──────────────────────────────────────────────────────
    if b.lib == "numpy":
        return to_numpy(data)

    # ── torch target ──────────────────────────────────────────────────────
    if b.lib == "torch":
        tensor = data if isinstance(data, torch.Tensor) else torch.from_numpy(to_numpy(data))
        # MPS has no 64-bit floats: downcast float64 → float32 and
        # complex128 → complex64 (complex64 itself is supported since torch 2.15).
        if b.is_mps and tensor.dtype in _MPS_DOWNCAST:
            tensor = tensor.to(_MPS_DOWNCAST[tensor.dtype])
        return tensor.to(device=b.torch_device)

    # ── cupy target ───────────────────────────────────────────────────────
    if b.lib == "cupy":
        try:
            import cupy as cp
        except ImportError:
            raise ImportError(
                "Backend 'cupy' requires CuPy. Install it with: uv sync --extra cupy"
            ) from None
        if _is_cupy_array(data):
            return data  # already a cupy array (single-GPU assumed)
        return cp.asarray(to_numpy(data))

    # ── jax target ────────────────────────────────────────────────────────
    if b.lib == "jax":
        try:
            import jax
        except ImportError:
            raise ImportError(
                "Backend 'jax' requires JAX. "
                "Install it with: uv sync --extra jax or --extra jax-metal"
            ) from None

        if _is_jax_array(data):
            # Already a JAX array — just move to the requested device.
            return jax.device_put(data, b.jax_device)

        # Convert to numpy first so we have a plain buffer to hand to device_put.
        # We deliberately avoid jnp.asarray() here: on Apple Silicon that call
        # stages through the default backend (Metal) even when targeting CPU,
        # which fails for complex dtypes that Metal does not support.
        arr_np = to_numpy(data)

        if b.device == "metal":
            # Metal does not support float64 → downcast
            if arr_np.dtype == np.float64:
                arr_np = arr_np.astype(np.float32)
            # Metal does not support complex numbers at all
            elif np.issubdtype(arr_np.dtype, np.complexfloating):
                raise TypeError(
                    "JAX Metal backend does not support complex-valued arrays "
                    f"(got dtype {arr_np.dtype}). "
                    "Use --backend torch-mps (complex64 on the Apple GPU) or "
                    "jax-cpu for complex SAR data."
                )

        # jax.device_put accepts a numpy array directly and places it on the
        # requested device without routing through the default backend.
        return jax.device_put(arr_np, b.jax_device)


# ── Central numpy extraction ──────────────────────────────────────────────────


def to_numpy(x: Array) -> np.ndarray:
    """Extract a plain numpy array from any backend array.

    Parameters
    ----------
    x : Array
        Source array (numpy, torch, cupy, or jax).

    Returns
    -------
    np.ndarray
    """
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if _is_cupy_array(x):
        import cupy as cp

        return cp.asnumpy(x)
    if _is_jax_array(x):
        return np.asarray(x)
    # Fallback: try numpy conversion
    return np.asarray(x)


# ── Random sampling ───────────────────────────────────────────────────────────






# ── Array manipulation helpers ────────────────────────────────────────────────










def concatenate(backend: Union[str, "Backend"], arrays: list[Array], axis: int = 0) -> Array:
    """Concatenate arrays along an axis, works for all backends.

    Parameters
    ----------
    backend : str or Backend
    arrays : list[Array]
    axis : int

    Returns
    -------
    Array
    """
    bm = get_backend_module(backend)
    if _is_torch_module(bm):
        return bm.cat(arrays, dim=axis)
    return bm.concatenate(arrays, axis=axis)


# ── Linear algebra helpers ────────────────────────────────────────────────────












# ── Precision policy ──────────────────────────────────────────────────────────




# ── Type/scalar utilities ─────────────────────────────────────────────────────






def cast_like(x: Array, ref: Array, backend: Union[str, "Backend"]) -> Array:
    """Cast *x* to the dtype of *ref*, in a backend-agnostic way.

    Abstracts the ``ndarray.astype()`` (numpy/cupy/jax) vs
    ``Tensor.to(dtype=...)`` (torch) difference so callers need not
    check which API is available.

    Parameters
    ----------
    x : Array
        Array to cast.
    ref : Array
        Reference array whose dtype is used as the target.
    backend : str or Backend

    Returns
    -------
    Array
        *x* with the same dtype as *ref*.
    """
    return to_dtype(x, ref.dtype, backend)




def to_dtype(X: Array, dtype, backend: Union[str, "Backend"]) -> Array:
    """Convert *X* to *dtype*, handling mixed numpy/torch dtype objects.

    Parameters
    ----------
    X : Array
    dtype : np.dtype or torch.dtype
    backend : str or Backend

    Returns
    -------
    Array
    """
    bm = get_backend_module(backend)

    if _is_torch_module(bm):
        # Convert numpy dtypes to torch dtypes if necessary
        if isinstance(dtype, np.dtype):
            _np_to_torch = {
                np.float32: torch.float32,
                np.float64: torch.float64,
                np.float16: torch.float16,
                np.complex64: torch.complex64,
                np.complex128: torch.complex128,
                np.int32: torch.int32,
                np.int64: torch.int64,
            }
            # key on the scalar type: np.dtype('complex64') does not hash like np.complex64
            # (look up np.dtype instances by their scalar type)
            dtype = _np_to_torch.get(dtype.type, dtype)
        return X.to(dtype=dtype)
    else:
        # Convert torch dtypes to numpy dtypes if necessary
        # (cupy and jax both accept numpy-compatible dtypes)
        if isinstance(dtype, torch.dtype):
            _torch_to_np = {
                torch.float32: np.float32,
                torch.float64: np.float64,
                torch.float16: np.float16,
                torch.complex64: np.complex64,
                torch.complex128: np.complex128,
                torch.int32: np.int32,
                torch.int64: np.int64,
            }
            dtype = _torch_to_np.get(dtype, dtype)
        return X.astype(dtype)








# ── Memory / device management helpers ───────────────────────────────────────




    # cupy/jax: no direct reset API; just free and move on






# ── 2-D sliding-window extractor ─────────────────────────────────────────────




# ── Special functions (added for closure-glrt) ────────────────────────────────


def bessel_i0e(backend: Union[str, "Backend"], x: Array) -> Array:
    r"""Exponentially scaled modified Bessel function $e^{-|x|}I_0(x)$, on any backend.

    numpy has no Bessel functions; they live in scipy (numpy), ``torch.special`` (torch,
    float32 on MPS), ``cupyx.scipy.special`` (cupy) and ``jax.scipy.special`` (jax).
    """
    b = _normalize_backend(backend)
    if b.is_torch:
        return torch.special.i0e(x)
    if b.is_cupy:
        import cupyx.scipy.special as cps

        return cps.i0e(x)
    if b.is_jax:
        import jax.scipy.special as jss

        return jss.i0e(x)
    from scipy import special

    return special.i0e(x)


def bessel_i1e(backend: Union[str, "Backend"], x: Array) -> Array:
    r"""Exponentially scaled modified Bessel function $e^{-|x|}I_1(x)$, on any backend."""
    b = _normalize_backend(backend)
    if b.is_torch:
        return torch.special.i1e(x)
    if b.is_cupy:
        import cupyx.scipy.special as cps

        return cps.i1e(x)
    if b.is_jax:
        import jax.scipy.special as jss

        return jss.i1e(x)
    from scipy import special

    return special.i1e(x)
