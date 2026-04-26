"""Numpy-first helpers for inferaxis numeric payloads.

inferaxis's core stores observation and action tensors as ``numpy.ndarray``.
These helpers keep that boundary handling explicit while still accepting
optional torch tensors or simple Python sequences at adapter edges.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import importlib
from numbers import Real
from typing import Any

import numpy as np

from .errors import InterfaceValidationError

_UNSET = object()
_TORCH_TENSOR_TYPE: object = _UNSET


def numpy_ndarray_type() -> type[Any]:
    """Return inferaxis's canonical ndarray type."""

    return np.ndarray


def torch_tensor_type() -> type[Any] | None:
    """Return ``torch.Tensor`` when torch is importable, else ``None``."""

    global _TORCH_TENSOR_TYPE
    if _TORCH_TENSOR_TYPE is _UNSET:
        try:
            module = importlib.import_module("torch")
        except ImportError:
            _TORCH_TENSOR_TYPE = None
        else:
            _TORCH_TENSOR_TYPE = module.Tensor
    return _TORCH_TENSOR_TYPE if isinstance(_TORCH_TENSOR_TYPE, type) else None


def optional_array_to_numpy(
    value: object,
    *,
    field_name: str,
    copy: bool = True,
) -> np.ndarray | None:
    """Convert optional ndarray/tensor inputs into ``numpy.ndarray``.

    Returns ``None`` when ``value`` is not a recognized optional array type.
    Torch tensors are detached and moved to CPU exactly once.
    """

    if isinstance(value, np.ndarray):
        return np.array(value, copy=True) if copy else value

    if isinstance(value, np.generic):
        return np.array(value, copy=True) if copy else np.asarray(value)

    torch_type = torch_tensor_type()
    if torch_type is None or not isinstance(value, torch_type):
        return None

    tensor = value.detach() if callable(getattr(value, "detach", None)) else value
    device = getattr(getattr(tensor, "device", None), "type", None)
    if device not in (None, "cpu"):
        cpu = getattr(tensor, "cpu", None)
        if not callable(cpu):
            raise InterfaceValidationError(
                f"{field_name} torch tensor is on device {device!r}, but does not "
                "expose cpu()."
            )
        tensor = cpu()

    to_numpy = getattr(tensor, "numpy", None)
    if not callable(to_numpy):
        raise InterfaceValidationError(
            f"{field_name} torch tensor does not expose numpy()."
        )
    array = to_numpy()
    return np.array(array, copy=True) if copy else np.asarray(array)


def to_numpy_array(
    value: object,
    *,
    field_name: str,
    wrap_scalar: bool = False,
    numeric_only: bool = False,
    allow_bool: bool = True,
    copy: bool = True,
    dtype: Any | None = None,
) -> np.ndarray:
    """Convert one value into ``numpy.ndarray`` with lightweight validation."""

    converted = optional_array_to_numpy(value, field_name=field_name, copy=copy)
    if converted is not None:
        array = converted
    elif wrap_scalar and isinstance(value, Real) and not isinstance(value, bool):
        array = (
            np.array([value], copy=True, dtype=dtype)
            if copy
            else np.asarray([value], dtype=dtype)
        )
    elif isinstance(value, (str, bytes)) or isinstance(value, Mapping):
        raise InterfaceValidationError(
            f"{field_name} must be array-like, got {type(value).__name__}."
        )
    elif isinstance(value, Sequence):
        array = (
            np.array(value, copy=True, dtype=dtype)
            if copy
            else np.asarray(value, dtype=dtype)
        )
    else:
        raise InterfaceValidationError(
            f"{field_name} must be array-like, got {type(value).__name__}."
        )

    if dtype is not None and array.dtype != np.dtype(dtype):
        array = array.astype(dtype, copy=False)
    if wrap_scalar and array.ndim == 0:
        array = array.reshape(1)

    if numeric_only:
        if np.issubdtype(array.dtype, np.bool_) and not allow_bool:
            raise InterfaceValidationError(
                f"{field_name} must use a real numeric dtype, got bool."
            )
        if not (
            np.issubdtype(array.dtype, np.integer)
            or np.issubdtype(array.dtype, np.floating)
            or (allow_bool and np.issubdtype(array.dtype, np.bool_))
        ):
            raise InterfaceValidationError(
                f"{field_name} must use a numeric dtype, got {array.dtype}."
            )

    if array.dtype == np.dtype("O"):
        raise InterfaceValidationError(f"{field_name} must not use object dtype.")
    return array


def optional_array_to_list(value: object, *, field_name: str) -> list[Any] | None:
    """Convert optional ndarray/tensor inputs into plain Python lists."""

    converted = optional_array_to_numpy(value, field_name=field_name, copy=False)
    if converted is None:
        return None
    return converted.tolist()


def to_python_value(value: object) -> object:
    """Recursively convert numpy values into plain Python containers."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {key: to_python_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [to_python_value(item) for item in value]
    return value


__all__ = [
    "numpy_ndarray_type",
    "optional_array_to_list",
    "optional_array_to_numpy",
    "to_numpy_array",
    "to_python_value",
    "torch_tensor_type",
]
