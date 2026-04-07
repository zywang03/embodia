"""Optional array/tensor helpers without hard dependencies.

embodia's public core intentionally stays free of heavy numeric dependencies.
These helpers let boundary coercion recognize ``numpy.ndarray`` or
``torch.Tensor`` values when those packages already exist in the user's
environment, without making them required install-time dependencies.
"""

from __future__ import annotations

import importlib
from typing import Any

from .errors import InterfaceValidationError

_UNSET = object()
_NUMPY_NDARRAY_TYPE: object = _UNSET
_TORCH_TENSOR_TYPE: object = _UNSET


def numpy_ndarray_type() -> type[Any] | None:
    """Return ``numpy.ndarray`` when numpy is importable, else ``None``."""

    global _NUMPY_NDARRAY_TYPE
    if _NUMPY_NDARRAY_TYPE is _UNSET:
        try:
            module = importlib.import_module("numpy")
        except ImportError:
            _NUMPY_NDARRAY_TYPE = None
        else:
            _NUMPY_NDARRAY_TYPE = module.ndarray
    return _NUMPY_NDARRAY_TYPE if isinstance(_NUMPY_NDARRAY_TYPE, type) else None


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


def _ensure_list_result(
    result: object,
    *,
    field_name: str,
    source_name: str,
) -> list[Any]:
    """Ensure ``tolist()`` produced a list-like action value."""

    if not isinstance(result, list):
        raise InterfaceValidationError(
            f"{field_name} converted from {source_name} must produce a list, "
            f"got {type(result).__name__}."
        )
    return result


def optional_array_to_list(value: object, *, field_name: str) -> list[Any] | None:
    """Convert optional numpy/torch values into plain Python lists.

    Returns ``None`` when ``value`` is not a recognized optional array/tensor
    type. Torch tensors are detached first; non-CPU tensors are moved to CPU
    exactly once before conversion into embodia's small Python-side action
    structure.
    """

    numpy_type = numpy_ndarray_type()
    if numpy_type is not None and isinstance(value, numpy_type):
        return _ensure_list_result(
            value.tolist(),
            field_name=field_name,
            source_name="numpy.ndarray",
        )

    torch_type = torch_tensor_type()
    if torch_type is not None and isinstance(value, torch_type):
        tensor = value.detach() if callable(getattr(value, "detach", None)) else value
        device = getattr(getattr(tensor, "device", None), "type", None)
        if device not in (None, "cpu"):
            cpu = getattr(tensor, "cpu", None)
            if not callable(cpu):
                raise InterfaceValidationError(
                    f"{field_name} torch tensor is on device {device!r}, but does "
                    "not expose cpu()."
                )
            tensor = cpu()

        return _ensure_list_result(
            tensor.tolist(),
            field_name=field_name,
            source_name="torch.Tensor",
        )

    return None


__all__ = ["numpy_ndarray_type", "optional_array_to_list", "torch_tensor_type"]
