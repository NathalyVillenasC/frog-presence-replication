"""Choosing between the GPU and the CPU for array work, without ever installing anything.

``cupy`` is optional. When it is importable and a small computation on the device works, array code
can run on the GPU; otherwise the same code runs with NumPy. Nothing here changes the environment.
"""

import numpy as np


def array_module(prefer_gpu: bool = True):
    """Return ``cupy`` if it can really run on a GPU, otherwise ``numpy``.

    Args:
        prefer_gpu (bool): When False, ``numpy`` is returned without trying the GPU.

    Returns:
        module: An array module with the NumPy interface.
    """
    if prefer_gpu:
        try:
            import cupy

            float(cupy.asarray([1.0, 2.0]).sum())   # forces the device and the runtime to initialise
            return cupy
        except Exception:
            pass
    return np


def to_numpy(array) -> np.ndarray:
    """A NumPy array from either a NumPy or a CuPy array."""
    return array.get() if hasattr(array, "get") else np.asarray(array)


def describe(xp) -> str:
    """One line saying where the arrays live."""
    if xp is np:
        return "CPU (numpy)"
    try:
        return f"GPU (cupy, {xp.cuda.runtime.getDeviceProperties(0)['name'].decode()})"
    except Exception:
        return "GPU (cupy)"
