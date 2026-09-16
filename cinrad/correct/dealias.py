# -*- coding: utf-8 -*-
# Author: PyCINRAD Developers

import inspect

import numpy as np
from xarray import Dataset
from skimage.restoration import unwrap_phase

_UNWRAP_PARAMS = inspect.signature(unwrap_phase).parameters


def _unwrap_2d(image: np.ma.MaskedArray) -> np.ndarray:
    kwargs = {"wrap_around": (True, False)}
    if "rng" in _UNWRAP_PARAMS:
        kwargs["rng"] = 0
    elif "seed" in _UNWRAP_PARAMS:
        kwargs["seed"] = 0
    return np.asarray(unwrap_phase(image, **kwargs))


def dealias_unwrap_2d(vdata: np.ndarray, nyquist_vel: float) -> np.ndarray:
    """Dealias using 2D phase unwrapping (sweep-by-sweep)."""
    scaled_sweep = np.asarray(vdata, dtype=np.float64) * np.pi / nyquist_vel
    sweep_mask = np.isnan(vdata)
    if scaled_sweep.size == 0:
        return np.zeros_like(scaled_sweep, dtype=np.float64)
    if np.all(sweep_mask):
        return np.zeros_like(scaled_sweep, dtype=np.float64)
    scaled_sweep = np.array(scaled_sweep, copy=True, order="C")
    scaled_sweep[sweep_mask] = 0
    wrapped = np.ma.array(scaled_sweep, mask=sweep_mask)
    unwrapped = _unwrap_2d(wrapped)
    return unwrapped * nyquist_vel / np.pi


def dealias(v_data: Dataset) -> Dataset:
    v_field = v_data["VEL"].data
    nyq = v_data.attrs.get("nyquist_vel")
    out_data = dealias_unwrap_2d(v_field, nyq)
    out_masked = np.ma.array(out_data, mask=np.isnan(v_field))
    v_ret = v_data.copy()
    v_ret["VEL"] = (v_data["VEL"].dims, out_masked)
    return v_ret
