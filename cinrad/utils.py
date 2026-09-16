# -*- coding: utf-8 -*-
# Author: PyCINRAD Developers

import os
from typing import Union

import numpy as np

from cinrad._typing import Array_T, Number_T

MODULE_DIR = os.path.dirname(__file__)

VIL_CONST = 3.44e-6
RM = 8500


def r2z(r: np.ndarray) -> np.ndarray:
    return 10 ** (r / 10)


def vert_integrated_liquid(
    ref: np.ndarray,
    distance: np.ndarray,
    elev: Array_T,
    beam_width: float = 0.99,
    threshold: Union[float, int] = 18.0,
    density: bool = False,
) -> np.ndarray:
    r"""
    Calculate vertically integrated liquid (VIL) in one full scan

    Parameters
    ----------
    ref: numpy.ndarray dim=3 (elevation angle, distance, azimuth)
        reflectivity data
    distance: numpy.ndarray dim=2 (distance, azimuth)
        distance from radar site
    elev: numpy.ndarray or list dim=1
        elevation angles in degree
    threshold: float
        minimum reflectivity value to take into calculation

    Returns
    -------
    data: numpy.ndarray
        vertically integrated liquid data
    """
    ref = np.asarray(ref, dtype=np.float64)
    distance = np.asarray(distance, dtype=np.float64)
    elev = np.asarray(elev, dtype=np.float64)
    zshape, xshape, yshape = ref.shape
    z = r2z(ref)
    dist = distance * 1000
    hi = dist * np.sin(np.deg2rad(beam_width) / 2)
    above = ref > threshold
    valid = above.any(axis=0)
    idx = np.arange(zshape)[:, None, None]
    pos_s = np.where(above, idx, zshape).min(axis=0)
    pos_e = np.where(above, idx, -1).max(axis=0)
    dsin = np.diff(np.sin(np.deg2rad(elev)))
    factor = ((z[:-1] + z[1:]) / 2) ** (4 / 7)
    contrib = VIL_CONST * factor * dist * dsin[:, None, None]
    m1 = np.where(np.arange(zshape - 1)[:, None, None] < pos_e, contrib, 0).sum(axis=0)
    i = np.arange(xshape)[:, None]
    j = np.arange(yshape)
    ps = np.clip(pos_s, 0, zshape - 1)
    pe = np.clip(pos_e, 0, zshape - 1)
    if not density:
        mb = VIL_CONST * z[ps, i, j] ** (4 / 7) * hi
        mt = VIL_CONST * z[pe, i, j] ** (4 / 7) * hi
        return np.where(valid, m1 + mb + mt, 0)
    h_lower = distance * np.sin(np.deg2rad(elev[ps])) + distance ** 2 / (2 * RM)
    h_higher = distance * np.sin(np.deg2rad(elev[pe])) + distance ** 2 / (2 * RM)
    vil = np.where(pos_s == pos_e, 0, m1 / (h_higher - h_lower))
    return np.where(valid, vil, 0)


def echo_top(
    ref: np.ndarray,
    distance: np.ndarray,
    elev: Array_T,
    radarheight: Number_T,
    threshold: Number_T = 18.0,
) -> np.ndarray:
    r"""
    Calculate height of echo tops (ET) in one full scan

    Parameters
    ----------
    ref: numpy.ndarray dim=3 (elevation angle, distance, azimuth)
        reflectivity data
    distance: numpy.ndarray dim=2 (distance, azimuth)
        distance from radar site
    elev: numpy.ndarray or list dim=1
        elevation angles in degree
    radarheight: int or float
        height of radar
    threshold: float
        minimum value of reflectivity to be taken into calculation

    Returns
    -------
    data: numpy.ndarray
        echo tops data
    """
    ref = np.asarray(ref, dtype=np.float64)
    distance = np.asarray(distance, dtype=np.float64)
    elev = np.asarray(elev, dtype=np.float64)
    zshape, xshape, yshape = ref.shape
    hght = (
        distance * np.sin(np.deg2rad(elev)[:, None, None])
        + distance ** 2 / (2 * RM)
        + radarheight
    )
    above = ref >= threshold
    idx = np.arange(zshape)[:, None, None]
    pos = np.where(above, idx, -1).max(axis=0)
    i = np.arange(xshape)[:, None]
    j = np.arange(yshape)
    if zshape == 1:
        interp = hght[0]
    else:
        pos_c = np.clip(pos, 0, zshape - 2)
        z1 = ref[pos_c, i, j]
        z2 = ref[pos_c + 1, i, j]
        h1 = hght[pos_c, i, j]
        h2 = hght[pos_c + 1, i, j]
        w1 = (z1 - threshold) / (z1 - z2)
        interp = w1 * h2 + (1 - w1) * h1
    return np.where(
        ~above.any(axis=0),
        0,
        np.where(ref[-1] >= threshold, hght[-1], np.where(pos == 0, hght[0], interp)),
    )


vert_integrated_liquid_py = vert_integrated_liquid
echo_top_py = echo_top
