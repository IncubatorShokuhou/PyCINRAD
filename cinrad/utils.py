# -*- coding: utf-8 -*-
# Author: PyCINRAD Developers

import os
from typing import Union

import numpy as np

from cinrad._typing import Array_T, Number_T

MODULE_DIR = os.path.dirname(__file__)

VIL_CONST = 3.44e-6
DEG2RAD = 3.141592653589793 / 180
RM = 8500


def r2z(r: np.ndarray) -> np.ndarray:
    return 10 ** (r / 10)


def _take_along_elev(arr: np.ndarray, index: np.ndarray) -> np.ndarray:
    xshape, yshape = index.shape
    return arr[index, np.arange(xshape)[:, None], np.arange(yshape)]


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
    vil = np.zeros((xshape, yshape), dtype=np.float64)
    if zshape == 0 or xshape == 0 or yshape == 0:
        return vil

    z = r2z(ref)
    dist = distance * 1000.0
    hi = dist * np.sin(beam_width * DEG2RAD / 2.0)

    above = ref > threshold
    valid = np.any(above, axis=0)
    if not np.any(valid):
        return vil

    layer = np.arange(zshape, dtype=np.intp)[:, None, None]
    pos_s = np.where(above, layer, zshape).min(axis=0)
    pos_e = np.where(above, layer, -1).max(axis=0)

    if zshape >= 2:
        dsin = np.diff(np.sin(elev * DEG2RAD))
        ht = dist[None, ...] * dsin[:, None, None]
        factor = ((z[:-1] + z[1:]) / 2.0) ** (4.0 / 7.0)
        contrib = VIL_CONST * factor * ht
        include = np.arange(zshape - 1, dtype=np.intp)[:, None, None] < pos_e[None, ...]
        m1 = np.where(include, contrib, 0.0).sum(axis=0)
    else:
        m1 = np.zeros((xshape, yshape), dtype=np.float64)

    pos_s_clip = np.clip(pos_s, 0, zshape - 1)
    pos_e_clip = np.clip(pos_e, 0, zshape - 1)
    z_s = _take_along_elev(z, pos_s_clip)
    z_e = _take_along_elev(z, pos_e_clip)

    if not density:
        mb = VIL_CONST * z_s ** (4.0 / 7.0) * hi
        mt = VIL_CONST * z_e ** (4.0 / 7.0) * hi
        return np.where(valid, m1 + mb + mt, 0.0)

    same = pos_s == pos_e
    elev_s = elev[pos_s_clip]
    elev_e = elev[pos_e_clip]
    h_lower = distance * np.sin(elev_s * DEG2RAD) + distance ** 2 / (2.0 * RM)
    h_higher = distance * np.sin(elev_e * DEG2RAD) + distance ** 2 / (2.0 * RM)
    with np.errstate(divide="ignore", invalid="ignore"):
        vild = m1 / (h_higher - h_lower)
    return np.where(valid, np.where(same, 0.0, vild), 0.0)


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
    et = np.zeros((xshape, yshape), dtype=np.float64)
    if zshape == 0 or xshape == 0 or yshape == 0:
        return et

    hght = (
        distance[None, ...] * np.sin(elev[:, None, None] * DEG2RAD)
        + distance[None, ...] ** 2 / (2.0 * RM)
        + radarheight
    )
    above = ref >= threshold
    valid = np.any(above, axis=0)
    layer = np.arange(zshape, dtype=np.intp)[:, None, None]
    pos = np.where(above, layer, -1).max(axis=0)
    highest = ref[-1] >= threshold
    pos0 = pos == 0

    if zshape == 1:
        interp = hght[0]
    else:
        pos_clip = np.clip(pos, 0, zshape - 2)
        z1 = _take_along_elev(ref, pos_clip)
        z2 = _take_along_elev(ref, pos_clip + 1)
        h1 = _take_along_elev(hght, pos_clip)
        h2 = _take_along_elev(hght, pos_clip + 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            w1 = (z1 - threshold) / (z1 - z2)
            interp = w1 * h2 + (1.0 - w1) * h1

    return np.where(
        ~valid, 0.0, np.where(highest, hght[-1], np.where(pos0, hght[0], interp))
    )


# Backward-compatible aliases (previously the pure-Python fallbacks).
vert_integrated_liquid_py = vert_integrated_liquid
echo_top_py = echo_top
