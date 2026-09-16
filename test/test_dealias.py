from pathlib import Path

import numpy as np
from xarray import Dataset, DataArray

from cinrad.correct.dealias import dealias, dealias_unwrap_2d

GOLDEN = Path(__file__).parent / "golden" / "cython_ref.npz"


def _golden(name):
    with np.load(GOLDEN) as data:
        return data[name]


def _simple():
    nyq = 27.0
    az, rng = np.meshgrid(
        np.linspace(0, 2 * np.pi, 36, endpoint=False),
        np.linspace(1, 100, 40),
        indexing="ij",
    )
    true_v = 10 * np.sin(az) * np.exp(-((rng - 50) / 40) ** 2)
    wrapped = np.angle(np.exp(1j * true_v * np.pi / nyq)) * nyq / np.pi
    return wrapped, nyq


def _folded():
    nyq = 27.0
    az, rng = np.meshgrid(
        np.linspace(0, 2 * np.pi, 36, endpoint=False),
        np.linspace(1, 100, 40),
        indexing="ij",
    )
    true_v = 40 * np.sin(az) * (rng / 100)
    wrapped = ((true_v + nyq) % (2 * nyq)) - nyq
    return wrapped, nyq


def _nanmask():
    wrapped, nyq = _folded()
    wrapped = wrapped.copy()
    wrapped[::5, ::7] = np.nan
    wrapped[10:12, :] = np.nan
    return wrapped, nyq


def _cinrad():
    nyq = 27.0
    az, rng = np.meshgrid(
        np.linspace(0, 2 * np.pi, 360, endpoint=False),
        np.linspace(0.25, 230, 230),
        indexing="ij",
    )
    true_v = 35 * np.sin(2 * az) * np.tanh(rng / 40)
    wrapped = ((true_v + nyq) % (2 * nyq)) - nyq
    return wrapped, nyq


CASES = {
    "simple": _simple,
    "folded": _folded,
    "nanmask": _nanmask,
    "cinrad": _cinrad,
    "tiny": lambda: (np.array([[1.0, 2.0], [3.0, -20.0]], dtype=np.float64), 27.0),
}


def test_dealias_matches_cython():
    for name, factory in CASES.items():
        vdata, nyq = factory()
        assert np.array_equal(dealias_unwrap_2d(vdata, nyq), _golden("dealias_" + name))


def test_dealias():
    vdata, nyq = _nanmask()
    ds = Dataset(
        {"VEL": DataArray(vdata, dims=["azimuth", "distance"])},
        attrs={"nyquist_vel": nyq},
    )
    out = np.asarray(dealias(ds)["VEL"].values)
    assert np.array_equal(np.isnan(out), np.isnan(vdata))
    finite = ~np.isnan(vdata)
    assert np.allclose(out[finite], dealias_unwrap_2d(vdata, nyq)[finite])
