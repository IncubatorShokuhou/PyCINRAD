from pathlib import Path

import numpy as np
from xarray import Dataset, DataArray

from cinrad.correct.dealias import dealias, dealias_unwrap_2d

GOLDEN_PATH = Path(__file__).parent / "golden" / "cython_ref.npz"


def _golden(name: str) -> np.ndarray:
    with np.load(GOLDEN_PATH) as data:
        return data[name].copy()


def _simple_field():
    nyq = 27.0
    az, rngb = np.meshgrid(
        np.linspace(0, 2 * np.pi, 36, endpoint=False),
        np.linspace(1, 100, 40),
        indexing="ij",
    )
    true_v = 10 * np.sin(az) * np.exp(-((rngb - 50) / 40) ** 2)
    wrapped = np.angle(np.exp(1j * true_v * np.pi / nyq)) * nyq / np.pi
    return wrapped, nyq


def _folded_field():
    nyq = 27.0
    az, rngb = np.meshgrid(
        np.linspace(0, 2 * np.pi, 36, endpoint=False),
        np.linspace(1, 100, 40),
        indexing="ij",
    )
    true_v = 40 * np.sin(az) * (rngb / 100)
    wrapped = ((true_v + nyq) % (2 * nyq)) - nyq
    return wrapped, nyq


def _nanmask_field():
    wrapped, nyq = _folded_field()
    wrapped = wrapped.copy()
    wrapped[::5, ::7] = np.nan
    wrapped[10:12, :] = np.nan
    return wrapped, nyq


def _cinrad_field():
    nyq = 27.0
    az, rngb = np.meshgrid(
        np.linspace(0, 2 * np.pi, 360, endpoint=False),
        np.linspace(0.25, 230, 230),
        indexing="ij",
    )
    true_v = 35 * np.sin(2 * az) * np.tanh(rngb / 40)
    wrapped = ((true_v + nyq) % (2 * nyq)) - nyq
    return wrapped, nyq


def _tiny_field():
    return np.array([[1.0, 2.0], [3.0, -20.0]], dtype=np.float64), 27.0


CASES = {
    "simple": _simple_field,
    "folded": _folded_field,
    "nanmask": _nanmask_field,
    "cinrad": _cinrad_field,
    "tiny": _tiny_field,
}


def test_dealias_matches_cython_goldens():
    for name, factory in CASES.items():
        vdata, nyq = factory()
        out = dealias_unwrap_2d(vdata, nyq)
        golden = _golden(f"dealias_{name}")
        assert np.allclose(out, golden, equal_nan=True, rtol=0, atol=0)


def test_dealias_dataset_masks_nan():
    vdata, nyq = _nanmask_field()
    ds = Dataset(
        {"VEL": DataArray(vdata, dims=["azimuth", "distance"])},
        attrs={"nyquist_vel": nyq},
    )
    out = dealias(ds)
    result = np.ma.asanyarray(out["VEL"].values)
    assert np.array_equal(np.ma.getmaskarray(result), np.isnan(vdata))
    finite = ~np.isnan(vdata)
    assert np.allclose(np.asarray(result)[finite], dealias_unwrap_2d(vdata, nyq)[finite])


def test_dealias_all_nan_is_masked_zeros():
    vdata = np.full((10, 12), np.nan)
    out = dealias_unwrap_2d(vdata, 27.0)
    assert np.all(out == 0)


def test_dealias_empty():
    out = dealias_unwrap_2d(np.zeros((0, 0)), 27.0)
    assert out.shape == (0, 0)
