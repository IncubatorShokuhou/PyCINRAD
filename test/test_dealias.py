from pathlib import Path

import numpy as np
from xarray import Dataset, DataArray

from cinrad.correct.dealias import dealias, dealias_unwrap_2d
from align_cases import vel_nanmask

GOLDEN = Path(__file__).parent / "golden" / "cython_ref.npz"


def _golden(name):
    with np.load(GOLDEN) as data:
        return data[name]


def test_exact_nyquist_unwrap_is_deterministic():
    # 原版 Cython 在 v=±nyq 上多次运行可差 2*nyq；这边只要求 scikit-image 确定。
    nyq = 27.0
    v = np.zeros((48, 64), dtype=np.float64)
    v[:, :] = nyq
    v[::2] = -nyq
    v[10:16, 20:30] = 0.0
    v[0, 0] = np.nan
    first = dealias_unwrap_2d(v, nyq)
    for _ in range(4):
        assert np.array_equal(dealias_unwrap_2d(v, nyq), first, equal_nan=True)


def test_dealias_dataset_nan_mask():
    vdata, nyq = vel_nanmask()
    ds = Dataset(
        {"VEL": DataArray(vdata, dims=["azimuth", "distance"])},
        attrs={"nyquist_vel": nyq},
    )
    out = np.asarray(dealias(ds)["VEL"].values)
    assert np.array_equal(np.isnan(out), np.isnan(vdata))
    finite = ~np.isnan(vdata)
    assert np.array_equal(out[finite], dealias_unwrap_2d(vdata, nyq)[finite])
    assert np.array_equal(dealias_unwrap_2d(vdata, nyq), _golden("dealias_nanmask"))
