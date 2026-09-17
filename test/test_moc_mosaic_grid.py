from io import BytesIO

import numpy as np

from cinrad.io._dtype import mocm_dtype
from cinrad.io.level3 import MocMosaic


def _moc_mosaic_file(nx, ny, edge_s, edge_w, edge_n, edge_e, dx, dy):
    header = np.zeros(1, dtype=mocm_dtype)
    header["label"] = np.frombuffer(b"MOC\x00", "S1")
    header["varname"] = b"CREF"
    header["year"] = 2024
    header["month"] = 1
    header["day"] = 1
    header["scale"] = 10
    header["nx"] = nx
    header["ny"] = ny
    header["dx"] = int(round(dx * 1000))
    header["dy"] = int(round(dy * 1000))
    header["edge_s"] = int(round(edge_s * 1000))
    header["edge_w"] = int(round(edge_w * 1000))
    header["edge_n"] = int(round(edge_n * 1000))
    header["edge_e"] = int(round(edge_e * 1000))
    body = np.zeros(ny * nx, dtype="i2").tobytes()
    return BytesIO(header.tobytes() + body)


def test_moc_geo_axis_v3_national_grid():
    from cinrad.io.level3 import _moc_geo_axis

    lon = _moc_geo_axis(73.0, 6200, 0.01)
    lat = _moc_geo_axis(12.2, 4200, 0.01)

    assert lon.shape == (6200,)
    assert lat.shape == (4200,)
    assert lon.shape != (6201,)
    assert lat.shape != (4201,)

    assert lon[0] == 73.0
    assert lat[0] == 12.2
    assert lon[0] != 73.01
    assert lat[0] != 12.21

    assert lon[-1] == 134.99
    np.testing.assert_allclose(lat[-1], 54.19, rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.diff(lon), 0.01)
    np.testing.assert_allclose(np.diff(lat), 0.01)

    assert not np.array_equal(lon, np.linspace(73.0, 135.0, 6200))
    assert not np.array_equal(lat, np.linspace(12.2, 54.2, 4200))


def test_moc_mosaic_uses_half_open_header_grid():
    nx, ny = 4, 3
    dx = dy = 0.01
    edge_w, edge_s = 73.0, 12.2
    edge_e, edge_n = edge_w + nx * dx, edge_s + ny * dy
    moc = MocMosaic(
        _moc_mosaic_file(nx, ny, edge_s, edge_w, edge_n, edge_e, dx, dy)
    )

    assert moc.lon.shape == (nx,)
    assert moc.lat.shape == (ny,)
    assert moc.data.shape == (ny, nx)
    assert moc.lon[0] == 73.0
    assert moc.lat[0] == 12.2
    np.testing.assert_array_equal(moc.lon, edge_w + np.arange(nx) * dx)
    np.testing.assert_array_equal(moc.lat, edge_s + np.arange(ny) * dy)
    assert moc.lon[-1] != edge_e
    assert moc.lat[-1] != edge_n
