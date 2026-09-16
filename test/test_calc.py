from pathlib import Path

import numpy as np
from cinrad.utils import vert_integrated_liquid, vert_integrated_liquid_py, echo_top, echo_top_py

GOLDEN = Path(__file__).parent / "golden" / "cython_ref.npz"


def _golden(name):
    with np.load(GOLDEN) as data:
        return data[name]


def _volume_small():
    a = np.arange(0, 27, 1, dtype=np.double).reshape(3, 3, 3)
    b = np.ascontiguousarray(np.broadcast_to(np.arange(0, 3), (3, 3)), dtype=np.double)
    c = np.arange(0, 3, 1, dtype=np.double)
    return a, b, c


def _volume_mid(dist_offset=0.0):
    a = np.arange(0, 64, 1, dtype=np.double).reshape(4, 4, 4)
    b = np.ascontiguousarray(np.broadcast_to(np.arange(0, 4), (4, 4)), dtype=np.double)
    b = b + dist_offset
    c = np.arange(0, 4, 1, dtype=np.double)
    return a, b, c


def _volume_cinrad():
    rng = np.random.default_rng(42)
    elev = np.array([0.5, 1.5, 2.4, 3.4, 4.3, 6.0, 9.9, 14.6, 19.5], dtype=np.double)
    dist = np.zeros((80, 360), dtype=np.double)
    dist[:, :] = np.linspace(0.25, 230, 80)[:, None]
    ref = rng.normal(25, 15, size=(len(elev), 80, 360)).astype(np.double)
    ref[ref < 0] = 0
    return ref, dist, elev


def _volume_below():
    ref = np.full((5, 10, 12), 5.0, dtype=np.double)
    dist = np.broadcast_to(np.arange(1, 11, dtype=np.double)[:, None], (10, 12)).copy()
    elev = np.linspace(0.5, 6, 5, dtype=np.double)
    return ref, dist, elev


def _volume_above():
    ref, dist, elev = _volume_below()
    return np.full_like(ref, 40.0), dist, elev


def _volume_single():
    ref = np.full((1, 8, 8), 25.0, dtype=np.double)
    dist = np.broadcast_to(np.arange(1, 9, dtype=np.double)[:, None], (8, 8)).copy()
    elev = np.array([0.5], dtype=np.double)
    return ref, dist, elev


def _volume_nan():
    ref, dist, elev = _volume_mid(1.0)
    ref = ref.copy()
    ref[1, 2, 2] = np.nan
    ref[0, 0, :] = np.nan
    return ref, dist, elev


def _volume_mix():
    ref = np.zeros((6, 16, 20), dtype=np.double)
    ref[2:5, 4:12, 3:15] = 30
    ref[3, 8, 8] = 50
    ref[0, :, :] = 10
    dist = np.broadcast_to(np.linspace(1, 100, 16)[:, None], (16, 20)).copy().astype(np.double)
    elev = np.array([0.5, 1.5, 2.4, 3.4, 6.0, 9.9], dtype=np.double)
    return ref, dist, elev


def _volume_zeros():
    ref = np.zeros((3, 2, 2), dtype=np.double)
    dist = np.ones((2, 2), dtype=np.double)
    elev = np.array([0.5, 1.5, 2.4], dtype=np.double)
    return ref, dist, elev


CASES = {
    "small": _volume_small,
    "mid": _volume_mid,
    "mid_d1": lambda: _volume_mid(1.0),
    "cinrad": _volume_cinrad,
    "below": _volume_below,
    "above": _volume_above,
    "single": _volume_single,
    "nan": _volume_nan,
    "mix": _volume_mix,
    "zeros": _volume_zeros,
}


def test_vil():
    a, b, c = _volume_small()
    vil = vert_integrated_liquid(a, b, c)
    true_vil = np.array(
        [
            [0.0, 0.00141174, 0.00322052],
            [0.0, 0.00209499, 0.0047792],
            [0.0, 0.00310893, 0.00709224],
        ]
    )
    assert np.allclose(vil, true_vil)


def test_vil_cy2py():
    a, b, c = _volume_mid()
    vil = vert_integrated_liquid(a, b, c)
    vil2 = vert_integrated_liquid_py(a, b, c)
    assert np.allclose(vil, vil2)


def test_et():
    a, b, c = _volume_small()
    et = echo_top(a, b, c, 0)
    true_et = np.array(
        [
            [0.0, 0.03495832023191273, 0.070034287522649],
            [0.0, 0.03495832023191273, 0.070034287522649],
            [0.0, 0.03495832023191273, 0.070034287522649],
        ]
    )
    assert np.array_equal(et, true_et)


def test_et_cy2py():
    a, b, c = _volume_mid()
    et = echo_top(a, b, c, 0)
    et2 = echo_top_py(a, b, c, 0)
    assert np.array_equal(et, et2)


def test_vil_matches_cython():
    for name, factory in CASES.items():
        ref, dist, elev = factory()
        out = vert_integrated_liquid(ref, dist, elev)
        assert np.allclose(out, _golden("vil_" + name), equal_nan=True, rtol=1e-12, atol=1e-15)


def test_vild_matches_cython():
    for name, factory in CASES.items():
        if name == "mid":
            continue
        ref, dist, elev = factory()
        out = vert_integrated_liquid(ref, dist, elev, density=True)
        assert np.allclose(out, _golden("vild_" + name), equal_nan=True, rtol=1e-12, atol=1e-15)


def test_et_matches_cython():
    for name, factory in CASES.items():
        ref, dist, elev = factory()
        assert np.array_equal(echo_top(ref, dist, elev, 0.0), _golden("et_" + name))
        assert np.array_equal(echo_top(ref, dist, elev, 137.0), _golden("et_h_" + name))


def test_no_cython_extension():
    import importlib.util

    assert importlib.util.find_spec("cinrad._utils") is None
    assert importlib.util.find_spec("cinrad.correct._unwrap_2d") is None
