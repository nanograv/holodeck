"""
"""

import pytest  # noqa
import numpy as np

# import holodeck as holo
from holodeck import utils


def test_minmax():
    data = np.random.uniform(-1000, 1000, 100)
    truth = [data.min(), data.max()]
    assert np.all(truth == utils.minmax(data))

    return


class Test__nyquist_freqs:

    def test_basic(self):
        dur = 10.0
        cad = 0.1
        truth = np.arange(1, 101) * 0.1
        test = utils.nyquist_freqs(dur, cad)
        assert np.allclose(truth, test)
        return

    def test_trim(self):
        dur = 10.0
        cad = 0.1
        # Remove the first (0.1) and last (10.0) elements with `trim`
        trim = [0.15, 9.95]
        truth = np.arange(1, 101)[1:-1] * 0.1
        test = utils.nyquist_freqs(dur, cad, trim=trim)
        assert np.allclose(truth, test)
        return

    def test_trim_raises(self):
        dur = 10.0
        cad = 0.1

        bad_trims = [0.0, 10.0, [-2.0]]
        for trim in bad_trims:
            with pytest.raises(ValueError):
                utils.nyquist_freqs(dur, cad, trim=trim)

        return


def test__a_to_z__z_to_a():
    vals = [
        [1.0, 0.0],
        [0.5, 1.0],
        [0.2, 4.0],
        [0.58723, 0.70291027]
    ]

    for aa, zz in vals:
        assert np.isclose(utils.a_to_z(aa), zz)
        assert np.isclose(utils.z_to_a(zz), aa)

    aa, zz = np.array(vals).T
    assert np.allclose(utils.a_to_z(aa), zz)
    assert np.allclose(utils.z_to_a(zz), aa)

    test = np.random.uniform(0.0, 1.0, 20)
    check = utils.z_to_a(utils.a_to_z(test))
    assert np.allclose(test, check)
    return
