"""
"""

import pytest
import numpy as np

# import holodeck as holo
from holodeck import utils
from holodeck.constants import YR


def test_minmax():
    data = np.random.uniform(-1000, 1000, 100)
    truth = [data.min(), data.max()]
    assert np.all(truth == utils.minmax(data))

    return

'''
class Test__nyquist_freqs:

    def test_basic(self):
        dur = 10.0
        cad = 0.1
        truth = np.arange(1, 51) * 0.1
        test = utils.nyquist_freqs(dur, cad)
        assert np.allclose(truth, test)
        return

    def test_trim(self):
        dur = 10.0
        cad = 0.1
        # Remove the first (0.1) and last (10.0) elements with `trim`
        trim = [0.15, 4.95]
        truth = np.arange(1, 51)[1:-1] * 0.1
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
'''

class Test_GW_Methods:

    def test_with_fixed_values(self):
        """

        All of these values were calculated in the `utils.ipynb` on 2022-08-08, by Luke Zoltan Kelley

        """
        freq = 1.0 / YR
        # time = GYR

        m1 = [1.57413313e+41, 8.14164709e+41, 3.60895311e+41, 9.19991375e+39, 1.06186683e+41]
        m2 = [1.62059681e+42, 9.26879287e+39, 3.91354504e+40, 3.44058558e+40, 3.10034300e+40]
        aa = [5.24456784e+17, 2.78829325e+16, 7.93618648e+15, 3.68590010e+14, 2.72097813e+16]
        ee = [7.25868969e-01, 3.75506309e-01, 3.42479721e-01, 5.43473856e-01, 9.99000000e-01]
        dc = [6.63399191e+25, 1.76236980e+24, 5.54714846e+22, 3.78302329e+23, 1.50096058e+20]

        mc = [3.92683830e+41, 5.54007398e+40, 9.32292666e+40, 1.48710242e+40, 4.81973673e+40]
        hs = [3.72246602e-15, 5.35750864e-15, 4.05245287e-13, 2.78780956e-15, 4.98735430e-11]
        gwlum = [9.75931974e+47, 1.42668479e+45, 8.08693051e+45, 1.77996894e+43, 8.96782668e+44]
        dedt = [-8.50812477e-18, -1.48341135e-15, -1.73541089e-13, -2.71156486e-10, -1.60176941e-09]
        dade = [2.16041096e+18, 7.42617422e+16, 2.15759501e+16, 1.05091558e+15, 2.72192326e+19]
        dadt = [-1.83810460e+01, -1.10160711e+02, -3.74431389e+03, -2.84962577e+05, -4.35989342e+10]
        dfdt = [2.92125254e-18, 7.34264884e-21, 1.50683392e-20, 2.20635454e-21, 2.94579604e-11]
        tau = [4.03507842e+11, 1.05535297e+13, 4.43271589e+12, 9.44834061e+13, 1.33112288e+13]

        check_mc = utils.chirp_mass(m1, m2)
        check_hs = utils.gw_strain_source(mc, dc, freq)
        check_gwlum = utils.gw_lum_circ(mc, freq)
        check_dedt = utils.gw_dedt(m1, m2, aa, ee)
        check_dade = utils.gw_dade(aa, ee)
        check_dadt = utils.gw_hardening_rate_dadt(m1, m2, aa, ee)
        check_dfdt, _ = utils.gw_hardening_rate_dfdt(m1, m2, freq, ee)
        check_tau = utils.gw_hardening_timescale_freq(mc, freq)

        keys = ['mc', 'hs', 'gwlum', 'dedt', 'dade', 'dadt', 'dfdt', 'tau']
        test = [mc, hs, gwlum, dedt, dade, dadt, dfdt, tau]
        check = [check_mc, check_hs, check_gwlum, check_dedt, check_dade, check_dadt, check_dfdt, check_tau]

        for kk, tt, cc in zip(keys, test, check):
            vv = np.isclose(tt, cc)
            print(f"{kk} :: {vv}")
            assert np.all(vv), f"{kk} did not match cached values!!"

        return

    # def test_hardening_dadt(self):
