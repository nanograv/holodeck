"""Functions and relations from Runnoe et al. 2012.

Runnoe+2012 [1201.5155] - Updating quasar bolometric luminosity corrections
https://ui.adsabs.harvard.edu/abs/2012MNRAS.422..478R/abstract

"""

import numpy as np

from holodeck import utils

__all__ = ["Runnoe2012"]


class Runnoe2012:

    def __init__(self):
        return

    @classmethod
    def lum5100_from_lbol(cls, lbol, scatter=False):
        """
        Runnoe+2012 [1201.5155]
        Eq.11 & 13

        log(Liso) = (4.89 ± 1.66) + (0.91 ± 0.04) log(5100, L5100),   [Eq.11]

        """
        alpha = (4.89, 1.66)
        beta = (0.91, 0.04)
        if not scatter:
            alpha = alpha[0]
            beta = beta[0]

        lam_lum_lam = _lbol_to_lband__pow_law(lbol, alpha, beta, lum0=1.0, fiso=0.75)
        return lam_lum_lam

    @classmethod
    def lum3000_from_lbol(cls, lbol):
        """
        Runnoe+2012 [1201.5155]
        Eq.10 & 13

        log(Liso) = (1.85 ± 1.27) + (0.98 ± 0.03) log(3000L3000).

        """
        lam_lum_lam = _lbol_to_lband__pow_law(lbol, 1.85, 0.98, lum0=1.0, fiso=0.75)
        return lam_lum_lam

    @classmethod
    def lum1450_from_lbol(cls, lbol):
        """
        Runnoe+2012 [1201.5155]
        Eq.9 & 13

        log(Liso) = (4.74 ± 1.00) + (0.91 ± 0.02) log(1450L1450).
        """
        lam_lum_lam = _lbol_to_lband__pow_law(lbol, 4.74, 0.91, lum0=1.0, fiso=0.75)
        return lam_lum_lam

    @classmethod
    def lbol_from_5100ang(cls, lam_lum_lam, scatter=False):
        """
        Runnoe+2012 [1201.5155]
        Eq.11 & 13

        log(Liso) = (4.89 ± 1.66) + (0.91 ± 0.04) log(5100, L5100),   [Eq.11]

        """
        alpha = (4.89, 1.66)
        beta = (0.91, 0.04)
        if not scatter:
            alpha = alpha[0]
            beta = beta[0]

        lbol = _lband_to_lbol__pow_law(lam_lum_lam, alpha, beta, lum0=1.0, fiso=0.75)
        return lbol

    @classmethod
    def lbol_from_3000ang(cls, lam_lum_lam):
        """
        Runnoe+2012 [1201.5155]
        Eq.10 & 13

        log(Liso) = (1.85 ± 1.27) + (0.98 ± 0.03) log(3000L3000).
        """
        lbol = _lband_to_lbol__pow_law(lam_lum_lam, 1.85, 0.98, lum0=1.0, fiso=0.75)
        return lbol

    @classmethod
    def lbol_from_1450ang(cls, lam_lum_lam):
        """
        Runnoe+2012 [1201.5155]
        Eq.9 & 13

        log(Liso) = (4.74 ± 1.00) + (0.91 ± 0.02) log(1450L1450).
        """
        lbol = _lband_to_lbol__pow_law(lam_lum_lam, 4.74, 0.91, lum0=1.0, fiso=0.75)
        return lbol

    @classmethod
    def lbol_from_2to10kev_all(cls, lam_lum_lam):
        """
        Bolometric correction from X-Ray (2–10keV) full sample of quasars

        Runnoe+2012 [1201.5155]
        Table 5

        log (Liso) = (25.14 ± 01.93) + (0.47 ± 0.043) log (L2-10 keV),
        """
        lbol = _lband_to_lbol__pow_law(lam_lum_lam, 25.14, 0.47, lum0=1.0, fiso=0.75)
        return lbol

    @classmethod
    def lbol_from_2to10kev_RL(cls, lam_lum_lam):
        """
        Bolometric correction from X-Ray (2–10keV) radio-loud quasars

        Runnoe+2012 [1201.5155]
        Eq.14

        log(Liso,RL) = (23.04 ± 03.60) + (0.52 ± 0.080) log(L2-10 keV),
        """
        lbol = _lband_to_lbol__pow_law(lam_lum_lam, 23.04, 0.52, lum0=1.0, fiso=0.75)
        return lbol

    @classmethod
    def lbol_from_2to10kev_RQ(cls, lam_lum_lam):
        """
        Bolometric correction from X-Ray (2–10keV) radio-quiet quasars

        Runnoe+2012 [1201.5155]
        Eq.15

        log(Liso,RQ) = (33.06 ± 03.17) + (0.29 ± 0.072) log(L2-10 keV).
        """
        lbol = _lband_to_lbol__pow_law(lam_lum_lam, 33.06, 0.29, lum0=1.0, fiso=0.75)
        return lbol

    @classmethod
    def iband_from_mass_fedd(cls, mass, fedd, eps=0.1, magnitude=False):
        lbol = utils.eddington_luminosity(mass, eps) * fedd
        l5100 = cls.lum5100_from_lbol_runnoe2012(lbol)
        L_lambda = l5100 * (5100e-8)
        if magnitude:
            imag = zastro.obs.lum_to_abs_mag('i', L_lambda, type='l')
        return imag




def _dist_pars(arg, num):
    """If `arg` is tuple (2,), draw `num` points from normal distribution with those parameters.
    """
    if isinstance(arg, tuple):
        if len(arg) != 2:
            raise ValueError("`arg` must be a tuple of (mean, std)!")
        arg = np.random.normal(*arg, size=num)

    return arg


def _lband_to_lbol__pow_law(lam_lum_lam, alpha, beta, lum0=1.0, fiso=1.0):
    """
    log(L_iso) = alpha + beta * log10(lambda * L_lambda / lum0)
    L_bol = fiso * L_iso
    """
    liso = alpha + beta*np.log10(lam_lum_lam / lum0)
    lbol = fiso * (10**liso)
    return lbol


def _lbol_to_lband__pow_law(lbol, alpha, beta, lum0=1.0, fiso=1.0):
    """Returns lambda*L_lambda

    log(L_iso) = alpha + beta * log10(lambda * L_lambda / lum0)
    L_iso = L_bol / fiso
    """
    liso_log = np.log10(lbol/fiso)
    num = np.size(lbol)

    alpha = _dist_pars(alpha, num)
    beta = _dist_pars(beta, num)

    lam_lum_lam = lum0 * np.power(10, (liso_log - alpha)/beta)
    return lam_lum_lam

