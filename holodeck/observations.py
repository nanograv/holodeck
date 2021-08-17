"""
"""

import numpy as np

from holodeck import log
from holodeck.constants import GYR, SPLC, MSOL, MPC, YR


def load_mcconnell_ma_2013(fname):
    header = []
    data_raw = []
    hcnt = 0
    cnt = 0
    for line in open(fname, 'r').readlines():
        line = line.strip()
        if (len(line) == 0) or (cnt < 2):
            cnt += 1
            continue

        if line.startswith('Col'):
            line = line.split(':')[-1].strip()
            header.append(line)
            cnt += 1
            hcnt += 1
            continue

        line = [ll.strip() for ll in line.split()]
        for ii in range(len(line)):
            try:
                line[ii] = float(line[ii])
            except ValueError:
                pass

        data_raw.append(line)
        cnt += 1

    data = dict()
    data['name'] = np.array([dr[0] for dr in data_raw])
    data['dist'] = np.array([dr[1] for dr in data_raw])
    data['mass'] = np.array([[dr[3], dr[2], dr[4]] for dr in data_raw])
    data['sigma'] = np.array([[dr[7], dr[6], dr[8]] for dr in data_raw])
    data['lumv'] = np.array([[dr[9], dr[10]] for dr in data_raw])
    data['mbulge'] = np.array([dr[13] for dr in data_raw])
    data['rinf'] = np.array([dr[14] for dr in data_raw])
    data['reffv'] = np.array([dr[17] for dr in data_raw])

    return data


class _Galaxy_Blackhole_Relation:
    """
    """

    FITS = {}
    NORM = {}
    _VALID_RELATIONS = ['vdisp', 'mbulge']

    def __init__(self):
        return

    def _values(self, relation):
        if relation not in self._VALID_RELATIONS:
            err = f"`relation` {relation} must be one of '{self._VALID_RELATIONS}'!"
            raise ValueError(err)

        fits = self.FITS[relation]
        alpha = fits['alpha']
        beta = fits['beta']
        eps = fits['eps']

        norm = self.NORM[relation]
        x0 = norm['x']
        y0 = norm['y']

        return x0, y0, alpha, beta, eps

    def _parse_scatter(self, scatter):
        if (scatter is False) or (scatter is None):
            scatter = 0.0
        elif scatter is True:
            scatter = 1.0

        if not np.isscalar(scatter):
            err = f"`scatter` ({scatter}) must be a scalar value!"
            log.error(err)
            raise ValueError(err)
        elif (scatter < 0.0) or not np.isfinite(scatter):
            err = f"`scatter` ({scatter}) must be a positive value!"
            log.error(err)
            raise ValueError(err)

        return scatter

    def mbh_from_mbulge(self, mbulge, scatter=False):
        return self._mbh_from_galaxy(mbulge, 'mbulge', forward=True, scatter=scatter)

    def mbh_from_vdisp(self, vdisp, scatter=False):
        return self._mbh_from_galaxy(vdisp, 'vdisp', forward=True, scatter=scatter)

    def mbulge_from_mbh(self, mbh, scatter=False):
        """

        Arguments
        ---------
        mbh : blackhole mass in [grams]

        Returns
        -------
        mbulge : bulge stellar-mass in units of [grams]

        """
        return self._mbh_relation(mbh, 'mbulge', forward=False, scatter=scatter)

    def vdisp_from_mbh(self, mbh, scatter=False):
        """

        Arguments
        ---------
        mbh : blackhole mass in [grams]

        Returns
        -------
        vdisp : velocity-dispersion (sigma) in units of [cm/s]

        """
        return self._mbh_relation(mbh, 'vdisp', forward=False, scatter=scatter)

    def _mbh_relation(self, vals, relation, forward, scatter):
        x0, y0, alpha, beta, eps = self._values(relation)
        scatter = self._parse_scatter(scatter)
        shape = np.shape(vals)

        params = [alpha, beta, [0.0, eps]]
        for ii, vv in enumerate(params):
            if (scatter > 0.0):
                vv = np.random.normal(vv[0], vv[1]*scatter, size=shape)
            else:
                vv = vv[0]

            params[ii] = vv

        alpha, beta, eps = params
        if forward:
            rv = self._forward_relation(vals, x0, y0, alpha, beta, eps)
        else:
            rv = self._reverse_relation(vals, x0, y0, alpha, beta, eps)

        return rv

    def _forward_relation(self, xx, x0, y0, alpha, beta, eps):
        mbh = alpha + beta * np.log10(xx/x0) + eps
        mbh = np.power(10.0, mbh) * y0
        return mbh

    def _reverse_relation(self, yy, x0, y0, alpha, beta, eps):
        gval = np.log10(yy / y0) - eps - alpha
        gval = x0 * np.power(10.0, gval / beta)
        return gval


class McConnell_Ma_2013(_Galaxy_Blackhole_Relation):
    """

    [MM13] - McConnell+Ma-2013 :
    - https://ui.adsabs.harvard.edu/abs/2013ApJ...764..184M/abstract

    Scaling-relations are of the form,
    `log_10(Mbh/Msol) = alpha + beta * log10(X) + eps`
        where `X` is:
        `sigma / (200 km/s)`
        `L / (1e11 Lsol)`
        `Mbulge / (1e11 Msol)`
        and `eps` is an intrinsic scatter in Mbh

    """

    # 1211.2816 - Table 2
    FITS = {
        # "All galaxies", first row ("MPFITEXY")
        'vdisp': {
            'alpha': [8.32, 0.05],   # normalization
            'beta': [5.64, 0.32],    # power-law index
            'eps': 0.38,      # overall scatter
            'norm': 200 * 1e5,       # units
        },
        # "Dynamical masses", first row ("MPFITEXY")
        'mbulge': {
            'alpha': [8.46, 0.08],
            'beta': [1.05, 0.11],
            'eps': 0.34,
            'norm': 1e11 * MSOL,
        }
    }

    NORM = {
        'vdisp': {
            'x': 200 * 1e5,   # velocity-dispersion units
            'y': MSOL,        # MBH units
        },

        'mbulge': {
            'x': 1e11 * MSOL,   # MBulge units
            'y': MSOL,        # MBH units
        },
    }


class Kormendy_Ho_2013(_Galaxy_Blackhole_Relation):
    """

    [KH13] - Kormendy+Ho-2013 : https://ui.adsabs.harvard.edu/abs/2013ARA%26A..51..511K/abstract
    -

    Scaling-relations are given in the form,
    `Mbh/(1e9 Msol) = [alpha ± da] * (X)^[beta ± db] + eps`
    and converted to
    `Mbh/(1e9 Msol) = [delta ± dd] + [beta ± db] * log10(X) + eps`
    s.t.  `delta = log10(alpha)`  and  `dd = (da/alpha) / ln(10)`

        where `X` is:
        `Mbulge / (1e11 Msol)`
        `sigma / (200 km/s)`
        and `eps` is an intrinsic scatter in Mbh

    """

    # 1304.7762
    FITS = {
        # Eq.12
        'vdisp': {
            'alpha': [-0.54, 0.07],  # normalization
            'beta': [4.26, 0.44],    # power-law index
            'eps': 0.30,             # overall scatter
        },
        # Eq.10
        'mbulge': {
            'alpha': [-0.3098, 0.05318],
            'beta': [1.16, 0.08],
            'eps': 0.29,
        }
    }

    NORM = {
        # Eq.12
        'vdisp': {
            'x': 200 * 1e5,     # velocity-dispersion units
            'y': 1e9 * MSOL,    # MBH units
        },
        # Eq.10
        'mbulge': {
            'x': 1e11 * MSOL,   # MBulge units
            'y': 1e9 * MSOL,    # MBH units
        },
    }
