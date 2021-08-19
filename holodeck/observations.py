"""

To-Do
-----
* [ ] Pass concentration-relation (or other method to calculate) to NFW classes on instantiation


References
----------
* [NFW-97] : Navarro, Frenk & White 1997
    A Universal Density Profile from Hierarchical Clustering
    https://ui.adsabs.harvard.edu/abs/1997ApJ...490..493N/abstract

* [Guo-2010] : Guo, White, Li & Boylan-Kolchin 2010
    How do galaxies populate dark matter haloes?
    https://ui.adsabs.harvard.edu/abs/2010MNRAS.404.1111G/abstract

* [Kormendy+2013] : Kormendy & Ho 2013
    Coevolution (Or Not) of Supermassive Black Holes and Host Galaxies
    https://ui.adsabs.harvard.edu/abs/2013ARA%26A..51..511K/abstract

* [McConnell+2013] : McConnell & Ma 2013
    Revisiting the Scaling Relations of Black Hole Masses and Host Galaxy Properties
    https://ui.adsabs.harvard.edu/abs/2013ApJ...764..184M/abstract

* [Behroozi+2013] : Behroozi, Wechsler & Conroy 2013
    The Average Star Formation Histories of Galaxies in Dark Matter Halos from z = 0-8
    https://ui.adsabs.harvard.edu/abs/2013ApJ...770...57B/abstract

* [Klypin+2016] : Klypin et al. 2016
    MultiDark simulations: the story of dark matter halo concentrations and density profiles
    https://ui.adsabs.harvard.edu/abs/2016MNRAS.457.4340K/abstract

"""

import abc

import numpy as np
import scipy as sp

from holodeck import log, cosmo, utils
from holodeck.constants import MSOL, NWTG


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


# =================================================================================================
# ====                              Density Profiles & Relations                               ====
# =================================================================================================


class Klypin_2016:
    """Interpolate between redshifts and masses to find DM halo concentrations.

    Eq. 24 & Table 2
    """
    _redz = [0.00e+00, 3.50e-01, 5.00e-01, 1.00e+00, 1.44e+00,
             2.15e+00, 2.50e+00, 2.90e+00, 4.10e+00, 5.40e+00]
    _c0 = [7.40e+00, 6.25e+00, 5.65e+00, 4.30e+00, 3.53e+00,
           2.70e+00, 2.42e+00, 2.20e+00, 1.92e+00, 1.65e+00]
    _gamma = [1.20e-01, 1.17e-01, 1.15e-01, 1.10e-01, 9.50e-02,
              8.50e-02, 8.00e-02, 8.00e-02, 8.00e-02, 8.00e-02]
    _mass0 = [5.50e+05, 1.00e+05, 2.00e+04, 9.00e+02, 3.00e+02,
              4.20e+01, 1.70e+01, 8.50e+00, 2.00e+00, 3.00e-01]

    _interp = lambda xx, yy: sp.interpolate.interp1d(xx, yy, kind='linear', bounds_error=False, fill_value=np.nan)
    _zz = np.log10(1 + np.array(_redz))
    _lin_interp_c0 = _interp(_zz, np.log10(_c0))
    _lin_interp_gamma = _interp(_zz, np.log10(_gamma))
    _lin_interp_mass0 = _interp(_zz, np.log10(_mass0)+np.log10(1e12 * MSOL / cosmo.h))

    @classmethod
    def _c0(cls, redz):
        xx = np.log10(1 + redz)
        yy = np.power(10.0, cls._lin_interp_c0(xx))
        return yy

    @classmethod
    def _gamma(cls, redz):
        xx = np.log10(1 + redz)
        yy = np.power(10.0, cls._lin_interp_gamma(xx))
        return yy

    @classmethod
    def _mass0(cls, redz):
        xx = np.log10(1 + redz)
        yy = np.power(10.0, cls._lin_interp_mass0(xx))
        return yy

    @classmethod
    def concentration(cls, mass, redz):
        c0 = cls._c0(redz)
        gamma = cls._gamma(redz)
        mass0 = cls._mass0(redz)
        f1 = np.power(mass/(1e12*MSOL/cosmo.h), -gamma)
        f2 = 1 + np.power(mass/mass0, 0.4)
        conc = c0 * f1 * f2
        return conc


class _Density_Profile(abc.ABC):

    @abc.abstractmethod
    def density(self, rads, *args, **kwargs):
        pass

    @classmethod
    def mass(cls, rads, *args, **kwargs):
        dens = cls.density(rads, *args, **kwargs)
        yy = 4*np.pi*rads**2 * dens
        mass = utils.cumtrapz_loglog(yy, rads)
        m0 = dens[0] * (4.0/3.0) * np.pi * rads[0] ** 3
        mass = np.concatenate([[m0], mass + m0])
        return mass

    @classmethod
    def time_dynamical(cls, rads, *args, **kwargs):
        """Dynamical time, defined as (G M_enc / r^3) ^ -1/2 = r / v_circ
        """
        tdyn = rads / cls.velocity_circular(rads, *args, **kwargs)
        return tdyn

    @classmethod
    def velocity_circular(cls, rads, *args, **kwargs):
        """Circular velocity, defined as (G M_enc / r) ^ -1/2
        """
        mass = cls.mass(rads, *args, **kwargs)
        time = NWTG * mass / rads
        time = time ** -0.5
        return time


class NFW(_Density_Profile):

    @classmethod
    def density(cls, rads, mhalo, redz):
        # Get Halo concentration
        conc = cls._concentration(mhalo, redz)
        dens = cls._density(rads, mhalo, conc, redz)
        return dens

    @staticmethod
    def _concentration(mhalo, redz):
        return Klypin_2016.concentration(mhalo, redz)

    @classmethod
    def radius_scale(cls, mhalo, redz):
        conc = cls._concentration(mhalo, redz)
        log_c_term = np.log(1 + conc) - conc/(1+conc)

        # Critical over-density
        delta_c = (200/3) * (conc**3) / log_c_term
        # NFW density (*not* the density at the characteristic-radius)
        rho_s = cosmo.critical_density(redz).cgs.value * delta_c
        # scale-radius
        rs = mhalo / (4*np.pi*rho_s*log_c_term)
        rs = np.power(rs, 1.0/3.0)
        return rs

    @staticmethod
    def _density(rads, mhalo, conc, redz):
        """NFW DM Density profile.

        [NFW-97]
        """
        log_c_term = np.log(1 + conc) - conc/(1+conc)

        # Critical over-density
        delta_c = (200/3) * (conc**3) / log_c_term
        # NFW density (*not* the density at the characteristic-radius)
        rho_s = cosmo.critical_density(redz).cgs.value * delta_c
        # scale-radius
        rs = mhalo / (4*np.pi*rho_s*log_c_term)
        rs = np.power(rs, 1.0/3.0)

        # Calculate NFW profile
        dens = rads / rs
        dens = dens * np.square(1 + dens)
        dens = rho_s / dens
        return dens


# =================================================================================================
# ====                             Stellar-Mass Halo-Mass Relation                             ====
# =================================================================================================


class _StellarMass_HaloMass(abc.ABC):

    _NUM_GRID = 200
    _MHALO_GRID_EXTR = [9, 16]

    def __init__(self):
        self._mhalo_grid = np.logspace(*self._MHALO_GRID_EXTR, self._NUM_GRID) * MSOL
        self._mstar = self.stellar_mass(self._mhalo)

        xx = np.log10(self._mstar / MSOL)
        yy = np.log10(self._mhalo_grid / MSOL)
        self._mhalo_from_mstar = sp.interpolate.interp1d(xx, yy, kind='linear', bounds_error=False, fill_value=np.nan)
        return

    @abc.abstractmethod
    def stellar_mass(self, mhalo):
        pass

    def halo_mass(self, mstar):
        ynew = MSOL * 10.0 ** self._mhalo_from_mstar(np.log10(mstar/MSOL))
        return ynew


class _StellarMass_HaloMass_Redshift(_StellarMass_HaloMass):

    _REDZ_GRID_EXTR = [0.0, 9.0]

    def __init__(self):
        self._mhalo_grid = np.logspace(*self._MHALO_GRID_EXTR, self._NUM_GRID) * MSOL
        self._redz_grid = np.linspace(*self._REDZ_GRID_EXTR, self._NUM_GRID+2)
        mhalo = self._mhalo_grid[:, np.newaxis]
        redz = self._redz_grid[np.newaxis, :]
        self._mstar = self.stellar_mass(mhalo, redz)  # should be units of [Msol]

        # ---- Construct interpolator to go from (mstar, redz) ==> (mhalo)
        # first: convert data to grid of (mstar, redz) ==> (mhalo)
        mstar = np.log10(self._mstar / MSOL)
        redz = self._redz_grid
        mhalo = np.log10(self._mhalo_grid / MSOL)
        aa = mstar.ravel()
        cc, bb = np.meshgrid(mhalo, redz, indexing='ij')
        bb, cc = [bc.ravel() for bc in [bb, cc]]

        shape = self._mstar.shape
        xx = np.linspace(aa.min(), aa.max(), shape[0])
        yy = redz
        xg, yg = np.meshgrid(xx, yy, indexing='ij')
        grid = sp.interpolate.griddata((aa, bb), cc, (xg, yg))
        # second: construct interpolator from grid to arbitrary scatter points
        interp = sp.interpolate.RegularGridInterpolator((xx, yy), grid)
        self._mhalo_from_mstar_redz = interp
        return

    @abc.abstractmethod
    def stellar_mass(self, mhalo, redz):
        pass

    def halo_mass(self, mstar, redz):
        if (np.ndim(mstar) not in [0, 1]) or np.any(np.shape(mstar) != np.shape(redz)):
            err = f"both `mstar` ({np.shape(mstar)}) and `redz` ({np.shape(redz)}) must be 1D and same length!"
            log.error(err)
            raise ValueError(err)

        squeeze = np.isscalar(mstar)

        vals = np.array([np.log10(mstar/MSOL), redz])
        ynew = MSOL * 10.0 ** self._mhalo_from_mstar_redz(vals.T)
        if squeeze:
            ynew = ynew.squeeze()

        return ynew


class Guo_2010(_StellarMass_HaloMass):
    """
    Guo+2010 : Eq.3
    https://ui.adsabs.harvard.edu/abs/2010MNRAS.404.1111G/abstract
    """

    _NORM = 0.129
    _M0 = (10**11.4) * MSOL
    _ALPHA = 0.926
    _BETA = 0.261
    _GAMMA = 2.440

    @classmethod
    def stellar_mass(cls, mhalo):
        M0 = cls._M0
        t1 = np.power(mhalo/M0, -cls._ALPHA)
        t2 = np.power(mhalo/M0, +cls._BETA)
        mstar = mhalo * cls._NORM * np.power(t1 + t2, -cls._GAMMA)
        return mstar


class Behroozi_2013(_StellarMass_HaloMass_Redshift):
    """
    [Behroozi+2013] best fit values are at the beginning of Section 5 (pg.9), uncertainties are 1-sigma
    """

    def __init__(self, *args, **kwargs):
        self._f0 = self._f_func(0.0)
        super().__init__(*args, **kwargs)
        return

    def _nu_func(sca):
        """[Behroozi+2013] Eq. 4"""
        return np.exp(-4.0 * sca*sca)

    @classmethod
    def _param_func(cls, redz, v0, va, vz, va2=None):
        """[Behroozi+2013] Eq. 4"""
        rv = v0
        sca = cosmo._z_to_a(redz)
        rv = rv + cls._nu_func(sca) * (va * (sca - 1.0) + vz * redz)
        if va2 is not None:
            rv += va2 * (sca - 1.0)
        return rv

    @classmethod
    def _eps(cls, redz=0.0):
        e0 = -1.777   # +0.133 -0.146
        ea = -0.006   # +0.113 -0.361
        ez = 0.0      # +0.003 -0.104
        ea2 = -0.119  # +0.061 -0.012
        eps = 10.0 ** cls._param_func(redz, e0, ea, ez, va2=ea2)
        return eps

    @classmethod
    def _m1(cls, redz=0.0):
        m0 = 11.514   # +0.053 -0.009
        ma = -1.793   # +0.315 -0.330
        mz = -0.251   # +0.012 -0.125
        m1 = MSOL * 10.0 ** cls._param_func(redz, m0, ma, mz)
        return m1

    @classmethod
    def _alpha(cls, redz=0.0):
        a0 = -1.412   # +0.020 -0.105
        aa = 0.731    # +0.344 -0.296
        az = 0.0
        alpha = cls._param_func(redz, a0, aa, az)
        return alpha

    @classmethod
    def _delta(cls, redz=0.0):
        d0 = 3.508   # +0.087 -0.369
        da = 2.608   # +2.446 -1.261
        dz = -0.043  # +0.958 -0.071
        delta = cls._param_func(redz, d0, da, dz)
        return delta

    @classmethod
    def _gamma(cls, redz=0.0):
        g0 = 0.316   # +0.076 -0.012
        ga = 1.319   # +0.584 -0.505
        gz = 0.279   # +0.256 -0.081
        gamma = cls._param_func(redz, g0, ga, gz)
        return gamma

    @classmethod
    def _xsi(cls, redz=0.0):
        """[Behroozi+2013] Eq.5"""
        x0 = 0.218   # +0.011 -0.033
        xa = -0.023  # +0.052 -0.068

        xsi = x0
        if redz > 0.0:
            sca = cosmo._z_to_z(redz)
            xsi += xa * (sca - 1.0)

        return xsi

    @classmethod
    def _f_func(cls, xx, redz=0.0):
        """[Behroozi+2013] Eq.3 (lower)"""
        alpha = cls._alpha(redz)
        delta = cls._delta(redz)
        gamma = cls._gamma(redz)

        t1 = -np.log10(10**(alpha*xx) + 1)
        t2 = np.log10(1 + np.exp(xx)) ** gamma
        t3 = 1 + np.exp(10.0 ** -xx)
        ff = t1 + delta * t2 / t3
        return ff

    def stellar_mass(self, mhalo, redz):
        """[Behroozi+2013] Eq.3 (upper)"""
        eps = self._eps(redz)
        m1 = self._m1(redz)
        mstar = np.log10(eps*m1/MSOL) + self._f_func(np.log10(mhalo/m1), redz) - self._f0
        mstar = np.power(10.0, mstar) * MSOL
        return mstar
