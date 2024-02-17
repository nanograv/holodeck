"""Semi-Analytic Model - Components

This module provides the key building blocks for the holodeck SAMs.  In particular, the:

* Galaxy Stellar-Mass Function (GSMF) : number-density of galaxies as a function of stellar mass;
* Galaxy Merger Rate (GMR) : rate of galaxy mergers per galaxy;
* Galaxy Pair Fraction (GPF) : fraction of galaxy pairs, relative to all galaxies;
* Galaxy Merger Time (GMT) : duration over which galaxy pairs are observable as pairs.

For more information see the :mod:`holodeck.sams` module, or the :doc:`SAMs getting-started guide
<../getting_started/index>`.

References
----------
* [Sesana2008]_ Sesana, Vecchio, Colacino 2008.
* [Rodriguez-Gomez2015]_  Rodriguez-Gomez, Genel, Vogelsberger, et al. 2015
    The merger rate of galaxies in the Illustris simulation: a comparison with observations and semi-empirical models
    https://ui.adsabs.harvard.edu/abs/2015MNRAS.449...49R/abstract
* [Chen2019]_ Chen, Sesana, Conselice 2019.
* [Leja2020]_ Leja, Speagle, Johnson, et al. 2020.
    A New Census of the 0.2 < z < 3.0 Universe. I. The Stellar Mass Function
    https://ui.adsabs.harvard.edu/abs/2020ApJ...893..111L/abstract

"""

import abc

import numpy as np

import holodeck as holo
from holodeck import cosmo, utils
from holodeck.constants import GYR, MSOL


# ----    Galaxy Stellar-Mass Function    ----


class _Galaxy_Stellar_Mass_Function(abc.ABC):
    """Galaxy Stellar-Mass Function base-class.  Used to calculate number-density of galaxies.
    """

    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        return

    @abc.abstractmethod
    def __call__(self, mstar, redz):
        """Return the number-density of galaxies at a given stellar mass, per log10 interval of stellar-mass.

        i.e. Phi = dn / dlog10(M)

        Parameters
        ----------
        mstar : scalar or ndarray
            Galaxy stellar-mass in units of [grams]
        redz : scalar or ndarray
            Redshift.

        Returns
        -------
        rv : scalar or ndarray
            Number-density of galaxies in units of [Mpc^-3]

        """
        return

    def mbh_mass_func(self, mbh, redz, mmbulge, scatter=None):
        """Convert from the GSMF to a MBH mass function (number density), using a given Mbh-Mbulge relation.

        Parameters
        ----------
        mbh : array_like
            Blackhole masses at which to evaluate the mass function.
        redz : array_like
            Redshift(s) at which to evaluate the mass function.
        mmbulge : `relations._MMBulge_Relation` subclass instance
            Scaling relation between galaxy and MBH masses.
        scatter : None, bool, or float
            Introduce scatter in masses.
            * `None` or `True` : use the value from `mmbulge._scatter_dex`
            * `False` : do not introduce scatter
            * float : introduce scatter with this amplitude (in dex)

        Returns
        -------
        ndens : array_like
            Number density of MBHs, in units of [Mpc^-3]

        """
        if scatter in [None, True]:
            scatter = mmbulge._scatter_dex

        mstar = mmbulge.mstar_from_mbh(mbh, scatter=False)
        # This is `dn_star / dlog10(M_star)`
        ndens = self(mstar, redz)    # units of  [1/Mpc^3]

        # dM_star / dM_bh
        dmstar_dmbh = mmbulge.dmstar_dmbh(mstar)   # [unitless]
        # convert to dlog10(M_star) / dlog10(M_bh) = (M_bh / M_star) * (dM_star / dM_bh)
        jac = (mbh/mstar) * dmstar_dmbh
        # convert galaxy number density to  to dn_bh / dlog10(M_bh)
        ndens *= jac

        if scatter is not False:
            ndens = holo.utils.scatter_redistribute_densities(mbh, ndens, scatter=scatter)

        return ndens


class GSMF_Schechter(_Galaxy_Stellar_Mass_Function):
    r"""Single Schechter Function - Galaxy Stellar Mass Function.

    This is density per unit log10-interval of stellar mass, i.e. $\Phi = dn / d\log_{10}(M)$

    See: [Chen2019]_ Eq.9 and enclosing section.

    """

    def __init__(self, phi0=-2.77, phiz=-0.27, mchar0_log10=11.24, mchar0=None, mcharz=0.0, alpha0=-1.24, alphaz=-0.03):
        mchar0, _ = utils._parse_val_log10_val_pars(
            mchar0, mchar0_log10, val_units=MSOL, name='mchar0', only_one=True
        )

        self._phi0 = phi0         # - 2.77  +/- [-0.29, +0.27]  [log10(1/Mpc^3)]
        self._phiz = phiz         # - 0.27  +/- [-0.21, +0.23]  [log10(1/Mpc^3)]
        self._mchar0 = mchar0       # 10^ (+11.24  +/- [-0.17, +0.20]  [log10(Msol)])
        self._mcharz = mcharz       #  0.0                        [log10(Msol)]    # noqa
        self._alpha0 = alpha0     # -1.24   +/- [-0.16, +0.16]
        self._alphaz = alphaz     # -0.03   +/- [-0.14, +0.16]
        return

    def __call__(self, mstar, redz):
        r"""Return the number-density of galaxies at a given stellar mass.

        See: [Chen2019]_ Eq.8

        Parameters
        ----------
        mstar : scalar or ndarray
            Galaxy stellar-mass in units of [grams]
        redz : scalar or ndarray
            Redshift.

        Returns
        -------
        rv : scalar or ndarray
            Number-density of galaxies per log-interval of mass in units of [Mpc^-3]
            i.e.  ``Phi = dn / d\\log_{10}(M)``

        """
        phi = self._phi_func(redz)
        mchar = self._mchar_func(redz)
        alpha = self._alpha_func(redz)
        xx = mstar / mchar
        # [Chen2019]_ Eq.8
        rv = np.log(10.0) * phi * np.power(xx, 1.0 + alpha) * np.exp(-xx)
        return rv

    def _phi_func(self, redz):
        """See: [Chen2019]_ Eq.9
        """
        return np.power(10.0, self._phi0 + self._phiz * redz)

    def _mchar_func(self, redz):
        """See: [Chen2019]_ Eq.10 - NOTE: added `redz` term
        """
        return self._mchar0 + self._mcharz * redz

    def _alpha_func(self, redz):
        """See: [Chen2019]_ Eq.11
        """
        return self._alpha0 + self._alphaz * redz


class _GSMF_Single_Schechter(_Galaxy_Stellar_Mass_Function):
    r"""Schechter function, with parameters as quadratics with respect to redshift.

    Parameterization follows [Leja2020]_ and is primarily for use in the
    :class:`GSMF_Double_Schechter` class.  From [Leja2020]_ Eq.14:

    .. math::

        \frac{\partial n}{\partial \log_{10} \! M} =
            \ln(10) \phi \left(\frac{M}{M_\star}\right)^{\alpha+1} \exp[-M/M_\star].

    The power-law index $\alpha$ is a scalar value, while the reference mass $M_\star$, and the
    normalization $\phi$ are defined as quadratics with respect to redshift:

    .. math::

        \log_{10}(\phi) & = a_0 + a_1 z + a_2 z^2, \\
        \log_{10}(M_\star) & = b_0 + b_1 z + b_2 z^2.

    Class instances are callable, see :meth:`__call__`, and return galaxy number densities in units
    of $[\mathrm{Mpc}^{-3} \, \mathrm{dex}^{-1}]$.

    """

    def __init__(self, log10_phi_terms, log10_mstar_terms, alpha):
        r"""Initialize a Schechter function GSMF.

        Arguments
        ---------
        log10_phi_terms : (3,) of float
            Three terms determining the redshift behavior of
            $\log_{10}(\phi / \mathrm{Mpc}^{-3} \mathrm{dex}^{-1})$, the normalization in units of
            $[\mathrm{Mpc}^{-3} \, \mathrm{dex}^{-1}]$.
        log10_mstar_terms : (3,) of float
            Three terms determining the redshift behavior of $\log_{10}(M_\star/M_\odot)$, the
            characteristic mass, in units of solar masses.
        alpha : float
            Power-law index for GSMF in terms of $dN/dM$, even though the function is written in
            terms of $dn/d \log_{10} \! M$.

        Returns
        -------
        None

        """
        self._log10_phi_terms = log10_phi_terms      #: these are the $a_i$ terms in the defintion.
        self._log10_mstar_terms = log10_mstar_terms  #: these are the $b_i$ terms.
        self._alpha = alpha                          #: power-law index
        return

    def __call__(self, mstar, redz):
        r"""Evaluate this GSMF instance at the target stellar-mass(es) and redshift(s).

        Arguments
        ---------
        mstar : array_like of float, [Msol]
            Stellar mass(es), in units of [gram], at which to evaluate the GSMF.
            Must be broadcastable against ``redz``.
        redz : array_like of float
            Redshift(s) at which to evaluate the GSMF.  Must be broadcastable against ``mstar``.

        Returns
        -------
        rv : float  or  array_like of float
            Number density of galaxies, $dn/d \log_{10} \! M$, in units of
            $[\mathrm{Mpc}^{-3} \, \mathrm{dex}^{-1}]$.

        """
        phi = self._phi_func(redz)
        mchar = self._mstar_func(redz)
        alpha = self._alpha
        xx = mstar / mchar
        rv = np.log(10.0) * phi * np.power(xx, 1.0 + alpha) * np.exp(-xx)
        return rv

    def _phi_func(self, redz):
        """Evaluate the GSMF normalization (phi) at the given redshift(s).

        Arguments
        ---------
        redz : array_like of float,
            Redshift(s) at which to calculate the GSMF normalization.

        Returns
        -------
        phi : array_like of float, [Mpc^-3 dex^-1]
            Number density normalization.

        """
        cc = self._log10_phi_terms
        phi = np.power(10.0, cc[0] + cc[1] * redz + cc[2] * redz**2)
        return phi

    def _mstar_func(self, redz):
        r"""Evaluate the GSMF characteristic mass ($M_\star$) at the given redshift(s).
        """
        cc = self._log10_mstar_terms
        mstar = MSOL * np.power(10.0, cc[0] + cc[1] * redz + cc[2] * redz**2)
        return mstar


class GSMF_Double_Schechter(_Galaxy_Stellar_Mass_Function):
    r"""Sum of two Schechter functions, each parameterized as quadratics in redshift.

    For each Schechter Function (:class:`_GSMF_Single_Schechter`), the normalizations ($\phi$) and
    characteristic masses ($M_\star$) are parameterized as quadratics with respect to redshift.

    Each Schechter function is parameterized as,

    .. math::

        \frac{\partial n}{\partial \log_{10} \! M} & =
            \ln(10) \, \phi \cdot \left(\frac{M}{M_\star}\right)^{\alpha+1} \exp[-M/M_\star], \\
        \log_{10}(\phi) & = a_0 + a_1 z + a_2 z^2, \\
        \log_{10}(M_\star) & = b_0 + b_1 z + b_2 z^2.

    The parameters for $\phi$ and $\alpha$ are different for the two functions, while the parameters
    for $M_\star$ are shared (i.e. the same characteristic mass is used for both).  Default
    parameters are the best fits from [Leja2020]_.  With uncertainties, these are::

        log10(phi_1):  [ -2.383 ± 0.027,  -0.264 ± 0.071,  -0.107 ± 0.030],
        log10(phi_2):  [ -2.818 ± 0.050,  -0.368 ± 0.070,  +0.046 ± 0.020],
        log10(M_star): [+10.767 ± 0.026,  +0.124 ± 0.045,  -0.033 ± 0.015].

    """

    def __init__(
        self,
        log10_phi1=[-2.383, -0.264, -0.107],
        log10_phi2=[-2.818, -0.368, +0.046],
        log10_mstar=[+10.767, +0.124, -0.033],
        alpha1=-0.28,
        alpha2=-1.48
    ):
        gsmf_one = _GSMF_Single_Schechter(log10_phi_terms=log10_phi1, log10_mstar_terms=log10_mstar, alpha=alpha1)
        gsmf_two = _GSMF_Single_Schechter(log10_phi_terms=log10_phi2, log10_mstar_terms=log10_mstar, alpha=alpha2)
        self._gsmf_one = gsmf_one
        self._gsmf_two = gsmf_two
        return

    def __call__(self, mstar, redz):
        """Evaluate the double Schechter function at the given stellar masses [Msol] and redshifts.

        Arguments
        ---------
        mstar : array_like of float, [Msol]
            Galaxy stellar masses at which to evaluate the GSMF.  Units of solar-masses.  Must be
            broadcastable against ``redz``.
        redz : array_like of float,
            Redshift(s) at which to evaluate the GSMF.  Must be broadcastable against ``mstar``.

        """
        vals = self._gsmf_one(mstar, redz)
        vals += self._gsmf_two(mstar, redz)
        return vals


# ----    Galaxy Merger Rate    ----


class _Galaxy_Merger_Rate(abc.ABC):
    r"""Galaxy Merger Rate base class, used to model merger rates of galaxy pairs.

    NOTE: the definition of mass is ambiguous, i.e. whether it is the primary mass, the
    combined system mass, the descendent mass, etc.

    :class:`_Galaxy_Merger_Rate` instances are callable, returning the specific galaxy merger-rate
    (i.e. the rate per galaxy), per unit mass-ratio, i.e.

    .. math::

        R = \frac{\partial N_\mathrm{mergers}(M_\star, q_\star, z)}{\partial q_\star \, \partial t}.

    Typically the mass $M_\star$ is taken to be that of the descendent (roughly the combined mass
    of the two galaxies), and the mass ratio is $q_\star \equiv M_{2,\star} / M_{1,\star} \leq 1.0$.

    """

    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        return

    @abc.abstractmethod
    def __call__(self, mass, mrat, redz):
        r"""Return the galaxy merger rate for the given parameters.

        Parameters
        ----------
        mass : (N,) array_like[scalar]
            Mass of the system, units of [grams].
        mrat : scalar or ndarray,
            Mass-ratio of the system (m2/m1 <= 1.0), dimensionless.
        redz : scalar or ndarray,
            Redshift.

        Returns
        -------
        rv : scalar or ndarray,
            Galaxy merger rate, per unit mass-ratio, in units of [1/sec], i.e.
            $\partial N / \partial q_\star \, \partial t$

        """
        return


class GMR_Illustris(_Galaxy_Merger_Rate):
    """Galaxy Merger Rate - based on multiple power-laws.

    See [Rodriguez-Gomez2015]_, Table 1.
    "merger rate as a function of descendant stellar mass M_star, progenitor stellar mass ratio mu_star"

    """

    def __init__(
        self,
        norm0_log10=None,
        normz=None,
        malpha0=None,
        malphaz=None,
        mdelta0=None,
        mdeltaz=None,
        qgamma0=None,
        qgammaz=None,
        qgammam=None,
    ):

        if norm0_log10 is None:
            norm0_log10 = -2.2287      # -2.2287 ± 0.0045    A0 [log10(A*Gyr)]
        if normz is None:
            normz = +2.4644            # +2.4644 ± 0.0128    eta
        if malpha0 is None:
            malpha0 = +0.2241          # +0.2241 ± 0.0038    alpha0
        if malphaz is None:
            malphaz = -1.1759          # -1.1759 ± 0.0316    alpha1
        if mdelta0 is None:
            mdelta0 = +0.7668          # +0.7668 ± 0.0202    delta0
        if mdeltaz is None:
            mdeltaz = -0.4695          # -0.4695 ± 0.0440    delta1
        if qgamma0 is None:
            qgamma0 = -1.2595          # -1.2595 ± 0.0026    beta0
        if qgammaz is None:
            qgammaz = +0.0611          # +0.0611 ± 0.0021    beta1
        if qgammam is None:
            qgammam = -0.0477          # -0.0477 ± 0.0013    gamma

        self._norm0 = (10.0 ** norm0_log10) / GYR              # [1/sec]
        self._normz = normz

        self._malpha0 = malpha0
        self._malphaz = malphaz
        self._mdelta0 = mdelta0
        self._mdeltaz = mdeltaz
        self._qgamma0 = qgamma0
        self._qgammaz = qgammaz
        self._qgammam = qgammam

        self._mref_delta = 2.0e11 * MSOL   # fixed value
        self._mref = 1.0e10 * MSOL   # fixed value
        return

    def _get_norm(self, redz):
        norm = self._norm0 * np.power(1.0 + redz, self._normz)
        return norm

    def _get_malpha(self, redz):
        malpha = self._malpha0 * np.power(1.0 + redz, self._malphaz)
        return malpha

    def _get_mdelta(self, redz):
        mdelta = self._mdelta0 * np.power(1.0 + redz, self._mdeltaz)
        return mdelta

    def _get_qgamma(self, redz, mtot):
        qgamma = self._qgamma0 * np.power(1.0 + redz, self._qgammaz)
        qgamma = qgamma + self._qgammam * np.log10(mtot/self._mref)
        return qgamma

    def __call__(self, mtot, mrat, redz):
        """Return the galaxy merger rate for the given parameters.

        Parameters
        ----------
        mtot : (N,) array_like[scalar]
            Total mass of the system, units of [grams].
        mrat : (N,) array_like[scalar]
            Mass ratio of each binary.
        redz : (N,) array_like[scalar]
            Redshifts of each binary.

        Returns
        -------
        rate : array_like
            Merger rate in [1/sec].

        """
        norm = self._get_norm(redz)
        malpha = self._get_malpha(redz)
        mdelta = self._get_mdelta(redz)
        qgamma = self._get_qgamma(redz, mtot)

        xx = (mtot/self._mref)
        mt = np.power(xx, malpha)
        yy = mtot/self._mref_delta
        mp1t = np.power(1.0 + yy, mdelta)
        qt = np.power(mrat, qgamma)

        rate = norm * mt * mp1t * qt
        return rate


# ----    Galaxy Pair Fraction    ----


class _Galaxy_Pair_Fraction(abc.ABC):
    """Galaxy Pair Fraction base class, used to describe the fraction of galaxies in mergers/pairs.
    """

    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        return

    @abc.abstractmethod
    def __call__(self, mass, mrat, redz):
        """Return the fraction of galaxies in pairs of the given parameters.

        Parameters
        ----------
        mass : array_like,
            Mass of the system, units of [grams].
            NOTE: the definition of mass is ambiguous, i.e. whether it is the primary mass, or the
            combined system mass.
        mrat : array_like,
            Mass-ratio of the system (m2/m1 <= 1.0), dimensionless.
        redz : array_like,
            Redshift.

        Returns
        -------
        rv : scalar or ndarray,
            Galaxy pair fraction, dimensionless.

        """
        return


class GPF_Power_Law(_Galaxy_Pair_Fraction):
    """Galaxy Pair Fraction - Single Power-Law
    """

    def __init__(self, frac_norm_allq=0.025, frac_norm=None, mref=None, mref_log10=11.0,
                 malpha=0.0, zbeta=0.8, qgamma=0.0, obs_conv_qlo=0.25, max_frac=1.0):

        mref, _ = utils._parse_val_log10_val_pars(
            mref, mref_log10, val_units=MSOL, name='mref', only_one=True
        )

        # If the pair-fraction integrated over all mass-ratios is given (f0), convert to regular (f0-prime)
        if frac_norm is None:
            if frac_norm_allq is None:
                raise ValueError("If `frac_norm` is not given, `frac_norm_allq` is requried!")
            pow = qgamma + 1.0
            qlo = obs_conv_qlo
            qhi = 1.00
            pair_norm = (qhi**pow - qlo**pow) / pow
            frac_norm = frac_norm_allq / pair_norm

        # normalization corresponds to f0-prime in [Chen2019]_
        self._frac_norm = frac_norm   # f0 = 0.025 b/t [+0.02, +0.03]  [+0.01, +0.05]
        self._malpha = malpha         #      0.0   b/t [-0.2 , +0.2 ]  [-0.5 , +0.5 ]  # noqa
        self._zbeta = zbeta           #      0.8   b/t [+0.6 , +0.1 ]  [+0.0 , +2.0 ]  # noqa
        self._qgamma = qgamma         #      0.0   b/t [-0.2 , +0.2 ]  [-0.2 , +0.2 ]  # noqa

        if (max_frac < 0.0) or (1.0 < max_frac):
            err = f"Given `max_frac`={max_frac:.4f} must be between [0.0, 1.0]!"
            holo.log.exception(err)
            raise ValueError(err)
        self._max_frac = max_frac

        self._mref = mref   # NOTE: this is `a * M_0 = 1e11 Msol` in papers
        return

    def __call__(self, mass, mrat, redz):
        """Return the fraction of galaxies in pairs of the given parameters.

        Parameters
        ----------
        mass : array_like,
            Mass of the system, units of [grams].
            NOTE: the definition of mass is ambiguous, i.e. whether it is the primary mass, or the
            combined system mass.
        mrat : array_like,
            Mass-ratio of the system (m2/m1 <= 1.0), dimensionless.
        redz : array_like,
            Redshift.

        Returns
        -------
        rv : scalar or ndarray,
            Galaxy pair fraction, dimensionless.

        """
        f0p = self._frac_norm
        am0 = self._mref
        aa = self._malpha
        bb = self._zbeta
        gg = self._qgamma
        rv = f0p * np.power(mass/am0, aa) * np.power(1.0 + redz, bb) * np.power(mrat, gg)
        rv = np.clip(rv, None, self._max_frac)
        return rv


# ----    Galaxy Merger Time    ----


class _Galaxy_Merger_Time(abc.ABC):
    """Galaxy Merger Time base class, used to model merger timescale of galaxy pairs.
    """

    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        return

    @abc.abstractmethod
    def __call__(self, mass, mrat, redz):
        """Return the galaxy merger time for the given parameters.

        Parameters
        ----------
        mass : (N,) array_like[scalar]
            Mass of the system, units of [grams].
            NOTE: the definition of mass is ambiguous, i.e. whether it is the primary mass, or the
            combined system mass.
        mrat : scalar or ndarray,
            Mass-ratio of the system (m2/m1 <= 1.0), dimensionless.
        redz : scalar or ndarray,
            Redshift.

        Returns
        -------
        rv : scalar or ndarray,
            Galaxy merger time, in units of [sec].

        """
        return

    def zprime(self, mass, mrat, redz, **kwargs):
        """Return the redshift after merger (i.e. input `redz` delayed by merger time).
        """
        tau0 = self(mass, mrat, redz, **kwargs)  # sec
        # Find the redshift of  t(z) + tau
        redz_prime = utils.redz_after(tau0, redz=redz)
        return redz_prime, tau0


class GMT_Power_Law(_Galaxy_Merger_Time):
    """Galaxy Merger Time - simple power law prescription
    """

    def __init__(self, time_norm=0.55*GYR, mref0=1.0e11*MSOL, malpha=0.0, zbeta=-0.5, qgamma=0.0):
        # tau0  [sec]
        self._time_norm = time_norm   # +0.55  b/t [+0.1, +2.0]  [+0.1, +10.0]  values for [Gyr]
        self._malpha = malpha         # +0.0   b/t [-0.2, +0.2]  [-0.2, +0.2 ]
        self._zbeta = zbeta           # -0.5   b/t [-2.0, +1.0]  [-3.0, +1.0 ]
        self._qgamma = qgamma         # +0.0   b/t [-0.2, +0.2]  [-0.2, +0.2 ]

        # [Msol]  NOTE: this is `b * M_0 = 0.4e11 Msol / h0` in [Chen2019]_
        # 7.2e10*MSOL
        mref = mref0 * (0.4 / cosmo.h)
        self._mref = mref
        return

    def __call__(self, mass, mrat, redz):
        """Return the galaxy merger time for the given parameters.

        Parameters
        ----------
        mass : (N,) array_like[scalar]
            Mass of the system, units of [grams].
            NOTE: the definition of mass is ambiguous, i.e. whether it is the primary mass, or the
            combined system mass.
        mrat : (N,) array_like[scalar]
            Mass ratio of each binary.
        redz : (N,) array_like[scalar]
            Redshifts of each binary.

        Returns
        -------
        mtime : (N,) ndarray[float]
            Merger time for each binary in [sec].

        """
        # convert to primary mass
        # mpri = utils.m1m2_from_mtmr(mtot, mrat)[0]   # [grams]
        tau0 = self._time_norm                       # [sec]
        bm0 = self._mref                             # [grams]
        aa = self._malpha
        bb = self._zbeta
        gg = self._qgamma
        mtime = tau0 * np.power(mass/bm0, aa) * np.power(1.0 + redz, bb) * np.power(mrat, gg)
        mtime = mtime
        return mtime
