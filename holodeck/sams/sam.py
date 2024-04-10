r"""Semi Analytic Modeling (SAM) submodule.

The core element of the SAM module is the :class:`Semi_Analytic_Model` class.  This class requires four
components as arguments:

(1) Galaxy Stellar Mass Function (GSMF): gives the comoving number-density of galaxies as a function
    of stellar mass.  This is implemented as subclasses of the :class:`_Galaxy_Stellar_Mass_Function`
    base class.
(2) Galaxy Pair Fraction (GPF): gives the fraction of galaxies that are in a 'pair' with a given
    mass ratio (and typically a function of redshift and primary-galaxy mass).  Implemented as
    subclasses of the :class:`_Galaxy_Pair_Fraction` subclass.
(3) Galaxy Merger Time (GMT): gives the characteristic time duration for galaxy 'mergers' to occur.
    Implemented as subclasses of the :class:`_Galaxy_Merger_Time` subclass.
(4) M_bh - M_bulge Relation (mmbulge): gives MBH properties for a given galaxy stellar-bulge mass.
    Implemented as subcalsses of the :class:`holodeck.host_relations._MMBulge_Relation` subclass.

The :class:`Semi_Analytic_Model` class defines a grid in parameter space of total MBH mass ($M=M_1 + M_2$),
MBH mass ratio ($q \\equiv M_1/M_2$), redshift ($z$), and at times binary separation
(semi-major axis $a$) or binary rest-frame orbital-frequency ($f_r$).  Over this grid, the distribution of
comoving number-density of MBH binaries in the Universe is calculated.  Methods are also provided
that interface with the `kalepy` package to draw 'samples' (discretized binaries) from the
distribution, and to calculate GW signatures.

The step of going from a number-density of binaries in $(M, q, z)$ space, to also the distribution
in $a$ or $f$ is subtle, as it requires modeling the binary evolution (i.e. hardening rate).


To-Do (sam.py)
--------------
* Allow SAM class to take M-sigma in addition to M-Mbulge.

References
----------
* [Sesana2008]_ Sesana, Vecchio, Colacino 2008.
* [Chen2019]_ Chen, Sesana, Conselice 2019.

"""

from datetime import datetime

import numpy as np
import scipy as sp
import scipy.interpolate  # noqa

import kalepy as kale

import holodeck as holo
from holodeck import cosmo, utils, log
from holodeck.constants import SPLC, MSOL, MPC
from holodeck import host_relations, single_sources
from holodeck.sams.components import (
    _Galaxy_Pair_Fraction, _Galaxy_Stellar_Mass_Function, _Galaxy_Merger_Time, _Galaxy_Merger_Rate,
    GSMF_Schechter, GPF_Power_Law, GMT_Power_Law, GMR_Illustris
)

REDZ_SAMPLE_VOLUME = True    #: get redshifts by sampling uniformly in 3D spatial volume, and converting

GSMF_USES_MTOT = False       #: the mass used in the GSMF is interpretted as M=m1+m2, otherwise use primary m1
GPF_USES_MTOT = False        #: the mass used in the GPF  is interpretted as M=m1+m2, otherwise use primary m1
GMT_USES_MTOT = False        #: the mass used in the GMT  is interpretted as M=m1+m2, otherwise use primary m1


# ===================================
# ====    Semi-Analytic Model    ====
# ===================================


class Semi_Analytic_Model:
    """Semi-Analytic Model (SAM) of MBH Binary populations.

    This class produces simulated MBH binary populations using idealized (semi-)analytic functions
    starting from galaxy populations, to massive black holes, to merger rates.  Using SAMs, MBH
    binary populations are calculated over a fixed, rectilinear grid of 3 or 4 dimensional
    parameter-space.  The starting parameter space is total mass, mass ratio, and redshift; but
    often the distribution of binaries are desired at particular orbital frequencies or separations,
    which adds a dimension.  Some parameters are calculated at grid edges (e.g. binary number
    densities), while others are calculated at grid centers (e.g. number of binaries in a universe).
    Ultimately, what the SAM calculates in the number (or number-density) of binaries at each point
    in this 3-4 dimensional parameter space.

    Conceptually, three components are required to build SAMs.

    (1) Galaxies and stellar-masses (i.e. how many galaxies there are as a function of stellar mass
        and redshift).  This component is provided by the Galaxy Stellar-Mass Function (GSMF), which
        are implemented as subclasses of the
        :class:`holodeck.sams.components._Galaxy_Stellar_Mass_Function` base-class.

    (2) Galaxy merger rates (GMRs; i.e. how often galaxies merge as a function of stellar mass, mass
        ratio, and redshift.)  This component can be provided in two ways:

        * Subclasses of :class:`holodeck.sams.components._Galaxy_Merger_Rate`, which provide merger
          rates directly.

        * Both a galaxy pair fraction (GPF; subclasses of
          :class:`holodeck.sams.components._Galaxy_Pair_Fraction`) which give the fraction of
          galaxies in the process of merger, and a galaxy merger time (GMT; subclasses of
          :class:`_Galaxy_Merger_Time`) which gives the duration of time that galaxies spend in the
          merger process.

    (3) MBH-Host relationships which determine MBH properties for a given host galaxy.  Currently
        these relationships are only fully implemented as Mbh-MBulge (MMBulge) relationships, which
        are subclasses of :class:`holodeck.host_relations._MMBulge_Relation`.

    """

    def __init__(
        self,
        mtot=(1.0e4*MSOL, 1.0e12*MSOL, 91),
        mrat=(1e-3, 1.0, 81),
        redz=(1e-3, 10.0, 101),
        shape=None,
        log=None,
        gsmf=GSMF_Schechter,
        gpf=None,
        gmt=None,
        gmr=None,
        mmbulge=host_relations.MMBulge_KH2013,
        **kwargs
    ):
        """Construct a new ``Semi_Analytic_Model`` instance.

        Parameters
        ----------
        mtot : (3,) tuple
            Specification for the domain of the grid in total-mass.
            Three arguments must be included, the 0th giving the lower-limit [grams], the 1th
            giving the upper-limit [grams], and the 2th giving the number of bin-edges (i.e. the
            number-of-bins plus one).
        mrat : (3,) tuple
            Specification for the domain of the grid in mass-ratio.
            Three arguments must be included, the 0th giving the lower-limit, the 1th giving the
            upper-limit, and the 2th giving the number of bin-edges (i.e. the number-of-bins plus
            one).
        redz : (3,) tuple
            Specification for the domain of the grid in redshift.
            Three arguments must be included, the 0th giving the lower-limit, the 1th giving the
            upper-limit, and the 2th giving the number of bin-edges (i.e. the number-of-bins plus
            one).
        shape : int  or  (3,) tuple
            The shape of the grid in total-mass, mass-ratio, and redshift.  This argument specifies
            the number of grid-edges in each dimension, and overrides the shape arguments of
            ``mtot``, ``mrat``, and ``redz``.
            * If a single `int` is given, then this is the shape applied to all dimensions.
            * If a (3,) iterable of values is given, then each value specifies the size of the grid
              in the corresponding dimension.  `None` values can be provided which indicate to use
              the default sizes (provided by the ``mtot``, ``mrat``, and ``redz`` arguments.)  For
              example, ``shape=(12, None, 14)`` would produce 12 grid edges in total mass, the
              default number of grid edges in mass ratio, and 14 grid edges in redshit.
        gsmf : None  or  :class:`_Galaxy_Stellar_Mass_Function` subclass instance
        gpf : None  or  :class:`_Galaxy_Pair_Fraction` subclass instance
        gmt : None  or  :class:`_Galaxy_Merger_Time` subclass instance
        gmr : None  or  :class:`_Galaxy_Merger_Rate` subclass instance
        mmbulge : None  or  :class:`_MMBulge_Relation` subclass instance

        """
        if log is None:
            log = holo.log
        self._log = log

        deprecated_keys = ['ZERO_DYNAMIC_STALLED_SYSTEMS', 'ZERO_GMT_STALLED_SYSTEMS']
        for key, val in kwargs.items():
            if key in deprecated_keys:
                log.error(f"Using deprecated kwarg: {key}: {val}!  In the future this will raise an error.")
            else:
                err = f"Unexpected kwarg {key=}: {val=}!"
                log.exception(err)
                raise ValueError(err)

        # ---- Process SAM components

        gsmf = utils.get_subclass_instance(gsmf, None, _Galaxy_Stellar_Mass_Function)
        mmbulge = utils.get_subclass_instance(mmbulge, None, host_relations._MMBulge_Relation)
        # if GMR is None, then we need both GMT and GPF
        if gmr is None:
            gmt = utils.get_subclass_instance(gmt, GMT_Power_Law, _Galaxy_Merger_Time)
            gpf = utils.get_subclass_instance(gpf, GPF_Power_Law, _Galaxy_Pair_Fraction)
        # if GMR is given, GMT can still be used - for calculating stalling
        else:
            gmr = utils.get_subclass_instance(gmr, GMR_Illustris, _Galaxy_Merger_Rate)
            gmt = utils.get_subclass_instance(gmt, None, _Galaxy_Merger_Time, allow_none=True)
            # if GMR is given, GPF is not used: make sure it is not given
            if (gpf is not None):
                err = f"When `GMR` ({gmr}) is provided, do not provide a GPF!"
                log.exception(err)
                raise ValueError(err)

        self._gsmf = gsmf             #: Galaxy Stellar-Mass Function (`_Galaxy_Stellar_Mass_Function` instance)
        self._mmbulge = mmbulge       #: Mbh-Mbulge relation (`host_relations._MMBulge_Relation` instance)
        self._gpf = gpf               #: Galaxy Pair Fraction (`_Galaxy_Pair_Fraction` instance)
        self._gmt = gmt               #: Galaxy Merger Time (`_Galaxy_Merger_Time` instance)
        self._gmr = gmr               #: Galaxy Merger Rate (`_Galaxy_Merger_Rate` instance)
        log.debug(f"{gsmf=}, {gmr=}, {gpf=}, {gmt=}, {mmbulge=}")

        # ---- Create SAM grid edges

        if shape is not None:
            if np.isscalar(shape):
                shape = [shape for ii in range(3)]

        params = [mtot, mrat, redz]
        param_names = ['mtot', 'mrat', 'redz']
        for ii, (par, name) in enumerate(zip(params, param_names)):
            if not isinstance(par, tuple) and (len(par) == 3):
                err = (
                    f"{name} (type={type(par)}, len={len(par)}) must be a (3,) tuple specifying a log-spacing, "
                    "or ndarray of grid edges!"
                )
                log.exception(err)
                raise ValueError(err)

            par = [pp for pp in par]
            if shape is not None:
                if shape[ii] is not None:
                    par[2] = shape[ii]
            params[ii] = np.logspace(*np.log10(par[:2]), par[2])
            log.debug(f"{name}: [{params[ii][0]}, {params[ii][-1]}] {params[ii].size}")

        mtot, mrat, redz = params
        self.mtot = mtot
        self.mrat = mrat
        self.redz = redz

        # ---- Set other parameters

        # These values are calculated as needed by the class when the corresponding methods are called
        self._density = None          #: Binary comoving number-density
        self._shape = None            #: Shape of the parameter-space domain (mtot, mrat, redz)
        self._gmt_time = None         #: GMT timescale of galaxy mergers [sec]
        self._redz_prime = None       #: redshift following galaxy merger process

        return

    @property
    def edges(self):
        """The grid edges defining the domain (list of: [`mtot`, `mrat`, `redz`])
        """
        return [self.mtot, self.mrat, self.redz]

    @property
    def shape(self):
        """Shape of the parameter space domain (number of edges in each dimension), (3,) tuple
        """
        if self._shape is None:
            self._shape = tuple([len(ee) for ee in self.edges])
        return self._shape

    def mass_stellar(self):
        """Calculate stellar masses for each MBH based on the M-MBulge relation.

        Returns
        -------
        masses : (2, N) ndarray of scalar,
            Galaxy total stellar masses for all MBH. [0, :] is primary, [1, :] is secondary [grams].

        """
        redz = self.redz[np.newaxis, np.newaxis, :]
        # total-mass, mass-ratio ==> (M1, M2)
        masses = utils.m1m2_from_mtmr(self.mtot[:, np.newaxis], self.mrat[np.newaxis, :])
        # BH-masses to stellar-masses
        mbh_pri = masses[0]
        mbh_sec = masses[1]
        args = [mbh_pri[..., np.newaxis], mbh_sec[..., np.newaxis], redz]
        # Convert to shape (M, Q, Z)
        mbh_pri, mbh_sec, redz = np.broadcast_arrays(*args)
        mstar_pri = self._mmbulge.mstar_from_mbh(mbh_pri, redz=redz, scatter=False)
        mstar_sec = self._mmbulge.mstar_from_mbh(mbh_sec, redz=redz, scatter=False)

        # q = m2 / m1
        mstar_rat = mstar_sec / mstar_pri
        # M = m1 + m2
        mstar_tot = mstar_pri + mstar_sec
        # args = [mstar_rat[..., np.newaxis]]
        # # Convert to shape (M, Q, Z)
        # mstar_rat = np.broadcast_arrays(*args)
        return mstar_pri, mstar_rat, mstar_tot, redz

    @property
    def static_binary_density(self):
        r"""The number-density of binaries at the edges of the grid in mtot, mrat, redz.

        The 'number density' is a density both in terms of volume (i.e. number of binaries per unit
        comoving-volume, $n = dN/dV_c$), and in terms of binary parameters (e.g. binaries per unit
        of log10 mass, $d n /d \log_{10} M$).  Specifically, the values returned are:

        .. math::
            d^3 n / [d \log_{10} M  d q  d z]

        For each :class:`Semi_Analytic_Model` instance, this value is calculated once and cached.

        Returns
        -------
        density : (M, Q, Z) ndarray [$cMpc^{-3}$]
            Number density of binaries, per unit redshift, mass-ratio, and log10 of mass.
            The values are in units of inverse cubic, comoving-Mpc.

        Notes
        -----
        * This function effectively calculates Eq.21 & 5 of [Chen2019]_; or equivalently, Eq. 6 of [Sesana2008]_.
        * Bins which 'merge' after redshift zero are set to zero density (using the `self._gmt` instance).

        """
        if self._density is None:
            log = self._log

            # ---- convert from MBH ===> mstar

            redz = self.redz[np.newaxis, np.newaxis, :]
            mstar_pri, mstar_rat, mstar_tot, redz = self.mass_stellar()

            # choose whether the primary mass, or total mass, is used in different calculations
            mass_gsmf = mstar_tot if GSMF_USES_MTOT else mstar_pri

            # ---- find galaxy-merger duration and redshift after merger

            if self._gmt is not None:
                log.debug(f"{GMT_USES_MTOT=}")
                mass_gmt = mstar_tot if GMT_USES_MTOT else mstar_pri

                # GMT returns `-1.0` for values beyond age of universe
                zprime, gmt_time = self._gmt.zprime(mass_gmt, mstar_rat, redz)
                self._gmt_time = gmt_time
                self._redz_prime = zprime

                # find valid entries (M, Q, Z)
                idx_stalled = (zprime < 0.0)
                # log.debug(f"Stalled SAM bins based on GMT: {utils.frac_str(idx_stalled)}")
            else:
                log.info("No GMT was provided, cannot calculate Galaxy-Merger based stalling.")
                idx_stalled = None

            # ---- get galaxy merger rate

            if self._gmr is None:
                log.debug("Calculating galaxy merger rate using pair-fraction (GPF) and merger-time (GMT)")
                log.debug(f"GPF_USES_MTOT ={GPF_USES_MTOT}")
                mass_gpf = mstar_tot if GPF_USES_MTOT else mstar_pri
                # `gmt` returns [sec]  `gpf` is dimensionless,  so this is [1/sec]
                gal_merger_rate = self._gpf(mass_gpf, mstar_rat, redz) / gmt_time
            else:
                log.debug("Calculating galaxy merger rate directly from GMR")
                gal_merger_rate = self._gmr(mstar_tot, mstar_rat, redz)

            # `gsmf` returns [1/Mpc^3]   `dtdz` returns [sec]   `gal_merger_rate` is [1/sec]  ===>  [Mpc^-3]
            dens = self._gsmf(mass_gsmf, redz) * gal_merger_rate * cosmo.dtdz(redz)

            # ---- Convert to MBH Binary density

            # we want ``dn_mbhb / [dlog10(M_bh) dq_bh qz]``
            # so far we have ``dn_gal / [dlog10(M_gal) dq_gal dz]``

            # dn / [dM dq dz] = (dn_gal / [dM_gal dq_gal dz]) * (dM_gal/dM_bh) * (dq_gal / dq_bh)
            mplaw = self._mmbulge._mplaw
            dqbh_dqgal = mplaw * np.power(mstar_rat, mplaw - 1.0)
            # (dMstar-pri / dMbh-pri) * (dMbh-pri/dMbh-tot) = (dMstar-pri / dMstar-tot) * (dMstar-tot/dMbh-tot)
            # ==> (dMstar-tot/dMbh-tot) = (dMstar-pri / dMbh-pri) * (dMbh-pri/dMbh-tot) / (dMstar-pri / dMstar-tot)
            #                           = (dMstar-pri / dMbh-pri) * (1 / (1+q_bh)) / (1 / (1+q_star))
            #                           = (dMstar-pri / dMbh-pri) * ((1+q_star) / (1+q_bh))
            dmstar_dmbh_pri = self._mmbulge.dmstar_dmbh(mstar_pri)   # [unitless]
            qterm = (1.0 + mstar_rat) / (1.0 + self.mrat[np.newaxis, :, np.newaxis])
            dmstar_dmbh = dmstar_dmbh_pri * qterm

            dens *= (self.mtot[:, np.newaxis, np.newaxis] / mstar_tot) * (dmstar_dmbh / dqbh_dqgal)

            # ---- Add scatter from the M-Mbulge relation

            scatter = self._mmbulge._scatter_dex
            log.debug(f"mmbulge scatter = {scatter}")
            if scatter > 0.0:
                log.info(f"Adding MMbulge scatter ({scatter:.4e})")
                log.info(f"\tdens bef: ({utils.stats(dens)})")
                dur = datetime.now()
                mass_bef = self._integrated_binary_density(dens, sum=True)
                self._dens_bef = np.copy(dens)
                dens = add_scatter_to_masses(self.mtot, self.mrat, dens, scatter, log=log)
                self._dens_aft = np.copy(dens)

                mass_aft = self._integrated_binary_density(dens, sum=True)
                dur = datetime.now() - dur
                dm = (mass_aft - mass_bef) / mass_bef
                log.info(f"Scatter added after {dur.total_seconds()} sec")
                log.info(f"\tdens aft: ({utils.stats(dens)})")
                msg = f"mass: {mass_bef:.2e} ==> {mass_aft:.2e} || change = {dm:.4e}"
                log.info(f"\t{msg}")
                if np.fabs(dm) > 0.2:
                    err = f"Warning, significant change in number-mass!  {msg}"
                    log.error(err)

            # set values after redshift zero to have zero density
            if idx_stalled is not None:
                log.info(f"zeroing out {utils.frac_str(idx_stalled)} bins stalled from GMT")
                dens[idx_stalled] = 0.0

            self._density = dens

        return self._density

    def dynamic_binary_number_at_fobs(self, hard, fobs_orb, **kwargs):

        if hard.CONSISTENT:
            edges, dnum, redz_final = self._dynamic_binary_number_at_fobs_consistent(hard, fobs_orb, **kwargs)
        else:
            edges, dnum, redz_final = self._dynamic_binary_number_at_fobs_inconsistent(hard, fobs_orb, **kwargs)

        return edges, dnum, redz_final

    def _dynamic_binary_number_at_fobs_consistent(self, hard, fobs_orb, steps=200, details=False):
        """Get correct redshifts for full binary-number calculation.

        Slower but more correct than old `dynamic_binary_number`.
        Same as new cython implementation `sam_cyutils.dynamic_binary_number_at_fobs`, which is
        more than 10x faster.
        LZK 2023-05-11

        # BUG doesn't work for Fixed_Time_2PL

        """
        fobs_orb = np.asarray(fobs_orb)
        edges = self.edges + [fobs_orb, ]

        # shape: (M, Q, Z)
        dens = self.static_binary_density   # d3n/[dlog10(M) dq dz]  units: [Mpc^-3]


        # start from the hardening model's initial separation
        rmax = hard._sepa_init
        # (M,) end at the ISCO
        rmin = utils.rad_isco(self.mtot)
        # Choose steps for each binary, log-spaced between rmin and rmax
        extr = np.log10([rmax * np.ones_like(rmin), rmin])     # (2,M,)
        rads = np.linspace(0.0, 1.0, steps)[np.newaxis, :]     # (1,X)
        # (M, S)  =  (M,1) * (1,S)
        rads = extr[0][:, np.newaxis] + (extr[1] - extr[0])[:, np.newaxis] * rads
        rads = 10.0 ** rads

        # (M, Q, S)
        mt, mr, rads, norm = np.broadcast_arrays(
            self.mtot[:, np.newaxis, np.newaxis],
            self.mrat[np.newaxis, :, np.newaxis],
            rads[:, np.newaxis, :],
            hard._norm[:, :, np.newaxis],
        )
        dadt_evo = hard.dadt(mt, mr, rads, norm=norm)

        # (M, Q, S-1)
        # Integrate (inverse) hardening rates to calculate total lifetime to each separation
        times_evo = -utils.trapz_loglog(-1.0 / dadt_evo, rads, axis=-1, cumsum=True)
        # Combine the binary-evolution time, with the galaxy-merger time
        # (M, Q, Z, S-1)
        rz = self.redz[np.newaxis, np.newaxis, :, np.newaxis]
        times_tot = times_evo[:, :, np.newaxis, :] + self._gmt_time[:, :, :, np.newaxis]
        redz_evo = utils.redz_after(times_tot, redz=rz)

        # convert from separations to rest-frame orbital frequencies
        # (M, Q, S)
        frst_orb_evo = utils.kepler_freq_from_sepa(mt, rads)
        # (M, Q, Z, S)
        fobs_orb_evo = frst_orb_evo[:, :, np.newaxis, :] / (1.0 + rz)

        # ---- interpolate to target frequencies
        # `ndinterp` interpolates over 1th dimension

        # (M, Q, Z, S-1)  ==>  (M*Q*Z, S-1)
        fobs_orb_evo, redz_evo = [mm.reshape(-1, steps-1) for mm in [fobs_orb_evo[:, :, :, 1:], redz_evo]]
        # (M*Q*Z, X)
        redz_final = utils.ndinterp(fobs_orb, fobs_orb_evo, redz_evo, xlog=True, ylog=False)

        # (M*Q*Z, X) ===> (M, Q, Z, X)
        redz_final = redz_final.reshape(self.shape + (fobs_orb.size,))
        coal = (redz_final > 0.0)
        frst_orb = fobs_orb * (1.0 + redz_final)
        frst_orb[frst_orb < 0.0] = 0.0
        redz_final[~coal] = -1.0

        # (M, Q, Z, X) comoving-distance in [Mpc]
        dc = np.zeros_like(redz_final)
        dc[coal] = cosmo.comoving_distance(redz_final[coal]).to('Mpc').value

        # (M, Q, Z, X) this is `(dVc/dz) * (dz/dt)` in units of [Mpc^3/s]
        cosmo_fact = np.zeros_like(redz_final)
        cosmo_fact[coal] = 4 * np.pi * (SPLC/MPC) * np.square(dc[coal]) * (1.0 + redz_final[coal])

        # (M, Q) calculate chirp-mass
        mt = self.mtot[:, np.newaxis, np.newaxis, np.newaxis]
        mr = self.mrat[np.newaxis, :, np.newaxis, np.newaxis]

        # Convert from observer-frame orbital freq, to rest-frame orbital freq
        sa = utils.kepler_sepa_from_freq(mt, frst_orb)
        mt, mr, sa, norm = np.broadcast_arrays(mt, mr, sa, hard._norm[:, :, np.newaxis, np.newaxis])
        # hardening rate, negative values, units of [cm/sec]
        dadt = hard.dadt(mt, mr, sa, norm=norm)
        # Calculate `tau = dt/dlnf_r = f_r / (df_r/dt)`
        # dfdt is positive (increasing frequency)
        dfdt, frst_orb = utils.dfdt_from_dadt(dadt, sa, frst_orb=frst_orb)
        tau = frst_orb / dfdt

        # (M, Q, Z, X) units: [1/s] i.e. number per second
        dnum = dens[..., np.newaxis] * cosmo_fact * tau
        dnum[~coal] = 0.0

        if details:
            tau[~coal] = 0.0
            dadt[~coal] = 0.0
            sa[~coal] = 0.0
            cosmo_fact[~coal] = 0.0
            # (M, Q, X)  ==>  (M, Q, Z, X)
            dets = dict(tau=tau, cosmo_fact=cosmo_fact, dadt=dadt, fobs=fobs_orb, sepa=sa)
            return edges, dnum, redz_final, dets

        return edges, dnum, redz_final

    def _dynamic_binary_number_at_fobs_inconsistent(self, hard, fobs_orb):
        fobs_orb = np.asarray(fobs_orb)
        edges = self.edges + [fobs_orb, ]

        # shape: (M, Q, Z)
        dens = self.static_binary_density   # d3n/[dlog10(M) dq dz]  units: [Mpc^-3]
        shape = dens.shape
        new_shape = shape + (fobs_orb.size, )

        rz = self._redz_prime[..., np.newaxis] * np.ones(new_shape)
        coal = (rz > 0.0)

        dc = cosmo.comoving_distance(rz[coal]).to('Mpc').value
        frst_orb = utils.frst_from_fobs(
            fobs_orb[np.newaxis, np.newaxis, np.newaxis, :], rz
        )

        # (Z,) this is `(dVc/dz) * (dz/dt)` in units of [Mpc^3/s]
        cosmo_fact = 4 * np.pi * (SPLC/MPC) * np.square(dc) * (1.0 + rz[coal])

        # # (M, Q) calculate chirp-mass
        mt = self.mtot[:, np.newaxis, np.newaxis, np.newaxis]
        mr = self.mrat[np.newaxis, :, np.newaxis, np.newaxis]
        mt, mr = [(mm * np.ones(new_shape))[coal] for mm in [mt, mr]]

        # Convert from observer-frame orbital freq, to rest-frame orbital freq
        sa = utils.kepler_sepa_from_freq(mt, frst_orb[coal])
        # (X, M*Q*Z), hardening rate, negative values, units of [cm/sec]
        args = [mt, mr, sa]
        dadt = hard.dadt(*args)
        # Calculate `tau = dt/dlnf_r = f_r / (df_r/dt)`
        # dfdt is positive (increasing frequency)
        dfdt, _ = utils.dfdt_from_dadt(dadt, sa, frst_orb=frst_orb[coal])
        tau = frst_orb[coal] / dfdt

        # (M, Q, Z) units: [1/s] i.e. number per second
        dnum = np.zeros(new_shape)
        dnum[coal] = (dens[..., np.newaxis] * np.ones(new_shape))[coal] * cosmo_fact * tau

        return edges, dnum, rz

    def _dynamic_binary_number_at_sepa_consistent(self, hard, target_sepa, steps=200, details=False):
        """Get correct redshifts for full binary-number calculation.

        Slower but more correct than old `dynamic_binary_number`.
        Same as new cython implementation `sam_cyutils.dynamic_binary_number_at_fobs`, which is
        more than 10x faster.
        LZK 2023-05-11

        """
        target_sepa = np.asarray(target_sepa)
        ntarget = target_sepa.size    # this will be refered to as 'X' in shapes
        edges = self.edges + [target_sepa, ]
        nmtot, nmrat, nredz = self.shape

        # start from the hardening model's initial separation
        rmax = hard._sepa_init
        # (M,) end at the ISCO
        rmin = utils.rad_isco(self.mtot)
        # Choose steps for each binary, log-spaced between rmin and rmax
        extr = np.log10([rmax * np.ones_like(rmin), rmin])     # (2,M,)
        rads = np.linspace(0.0, 1.0, steps)[np.newaxis, :]     # (1,X)
        # (M, S)  =  (M,1) * (1,S)
        rads = extr[0][:, np.newaxis] + (extr[1] - extr[0])[:, np.newaxis] * rads
        rads = 10.0 ** rads

        # (M, Q, Z, S)
        norm = hard._norm
        mt, mr, rz, rads, norm = np.broadcast_arrays(
            self.mtot[:, np.newaxis, np.newaxis, np.newaxis],
            self.mrat[np.newaxis, :, np.newaxis, np.newaxis],
            self.redz[np.newaxis, np.newaxis, :, np.newaxis],
            rads[:, np.newaxis, np.newaxis, :],
            norm[:, :, np.newaxis, np.newaxis]
        )

        # (M, Q, Z, S)  ==>  (M, Q, Z*S)
        mt, mr, rz, rads, norm = [mm.reshape(nmtot, nmrat, -1) for mm in [mt, mr, rz, rads, norm]]
        dadt_evo = hard.dadt(mt, mr, rads, norm=norm)

        # (M, Q, Z*S)  ==>  (M, Q, Z, S)
        dadt_evo, rz, rads = [mm.reshape(nmtot, nmrat, nredz, steps) for mm in [dadt_evo, rz, rads]]
        # dadt_evo = dadt_evo.reshape(nmtot, nmrat, nredz, steps)

        # Integrate (inverse) hardening rates to calculate total lifetime to each separation
        times_evo = -utils.trapz_loglog(-1.0 / dadt_evo, rads, axis=-1, cumsum=True)
        # Combine the binary-evolution time, with the galaxy-merger time
        times_tot = times_evo + self._gmt_time[:, :, :, np.newaxis]
        redz_evo = utils.redz_after(times_tot, redz=rz[:, :, :, 1:])

        # ---- interpolate to target frequencies

        # `ndinterp` interpolates over axis=1,  so get steps (S,) and target radii (X,) to axis=1

        # get our target separations in the appropriate shape to match evolution arrays
        # (X,)  ==>  (M, Q, Z, X)
        sepa = target_sepa[np.newaxis, np.newaxis, np.newaxis, :] * np.ones(self.shape)[..., np.newaxis]
        # (M, Q, Z, X)  ==>  (M*Q*Z, X)
        sepa = sepa.reshape(-1, target_sepa.size)
        # (M, Q, Z, S-1) ==>  (M*Q*Z, S-1)
        rads, redz_evo = [mm.reshape(-1, steps-1) for mm in [rads[:, :, :, 1:], redz_evo]]

        # `rads` MUST BE INCREASING for interpolation, so reverse the steps
        rads = rads[:, ::-1]
        redz_evo = rads[:, ::-1]
        redz_final = utils.ndinterp(sepa, rads, redz_evo, xlog=True, ylog=False)

        # (M*Q*Z, X) ===> (M, Q, Z, X)
        redz_final = redz_final.reshape(self.shape + (ntarget,))
        coal = (redz_final > 0.0)
        redz_final[~coal] = -1.0

        # shape: (M, Q, Z)
        dens = self.static_binary_density   # d3n/[dlog10(M) dq dz]  units: [Mpc^-3]

        # (M, Q, Z, X) comoving-distance in [Mpc]
        dc = np.zeros_like(redz_final)
        dc[coal] = cosmo.comoving_distance(redz_final[coal]).to('Mpc').value

        # (M, Q, Z, X) this is `(dVc/dz) * (dz/dt)` in units of [Mpc^3/s]
        cosmo_fact = np.zeros_like(redz_final)
        cosmo_fact[coal] = 4 * np.pi * (SPLC/MPC) * np.square(dc[coal]) * (1.0 + redz_final[coal])

        # ---- Calculate timescale `tau = dt/dlnf_r = f_r / (df_r/dt)`

        mt, mr, sepa, norm = np.broadcast_arrays(
            self.mtot[:, np.newaxis, np.newaxis],
            self.mrat[np.newaxis, :, np.newaxis],
            target_sepa[np.newaxis, np.newaxis, :],
            hard._norm[:, :, np.newaxis],
        )
        # hardening rate, negative values, units of [cm/sec]
        dadt = hard.dadt(mt, mr, sepa, norm=norm)
        # dfdt is positive (increasing frequency)
        dfdt, frst_orb = utils.dfdt_from_dadt(dadt, sepa, mtot=mt)
        tau = frst_orb / dfdt

        # (M, Q, Z) units: [1/s] i.e. number per second
        dnum = dens[..., np.newaxis] * cosmo_fact * tau[:, :, np.newaxis, :]
        dnum[~coal] = 0.0

        if details:
            # (M, Q, X)  ==>  (M, Q, Z, X)
            tau = tau[:, :, np.newaxis, :] * np.ones_like(redz_final)
            dadt = dadt[:, :, np.newaxis, :] * np.ones_like(redz_final)
            dets = dict(tau=tau, cosmo_fact=cosmo_fact, dadt=dadt, sepa=target_sepa)
            return edges, dnum, redz_final, dets

        return edges, dnum, redz_final

    def gwb_new(self, fobs_gw_edges, hard=holo.hardening.Hard_GW(), realize=100):
        """Calculate GWB using new cython implementation, 10x faster!
        """
        from . import sam_cyutils

        assert isinstance(hard, (holo.hardening.Fixed_Time_2PL_SAM, holo.hardening.Hard_GW))

        fobs_gw_cents = kale.utils.midpoints(fobs_gw_edges)

        # convert to orbital-frequency (from GW-frequency)
        fobs_orb_edges = fobs_gw_edges / 2.0
        fobs_orb_cents = fobs_gw_cents / 2.0

        # ---- Calculate number of binaries in each bin

        redz_final, diff_num = sam_cyutils.dynamic_binary_number_at_fobs(
            fobs_orb_cents, self, hard, cosmo
        )

        edges = [self.mtot, self.mrat, self.redz, fobs_orb_edges]
        number = sam_cyutils.integrate_differential_number_3dx1d(edges, diff_num)

        # ---- Get the GWB spectrum from number of binaries over grid

        gwb = holo.gravwaves._gws_from_number_grid_integrated_redz(edges, redz_final, number, realize)

        return gwb

    def gwb_old(self, fobs_gw_edges, hard=holo.hardening.Hard_GW, realize=100):
        """Calculate GWB using new `dynamic_binary_number_at_fobs` method, better, but slower.
        """

        fobs_gw_edges = np.atleast_1d(fobs_gw_edges)
        fobs_gw_cents = kale.utils.midpoints(fobs_gw_edges)
        # convert to orbital-frequency (from GW-frequency)
        fobs_orb_edges = fobs_gw_edges / 2.0
        fobs_orb_cents = fobs_gw_cents / 2.0

        edges, dnum, redz_final = self.dynamic_binary_number_at_fobs(hard, fobs_orb_cents)
        edges[-1] = fobs_orb_edges

        number = utils._integrate_grid_differential_number(edges, dnum, freq=False)
        number = number * np.diff(np.log(fobs_gw_edges))

        # ---- Get the GWB spectrum from number of binaries over grid

        gwb = holo.gravwaves._gws_from_number_grid_integrated_redz(edges, redz_final, number, realize)

        return gwb

    def gwb_ideal(self, fobs_gw, sum=True, redz_prime=None):
        """Calculate the idealized, continuous GWB amplitude.

        Calculation follows [Phinney2001]_ (Eq.5) or equivalently [Enoki+Nagashima-2007] (Eq.3.6).
        This calculation assumes a smooth, continuous population of binaries that are purely GW driven.
        * There are no finite-number effects.
        * There are no environmental or non-GW driven evolution effects.
        * There is no coalescence of binaries cutting them off at high-frequencies.

        """
        redz = self.redz[np.newaxis, np.newaxis, :]
        mstar_pri, mstar_rat, mstar_tot, redz = self.mass_stellar()

        # default to using `redz_prime` values if a GMT instance is stored
        if redz_prime is None:
            redz_prime = (self._gmt is not None)
        elif redz_prime and (self._gmt is None):
            err = "No `GMT` instance stored, cannot use `redz_prime` values!"
            self._log.exception(err)
            raise AttributeError(err)

        rz = self.redz
        if redz_prime:
            gmt_mass = mstar_tot if GMT_USES_MTOT else mstar_pri
            rz, _ = self._gmt.zprime(gmt_mass, mstar_rat, rz)
            print(f"{self} :: {utils.stats(rz)=}")

        # d^3 n / [dlog10(M) dq dz] in units of [Mpc^-3], convert to [cm^-3]
        ndens = self.static_binary_density / (MPC**3)

        mt = self.mtot[:, np.newaxis, np.newaxis]
        mr = self.mrat[np.newaxis, :, np.newaxis]
        gwb = holo.gravwaves.gwb_ideal(fobs_gw, ndens, mt, mr, rz, dlog10=True, sum=sum)
        return gwb

    def gwb(self, fobs_gw_edges, hard=holo.hardening.Hard_GW(), realize=100, loudest=1, params=False):
        """Calculate the (smooth/semi-analytic) GWB and CWs at the given observed GW-frequencies.

        Parameters
        ----------
        fobs_gw_edges : (F,) array_like of scalar,
            Observer-frame GW-frequencies. [1/sec]
            These are the frequency bin edges, which are integrated across to get the number of binaries in each
            frequency bin.
        hard : holodeck.evolution._Hardening class or instance
            Hardening mechanism to apply over the range of `fobs_gw`.
        realize : int
            Specification of how many discrete realizations to construct.
            Realizations approximate the finite-source effects of a realistic population.
        loudest : int
            Number of loudest single sources to distinguish from the background.
        params : Boolean
            Whether or not to return astrophysical parameters of the binaries.

        Returns
        -------
        hc_ss : (F, R, L) NDarray of scalars
            The characteristic strain of the L loudest single sources at each frequency.
        hc_bg : (F, R) NDarray of scalars
            Characteristic strain of the GWB.
        sspar : (4, F, R, L) NDarray of scalars
            Astrophysical parametes (total mass, mass ratio, initial redshift, final redshift) of each
            loud single sources, for each frequency and realization.
            Returned only if params = True.
        bgpar : (7, F, R) NDarray of scalars
            Average effective binary astrophysical parameters (total mass, mass ratio, initial redshift,
            final redshift, final comoving distance, final separation, final angular separation)
            for background sources at each frequency and realization,
            Returned only if params = True.

        """
        from . import sam_cyutils

        if not isinstance(hard, (holo.hardening.Fixed_Time_2PL_SAM, holo.hardening.Hard_GW)):
            err = (
                "`sam_cyutils` methods only work with `Fixed_Time_2PL_SAM` or `Hard_GW` hardening models!  "
                "Use `gwb_only` for alternative classes!"
            )
            self._log.exception(err)
            raise ValueError(err)

        fobs_gw_cents = kale.utils.midpoints(fobs_gw_edges)

        # convert to orbital-frequency (from GW-frequency)
        fobs_orb_edges = fobs_gw_edges / 2.0
        fobs_orb_cents = fobs_gw_cents / 2.0

        # ---- Calculate number of binaries in each bin

        redz_final, diff_num = sam_cyutils.dynamic_binary_number_at_fobs(
            fobs_orb_cents, self, hard, cosmo
        )

        edges = [self.mtot, self.mrat, self.redz, fobs_orb_edges]
        number = sam_cyutils.integrate_differential_number_3dx1d(edges, diff_num)

        # ---- Get the Single Source and GWB spectrum from number of binaries over grid

        ret_vals = single_sources.ss_gws_redz(edges, redz_final, number,
                                              realize=realize, loudest=loudest, params=params)

        hc_ss = ret_vals[0]
        hc_bg = ret_vals[1]
        if params:
            sspar = ret_vals[2]
            bgpar = ret_vals[3]

        if params:
            return hc_ss, hc_bg, sspar, bgpar

        return hc_ss, hc_bg

    def rate_chirps(self, hard=None, integrate=True):
        """Find the event rate of binary coalescences ('chirps').

        Get the number of coalescence events per unit time, in units of [1/sec].

        Parameters
        ----------
        hard : None  or  `_Hardening` subclass instance
        integrate : bool

        Returns
        -------
        redz_final : (M, Q, Z)
            Redshift of binary coalescence.  Binaries stalling before `z=0`, have values set to
            `-1.0`.
        rate : ndarray
            Rate of coalescence events in each bin, in units of [1/sec].
            The shape and meaning depends on the value of the `integrate` flag:

            * if `integrate == True`,
              then the returned values is ``dN/dt``, with shape (M-1, Q-1, Z-1)

            * if `integrate == False`,
              then the returned values is ``dN/[dlog10M dq dz dt]``, with shape (M, Q, Z)

        """
        log = self._log

        if hard is not None:
            if not isinstance(hard, (holo.hardening.Fixed_Time_2PL_SAM,)):
                err = "Only the `Fixed_Time` models, or no hardening, are supported for rate calculation!"
                log.exception(err)
                raise ValueError(err)

        # get number density of binaries dn/[dlog10M dq dz] in units of [Mpc^-3]
        # NOTE: `static_binary_density` must be called for `_gmt_time` to be set
        ndens = self.static_binary_density

        # ---- Get redshift of coalescence

        # Find initialization times (Z,) in [sec]
        time_init = cosmo.z_to_tage(self.redz)
        # NOTE: `static_binary_density` must be called for `_gmt_time` to be set
        # Galaxy Merger Time, time in [sec]  shape (M, Q, Z)
        time_gmt = self._gmt_time
        # if `sam` is using galaxy merger rate (GMR), then `gmt_time` will be `None`
        if time_gmt is None:
            log.info("`gmt_time` not calculated in SAM.  Setting to zeros.")
            time_gmt = np.zeros(self.shape)

        if hard is None:
            time_life = 0.0
        else:
            # Target lifetime of fixed_time model, single scalar value in [sec]
            time_life = hard._target_time

        time_tot = time_init[np.newaxis, np.newaxis, :] + time_gmt + time_life
        # when `time_tot` is greater than age of Universe, `redz_final` will be NaN
        redz_final = cosmo.tage_to_z(time_tot)
        # find bins where the binary coalesces before redshift zero (this includes checking that redz_final is not NaN)
        valid = (redz_final > 0.0)
        # set redshift final for stalled binaries to -1.0
        redz_final[~valid] = -1.0

        # ---- calculate coalescence properties

        mt, mr, _ = np.meshgrid(self.mtot, self.mrat, None, indexing='ij')
        m1, m2 = utils.m1m2_from_mtmr(mt, mr)
        mc = utils.chirp_mass_mtmr(mt, mr)

        # Place all binaries at the ISCO, find the corresponding frequency, strain, and characteristic strain
        risco = utils.rad_isco(mt)
        fisco_rst = utils.kepler_freq_from_sepa(mt, risco)
        fisco = fisco_rst / (1.0 + redz_final)

        dc = cosmo.z_to_dcom(redz_final)
        hs = utils.gw_strain_source(mc, dc, fisco_rst)
        dadt = utils.gw_hardening_rate_dadt(m1, m2, risco)
        dfdt, _ = utils.dfdt_from_dadt(dadt, risco, mtot=mt, frst_orb=fisco_rst)

        log.warning("!! assuming ncycles = f^2 / (dfdt) !!")
        ncycles = fisco_rst**2 / dfdt

        hc = np.sqrt(ncycles) * hs
        fisco[~valid] = np.nan

        # ---- calculate event rates

        # (M, Q, Z)
        rate = np.zeros(self.shape)
        rz = redz_final[valid]
        # get rest-frame dz/dt
        dzdt = 1.0 / cosmo.dtdz(rz)
        # get dVc/dz in units of [Mpc^3] to match `ndens` in units of [Mpc^-3]
        dVcdz = cosmo.dVcdz(rz, cgs=False)

        # factor of 1/1+z to convert from rest-frame to observer-frame time-interval
        rate[valid] = ndens[valid] * dzdt * dVcdz / (1.0 + rz)

        # integrate over each bin to go from ``dN/[dlog10M dq dz dt]`` to ``dN/dt``
        if integrate:
            rate = self._integrate_event_rate(rate)

        return redz_final, rate, fisco, hc

    def _integrate_event_rate(self, rate):
        # (Z-1,)
        dz = np.diff(self.redz)
        # perform 'integration', but don't sum over redshift bins
        # (M, Q, Z-1)
        integ = 0.5 * (rate[:, :, :-1] + rate[:, :, 1:]) * dz

        # ---- Integrate over mass and mass-ratio
        # (M-1,)
        dlogm = np.diff(np.log10(self.mtot))
        # (Q-1,)
        dq = np.diff(self.mrat)
        # (M-1, Q, Z-1)
        integ = 0.5 * (integ[:-1, :, :] + integ[1:, :, :]) * dlogm[:, np.newaxis, np.newaxis]
        # (M-1, Q-1, Z-1)
        integ = 0.5 * (integ[:, :-1, :] + integ[:, 1:, :]) * dq[np.newaxis, :, np.newaxis]

        return integ

    def _ndens_gal(self, mass_gal, mrat_gal, redz):
        if GSMF_USES_MTOT or GPF_USES_MTOT or GMT_USES_MTOT:
            self._log.warning("{self.__class__}._ndens_gal assumes that primary mass is used for GSMF, GPF and GMT!")

        # NOTE: dlog10(M_1) / dlog10(M) = (M/M_1) * (dM_1/dM) = 1
        nd = self._gsmf(mass_gal, redz) * self._gpf(mass_gal, mrat_gal, redz)
        nd = nd * cosmo.dtdz(redz) / self._gmt(mass_gal, mrat_gal, redz)
        return nd

    def _ndens_mbh(self, mass_gal, mrat_gal, redz):
        if GSMF_USES_MTOT or GPF_USES_MTOT or GMT_USES_MTOT:
            self._log.warning("{self.__class__}._ndens_mbh assumes that primary mass is used for GSMF, GPF and GMT!")

        # this is  d^3 n / [dlog10(M_gal-pri) dq_gal dz]
        nd_gal = self._ndens_gal(mass_gal, mrat_gal, redz)

        mplaw = self._mmbulge._mplaw
        dqbh_dqgal = mplaw * np.power(mrat_gal, mplaw - 1.0)

        dmstar_dmbh__pri = self._mmbulge.dmstar_dmbh(mass_gal)   # [unitless]
        mbh_pri = self._mmbulge.mbh_from_mstar(mass_gal, scatter=False)
        mbh_sec = self._mmbulge.mbh_from_mstar(mass_gal * mrat_gal, scatter=False)
        mbh = mbh_pri + mbh_sec
        mrat_mbh = mbh_sec / mbh_pri

        dlm_dlm = (mbh / mass_gal) * dmstar_dmbh__pri / (1.0 + mrat_mbh)
        dens = nd_gal * dlm_dlm / dqbh_dqgal
        return dens

    def _integrated_binary_density(self, ndens=None, sum=True):
        # d^3 n / [dlog10M dq dz]
        if ndens is None:
            ndens = self.static_binary_density
        integ = utils.trapz(ndens, np.log10(self.mtot), axis=0, cumsum=False)
        integ = utils.trapz(integ, self.mrat, axis=1, cumsum=False)
        integ = utils.trapz(integ, self.redz, axis=2, cumsum=False)
        if sum:
            integ = integ.sum()
        return integ

    @utils.deprecated_fail("`dynamic_binary_number_at_fobs` or `sam_cyutils.dynamic_binary_number_at_fobs`")
    def dynamic_binary_number(self, *args, **kwargs):
        pass

    @utils.deprecated_fail("`gwb_new`")
    def new_gwb(self, *args, **kwargs):
        pass


# ===========================================
# ====    Evolution & Utility Methods    ====
# ===========================================


def sample_sam_with_hardening(
        sam, hard,
        fobs_orb=None, sepa=None, sample_threshold=10.0, cut_below_mass=None, limit_merger_time=None,
        **sample_kwargs
):
    """Discretize Semi-Analytic Model into sampled binaries assuming the given binary hardening rate.

    Parameters
    ----------
    sam : `Semi_Analytic_Model`
        Instance of an initialized semi-analytic model.
    hard : `holodeck.evolution._Hardening`
        Binary hardening model for calculating binary hardening rates (dadt or dfdt).
    fobs_orb : ArrayLike
        Observer-frame orbital-frequencies.  Units of [1/sec].
        NOTE: Either `fobs_orb` or `sepa` must be provided, and not both.
    sepa : ArrayLike
        Binary orbital separation.  Units of [cm].
        NOTE: Either `fobs_orb` or `sepa` must be provided, and not both.

    Returns
    -------
    vals : (4, S) ndarray of scalar
        Parameters of sampled binaries.  Four parameters are:
        * mtot : total mass of binary (m1+m2) in [grams]
        * mrat : mass ratio of binary (m2/m1 <= 1)
        * redz : redshift of binary
        * fobs_orb / sepa : observer-frame orbital-frequency [1/s]  or  binary separation [cm]
    weights : (S,) ndarray of scalar
        Weights of each sample point.
    edges : (4,) of list of scalars
        Edges of parameter-space grid for each of above parameters (mtot, mrat, redz, fobs_orb)
        The lengths of each list will be [(M,), (Q,), (Z,), (F,)]
    dnum : (M, Q, Z, F) ndarray of scalar
        Number-density of binaries over grid specified by `edges`.

    """

    if (sample_threshold < 1.0) and (sample_threshold > 0.0):
        msg = (
            f"`sample_threshold={sample_threshold}` values less than unity can lead to surprising behavior!"
        )
        log.warning(msg)

    # returns  dN/[dlog10(M) dq dz dln(f_r)]
    # edges: Mtot [grams], mrat (q), redz (z), {fobs_orb (f) [1/s]   OR   sepa (a) [cm]}
    # `fobs_orb` is observer-frame orbital-frequency
    edges, dnum = sam.dynamic_binary_number(hard, fobs_orb=fobs_orb, sepa=sepa, limit_merger_time=limit_merger_time)

    edges_integrate = [np.copy(ee) for ee in edges]
    edges_sample = [np.log10(edges[0]), edges[1], edges[2], np.log(edges[3])]

    if cut_below_mass is not None:
        m2 = edges[0][:, np.newaxis] * edges[1][np.newaxis, :]
        bads = (m2 < cut_below_mass)
        dnum[bads] = 0.0
        num_bad = np.count_nonzero(bads)
        msg = (
            f"Cutting out systems with secondary below {cut_below_mass/MSOL:.2e} Msol;"
            f" {num_bad:.2e}/{bads.size:.2e} = {num_bad/bads.size:.4f}"
        )
        log.warning(msg)

    # Sample redshift by first converting to comoving volume, sampling, then convert back
    if REDZ_SAMPLE_VOLUME:
        redz = edges[2]
        volume = cosmo.comoving_volume(redz).to('Mpc3').value

        # convert from dN/dz to dN/dVc, dN/dVc = (dN/dz) * (dz/dVc) = (dN/dz) / (dVc/dz)
        dvcdz = cosmo.dVcdz(redz, cgs=False).value
        dnum = dnum / dvcdz[np.newaxis, np.newaxis, :, np.newaxis]

        # change variable from redshift to comoving-volume, both sampling and integration
        edges_sample[2] = volume
        edges_integrate[2] = volume
    else:
        msg = (
            "Sampling redshifts directly, instead of via comoving-volume.  This is less accurate!"
        )
        log.warning(msg)

    # Find the 'mass' (total number of binaries in each bin) by multiplying each bin by its volume
    # NOTE: this needs to be done manually, instead of within kalepy, because of log-spacings
    mass = utils._integrate_grid_differential_number(edges_integrate, dnum, freq=True)

    # ---- sample binaries from distribution
    if (sample_threshold is None) or (sample_threshold == 0.0):
        msg = (
            f"Sampling *all* binaries (~{mass.sum():.2e}).  "
            "Set `sample_threshold` to only sample outliers."
        )
        log.warning(msg)
        vals = kale.sample_grid(edges_sample, dnum, mass=mass, **sample_kwargs)
        weights = np.ones(vals.shape[1], dtype=int)
    else:
        vals, weights = kale.sample_outliers(
            edges_sample, dnum, sample_threshold, mass=mass, **sample_kwargs
        )

    vals[0] = 10.0 ** vals[0]
    vals[3] = np.e ** vals[3]

    # If we sampled in comoving-volume, instead of redshift, convert back to redshift
    if REDZ_SAMPLE_VOLUME:
        vals[2] = np.power(vals[2] / (4.0*np.pi/3.0), 1.0/3.0)
        vals[2] = cosmo.dcom_to_z(vals[2] * MPC)

    # Remove low-mass systems after sampling also
    if cut_below_mass is not None:
        bads = (vals[0] * vals[1] < cut_below_mass)
        vals = vals.T[~bads].T
        weights = weights[~bads]

    return vals, weights, edges, dnum, mass


def evolve_eccen_uniform_single(sam, eccen_init, sepa_init, nsteps):
    """Evolve binary eccentricity from an initial value along a range of separations.

    Parameters
    ----------
    sam : `holodeck.sam.Semi_Analytic_Model` instance
        The input semi-analytic model.  All this does is provide the range of total-masses to
        determine the minimum ISCO radius, which then determines the smallest separations to
        evolve until.
    eccen_init : float,
        Initial eccentricity of binaries at the given initial separation `sepa_init`.
        Must be between [0.0, 1.0).
    sepa_init : float,
        Initial binary separation at which evolution begins.  Units of [cm].
    nsteps : int,
        Number of (log-spaced) steps in separation between the initial separation `sepa_init`,
        and the final separation which is determined as the minimum ISCO radius based on the
        smallest total-mass of binaries in the `sam` instance.

    Returns
    -------
    sepa : (E,) ndarray of float
        The separations at which the eccentricity evolution is defined over.  This is the
        independent variable of the evolution.
        The shape `E` is the value of the `nsteps` parameter.
    eccen : (E,)
        The eccentricity of the binaries at each location in separation given by `sepa`.
        The shape `E` is the value of the `nsteps` parameter.

    """
    assert (0.0 <= eccen_init) and (eccen_init < 1.0)

    #! CHECK FOR COALESCENCE !#

    eccen = np.zeros(nsteps)
    eccen[0] = eccen_init

    sepa_max = sepa_init
    sepa_coal = holo.utils.schwarzschild_radius(sam.mtot) * 3
    # frst_coal = utils.kepler_freq_from_sepa(sam.mtot, sepa_coal)
    sepa_min = sepa_coal.min()
    sepa = np.logspace(*np.log10([sepa_max, sepa_min]), nsteps)

    for step in range(1, nsteps):
        a0 = sepa[step-1]
        a1 = sepa[step]
        da = (a1 - a0)
        e0 = eccen[step-1]

        _, e1 = holo.utils.rk4_step(holo.hardening.Hard_GW.deda, x0=a0, y0=e0, dx=da)
        e1 = np.clip(e1, 0.0, None)
        eccen[step] = e1

    return sepa, eccen


def add_scatter_to_masses(mtot, mrat, dens, scatter, refine=4, log=None):
    """Add the given scatter to masses m1 and m2, for the given distribution of binaries.

    The procedure is as follows (see `dev-notebooks/sam-ndens-scatter.ipynb`):

    * (1) The density is first interpolated to a uniform, regular grid in (m1, m2) space.  A 2nd
          order interpolant is used first.  A 0th-order interpolant is used to fill-in bad values.
          In-between, a 1st-order interpolant is used if `linear_interp_backup` is True.

    * (2) The density distribution is convolved with a smoothing function along each axis (m1, m2)
          to account for scatter.

    * (3) The new density distribution is interpolated back to the original (mtot, mrat) grid.

    Parameters
    ----------
    mtot : (M,) ndarray
        Total masses in grams.
    mrat : (Q,) ndarray
        Mass ratios.
    dens : (M, Q) ndarray
        Density of binaries over the given mtot and mrat domain.
    scatter : float
        Amount of scatter in the M-MBulge relationship, in dex (i.e. over log10 of masses).
    refine : int,
        The increased density of grid-points used in the intermediate (m1, m2) domain, in step (1).
    linear_interp_backup : bool,
        Whether a linear interpolant is used to fill-in bad values after the 2nd order interpolant.
        This generally doesn't seem to fix any values.
    logspace_interp : bool,
        Whether interpolation should be performed in the log-space of masses.
        NOTE: strongly recommended.

    Returns
    -------
    m1m2_dens : (M, Q) ndarray,
        Binary density with scatter introduced.

    """
    if log is None:
        log = holo.log

    assert np.ndim(dens) == 3
    assert np.shape(dens)[:2] == (mtot.size, mrat.size)
    dist = sp.stats.norm(loc=0.0, scale=scatter)
    output = np.zeros_like(dens)

    # Get the primary and secondary masses corresponding to these total-mass and mass-ratios
    m1, m2 = utils.m1m2_from_mtmr(mtot[:, np.newaxis], mrat[np.newaxis, :])
    m1m2_on_mtmr_grid = (m1.flatten(), m2.flatten())

    # Construct a symmetric rectilinear grid in (m1, m2) space
    grid_size = m1.shape[0] * refine
    # make sure the extrema will fully span the required domain
    mextr = utils.minmax([0.9*mtot[0]*mrat[0]/(1.0 + mrat[0]), mtot[-1]*(1.0 + mrat[0])/mrat[0]])
    _mgrid = np.logspace(*np.log10(mextr), grid_size)

    # Interpolate in log-space [recommended]
    mgrid_log10 = np.log10(_mgrid)
    m1m2_on_mtmr_grid = tuple([np.log10(mm) for mm in m1m2_on_mtmr_grid])

    m1m2_grid = np.meshgrid(mgrid_log10, mgrid_log10, indexing='ij')

    # Interpolate from irregular m1m2 space (based on mtmr space), into regular m1m2 grid
    numz = np.shape(dens)[2]
    dlay = None
    weights = utils._get_rolled_weights(mgrid_log10, dist)

    for ii in range(numz):
        dens_redz = dens[:, :, ii]
        if dlay is None:
            points = m1m2_on_mtmr_grid
        else:
            points = dlay
        interp = sp.interpolate.CloughTocher2DInterpolator(points, dens_redz.flatten())
        m1m2_dens = interp(tuple(m1m2_grid))
        if dlay is None:
            dlay = interp.tri

        # Fill in problematic values with zeroth-order interpolant
        bads = np.isnan(m1m2_dens) | (m1m2_dens < 0.0)
        log.debug(f"After interpolation, {utils.frac_str(bads)} bad values exist")
        if np.any(bads):
            # temp = sp.interpolate.griddata(m1m2_on_mtmr_grid, dens.flatten(), tuple(m1m2_grid), method='nearest')
            interp = sp.interpolate.NearestNDInterpolator(points, dens_redz.flatten())
            temp = interp(tuple(m1m2_grid))
            m1m2_dens[bads] = temp[bads]
            bads = np.isnan(m1m2_dens) | (m1m2_dens < 0.0)
            log.debug(f"After 0th order interpolation, {utils.frac_str(bads)} bad values exist")
            if np.any(bads):
                err = f"After 0th order interpolation, {utils.frac_str(bads)} remain!"
                log.exception(err)
                raise ValueError(err)

        # Introduce scatter along both the 0th (primary) and 1th (secondary) axes
        m1m2_dens = utils._scatter_with_weights(m1m2_dens, weights, axis=0)
        m1m2_dens = utils._scatter_with_weights(m1m2_dens, weights, axis=1)

        # Interpolate result back to mtmr grid
        interp = sp.interpolate.RegularGridInterpolator((mgrid_log10, mgrid_log10), m1m2_dens)
        m1m2_dens = interp(m1m2_on_mtmr_grid, method='linear').reshape(m1.shape)
        output[:, :, ii] = m1m2_dens[...]

    return output

