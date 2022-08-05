"""Module for binary evolution from the time of formation/galaxy-merger until BH coalescence.

In `holodeck`, initial binary populations are typically defined near the time of galaxy-galaxy
merger, when two MBHs come together at roughly kiloparsec scales.  Environmental 'hardening'
mechanisms are required to dissipate orbital energy and angular momentum, allowing the binary
separation to shrink ('harden'). Typically *dynamical friction (DF)* is most important at large
scales ($\sim \mathrm{kpc}$).  Near where the pair of MBHs become a gravitationally-bound binary,
the DF approximations break down, and individual *stellar scattering* events (from stars in the
'loss cone' of parameter space) need to be considered.  In the presence of significant gas (i.e.
from accretion), a circumbinary accretion disk (CBD) may form, and gravitational
*circumbinary disk torques* from the gas distribution (typically spiral waves) may become important.
Finally, at the smaller separations, *GW emission* takes over.  The classic work describing the
overall process of binary evolution in its different stages is [BBR1980]_.  [Kelley2017a]_ goes
into significant detail on implementations and properties of each component of binary evolution
also.  Triple MBH interactions, and perturbations from other massive objects (e.g. molecular
clouds) can also be important.

The :mod:`holodeck.evolution` submodule provides tools for modeling the binary evolution from the
time of binary 'formation' (i.e. galaxy merger) until coalescence.  Models for binary evolutionary
can range tremendously in complexity.  In the simplest models, binaries are assumed to coalesce
instantaneously (in that the age of the universe is the same at formation and coalescence), and are
also assumed to evolve purely due to GW emission (in that the time spent in any range of orbital
frequencies can be calculated from the GW hardening timescale).  Note that these two assumptions
are contradictory, but commonly employed in the literature.  The ideal, self-consistent approach,
is to model binary evolution using self-consistently derived environments (e.g. gas and stellar
mass distributions), and applying the same time-evolution prescription to both the redshift
evolution of each binary, and also the GW calculation.  Note that GWs can only be calculated based
on some sort of model for binary evolution.  The model may be extremely simple, in which case it is
sometimes glanced over.

The core component of the evolution module is the :class:`Evolution` class.  This class combines a
population of initial binary parameters (i.e. from the :class:`holodeck.population.Pop_Illustris`
class), along with a specific binary hardening model (:class:`_Hardening` subclass), and performs
the numerical integration of each binary over time - from large separations to coalescence.  The
:class:`Evolution` class also acts to store the binary evolution histories ('trajectories' or
'tracks'), which are then used to calculate GW signals (e.g. the GWB).  To facilitate GW and
similar calculations, :class:`Evolution` also provides interpolation functionality along binary
trajectories.


To-Do
-----
* General
    * evolution modifiers should act at each step, instead of after all steps?  This would be
      a way to implement a changing accretion rate, for example; or to set a max/min hardening rate.
    * re-implement "magic" hardening models that coalesce in zero change-of-redshift or fixed
      amounts of time.
* Dynamical_Friction_NFW
    * Allow stellar-density profiles to also be specified (instead of using a hard-coded
      Dehnen profile)
    * Generalize calculation of stellar characteristic radius.  Make self-consistent with
      stellar-profile, and user-specifiable.
* Sesana_Scattering
    * Allow stellar-density profile (or otherwise the binding-radius) to be user-specified
      and flexible.  Currently hard-coded to Dehnen profile estimate.
* _SHM06
    * Interpolants of hardening parameters return 2D arrays which we then take the diagonal
      of, but there should be a better way of doing this.
* Fixed_Time
    * Handle `rchar` better with respect to interpolation.  Currently not an interpolation
      variable, which restricts it's usage.

References
----------
* [Quinlan1996]_ Quinlan 1996.
* [Sesana2006]_ Sesana, Haardt & Madau et al. 2006.
* [BBR1980]_ Begelman, Blandford & Rees 1980.
* [Sesana2010]_ Sesana 2010.
* [Kelley2017a]_ Kelley, Blecha & Hernquist 2017.
* [Chen2017]_ Chen, Sesana, & Del Pozzo 2017.

"""
from __future__ import annotations

import abc
import inspect
import json
import os
# from typing import Union, TypeVar  # , Callable, Iterator
import warnings

import numpy as np
import scipy as sp
import scipy.interpolate   # noqa

import kalepy as kale

import holodeck as holo
from holodeck import utils, cosmo, log, _PATH_DATA
from holodeck.constants import GYR, NWTG, PC, MSOL, YR, SPLC

_DEF_TIME_DELAY = (5.0*GYR, 0.2)   #: default delay-time parameters, (mean, stdev)
_SCATTERING_DATA_FILENAME = "SHM06_scattering_experiments.json"

# Pop = TypeVar('Pop', bound=holo.population._Population_Discrete)  # Must be exactly str or bytes
# A = TypeVar('A', str, bytes)  # Must be exactly str or bytes
# S = TypeVar('S', bound=str)  # Can be any subtype of str
# Hard = TypeVar('Hard', bound=holo.evolution._Hardening)  # Can be any subtype of str
# Hard = TypeVar('Hard', holo.evolution._Hardening, list[holo.evolution._Hardening])  # Can be any subtype of str
# AliasType = Union[list[dict[tuple[int, str], set[int]]], tuple[str, list[str]]]
# Hard_list = Union[list[Hard], Hard]


# =================================================================================================
# ====    Evolution Class    ====
# =================================================================================================


class Evolution:
    """Base class to evolve discrete binary populations forward in time.

    NOTE: This class is primary built to be used with :class:`holodeck.population.Pop_Illustris`.

    The `Evolution` class is instantiated with a :class:`holodeck.population._Population_Discrete`
    instance, and a particular binary hardening model (subclass of :class:`_Hardening`).  It then
    numerically integrates each binary from their initial separation to coalescence, determining
    how much time that process takes, and thus the rate of redshift/time evolution.

    **Initialization**: all attributes are set to empty arrays of the appropriate size.
    NOTE: the 0th step is *not* initialized at this time, it happens in :meth:`Evolution.evolve()`.

    **Evolution**: binary evolution is performed by running the :meth:`Evolution.evolve()` function.
    This function first calls :meth:`Evolution._init_step_zero()`, which sets the 0th step values,
    and then iterates over each subsequent step, calling :meth:`Evolution._take_next_step()`.  Once
    all steps are taken (integration is completed), then :meth:`Evolution._finalize()` is called,
    at which points any stored modifiers (:class:`utils._Modifier` subclasses, in the
    :attr:`Evolution._mods` attribute) are applied.

    """

    _EVO_PARS = ['mass', 'sepa', 'eccen', 'scafa', 'tlbk', 'dadt', 'dedt']
    _LIN_INTERP_PARS = ['eccen', 'scafa', 'tlbk', 'dadt', 'dedt']
    _SELF_CONSISTENT = None
    _STORE_FROM_POP = ['_sample_volume']

    # scafa: np.ndarray      #: scale-factor of the universe, set to 1.0 after z=0
    # tlbk: np.ndarray       #: lookback time, negative after redshift zero [sec]
    # sepa: np.ndarray       #: semi-major axis (separation) [cm]
    # mass: np.ndarray       #: mass of BHs, 0-primary, 1-secondary, [g]
    # mdot: np.ndarray       #: accretion rate onto each component of binary [g/s]
    # dadt: np.ndarray       #: hardening rate in separation [cm/s]
    # eccen: np.ndarray      #: eccentricity, `None` if not being evolved []
    # dedt: np.ndarray       #: eccen evolution, `None` if not evolved [1/s]

    def __init__(self, pop, hard, nsteps: int = 100, mods=None, debug: bool = False):
        """Initialize a new Evolution instance.

        Parameters
        ----------
        pop : `population._Population_Discrete` instance,
            Binary population with initial parameters of the binary from which to start evolution.
        hard : `_Hardening` instance, or list of
            Model for binary hardening used to evolve population's separation over time.
        nsteps : int,
            Number of steps between initial separations and coalescence for all binaries.
        mods : None, or list of `utils._Modifier` subclasses,
            NOTE: not fully implemented!
        debug : bool,
            Include verbose/debugging output information.

        """
        # --- Store basic parameters to instance
        self._pop = pop                       #: initial binary population instance
        self._debug = debug                   #: debug flag for performing extra diagnostics and output
        self._nsteps = nsteps                 #: number of integration steps for each binary
        self._mods = mods                     #: modifiers to be applied after evolution is completed

        # Store hardening instances as a list
        if not np.iterable(hard):
            hard = [hard, ]
        self._hard = hard

        # Make sure types look right
        if not isinstance(pop, holo.population._Population_Discrete):
            err = f"`pop` is {pop}, must be subclass of `holo.population._Population_Discrete`!"
            log.exception(err)
            raise TypeError(err)

        for hh in self._hard:
            if not isinstance(hh, _Hardening):
                err = f"hardening instance is {hh}, must be subclass of `holo.evolution._Hardening`!"
                log.exception(err)
                raise TypeError(err)

        # Store additional parameters
        for par in self._STORE_FROM_POP:
            setattr(self, par, getattr(pop, par))

        size = pop.size
        shape = (size, nsteps)
        self._shape = shape

        if pop.eccen is not None:
            eccen = np.zeros(shape)
            dedt = np.zeros(shape)
        else:
            eccen = None
            dedt = None

        # ---- Initialize empty arrays for tracking binary evolution
        self.scafa = np.zeros(shape)           #: scale-factor of the universe, set to 1.0 after z=0
        self.tlbk = np.zeros(shape)            #: lookback time, negative after redshift zero [sec]
        self.sepa = np.zeros(shape)            #: semi-major axis (separation) [cm]
        self.mass = np.zeros(shape + (2,))     #: mass of BHs, 0-primary, 1-secondary, [g]
        self.mdot = np.zeros(shape + (2,))     #: accretion rate onto each component of binary [g/s]
        self.dadt = np.zeros(shape)            #: hardening rate in separation [cm/s]
        self.eccen = eccen                     #: eccentricity, `None` if not being evolved []
        self.dedt = dedt                       #: eccen evolution, `None` if not evolved [1/s]

        self._dadt_0 = None   # this is a placeholder for initializing debug output

        # Derived and internal parameters
        self._freq_orb_rest = None
        self._evolved = False
        self._coal = None

        return

    # ==== API and Core Functions

    def evolve(self, progress=True):
        """Evolve binary population from initial separation until coalescence in discrete steps.

        Each binary has a fixed number of 'steps' from initial separation until coalescence.  The
        role of the `evolve()` method is to determine the amount of time each step takes, based on
        the 'hardening rate' (in separation and possible eccentricity i.e. $da/dt$ and $de/dt$).
        The hardening rate is calculated from the stored :class:`_Hardening` instances in the
        iterable :attr:`Evolution._hard` attribute.

        When :meth:`Evolution.evolve()` is called, the 0th step is initialized separately, using
        the :meth:`Evolution._init_step_zero()` method, and then each step is integrated by calling
        the :meth:`Evolution._take_next_step()` method.  Once all steps are completed, the
        :meth:`Evolution._finalize()` method is called, where any stored modifiers are applied.

        Parameters
        ----------
        progress : bool,
            Show progress-bar using `tqdm` package.

        """
        # ---- Initialize Integration Step Zero
        self._init_step_zero()

        # ---- Iterate through all integration steps
        size, nsteps = self.shape
        steps_list = range(1, nsteps)
        steps_list = utils.tqdm(steps_list, desc="evolving binaries") if progress else steps_list
        for step in steps_list:
            self._take_next_step(step)

        # ---- Finalize
        self._finalize()
        return

    def modify(self, mods=None):
        """Apply and modifiers after evolution has been completed.
        """
        self._check_evolved()

        if mods is None:
            mods = self._mods

        # Sanitize
        if mods is None:
            mods = []
        elif not isinstance(mods, list):
            mods = [mods]

        # Run Modifiers
        for mod in mods:
            mod(self)

        # Update derived quantites (following modifications)
        self._update_derived()
        return

    def at(self, xpar, targets, pars=None, coal=False, lin_interp=None):
        """Interpolate evolution to the given observed, orbital frequency.

        Parameters
        ----------
        xpar : str, in ['fobs', 'sepa']
            String specifying the variable to interpolate to.
        targets : float or array_like,
            Locations to interpolate to.
            `sepa` : units of cm
            `fobs` : units of 1/s [Hz]
        pars : None or (list of str)
            Parameters that should be interpolated.
            If `None`, defaults to `self._EVO_PARS` attribute.
        coal : bool,
            Only return evolution values for binaries coalescing before redshift zero.
        lin_interp : None or bool,

        Returns
        -------
        vals : dict,
            Dictionary of arrays for each interpolated parameter.
            The returned shape is (N, T), where `T` is the number of target locations to interpolate
            to, and `N` is the total number of binaries (``coal=False``) or the number of binaries
            coalescing before redshift zero (``coal=True``).

        Notes
        -----
        Out of bounds values are set to `np.nan`.
        Interpolation is 1st-order in log-log space.

        """
        self._check_evolved()
        _allowed = ['sepa', 'fobs']
        if xpar not in _allowed:
            raise ValueError("`xpar` must be one of '{}'!".format(_allowed))

        if pars is None:
            pars = self._EVO_PARS
        if np.isscalar(pars):
            pars = [pars]
        squeeze = False
        if np.isscalar(targets):
            targets = np.atleast_1d(targets)
            squeeze = True

        size, nsteps = self.shape
        # Observed-Frequency, units of 1/yr
        if xpar == 'fobs':
            # frequency is already increasing
            _xvals = np.log10(self.freq_orb_obs)
            scafa = self.scafa[:, :]
            tt = np.log10(targets)
            rev = False
        # Binary-Separation, units of pc
        elif xpar == 'sepa':
            # separation is decreasing, reverse to increasing
            _xvals = np.log10(self.sepa)[:, ::-1]
            scafa = self.scafa[:, ::-1]
            tt = np.log10(targets)
            rev = True
        else:
            raise ValueError("Bad `xpar` {}!".format(xpar))

        # Find the evolution-steps immediately before and after the target frequency
        textr = utils.minmax(tt)
        xextr = utils.minmax(_xvals)
        if (textr[1] < xextr[0]) | (textr[0] > xextr[1]):
            err = "`targets` extrema ({}) ourside `xvals` extema ({})!  Bad units?".format(
                (10.0**textr), (10.0**xextr))
            raise ValueError(err)

        # Convert to (N, T, M)
        #     `tt` is (T,)  `xvals` is (N, M) for N-binaries and M-steps
        select = (tt[np.newaxis, :, np.newaxis] <= _xvals[:, np.newaxis, :])
        # Select only binaries that coalesce before redshift zero (a=1.0)
        if coal:
            select = select & (scafa[:, np.newaxis, :] > 0.0) & (scafa[:, np.newaxis, :] < 1.0)

        # (N, T), find the index of the xvalue following each target point (T,), for each binary (N,)
        aft = np.argmax(select, axis=-1)
        # zero values in `aft` mean no xvals after the targets were found
        valid = (aft > 0)
        bef = np.copy(aft)
        bef[valid] -= 1

        # (2, N, T)
        cut = np.array([aft, bef])
        # Valid binaries must be valid at both `bef` and `aft` indices
        for cc in cut:
            valid = valid & np.isfinite(np.take_along_axis(scafa, cc, axis=-1))

        inval = ~valid
        # Get the x-values before and after the target locations  (2, N, T)
        xvals = [np.take_along_axis(_xvals, cc, axis=-1) for cc in cut]
        # Find how far to interpolate between values (in log-space)
        #     (N, T)
        temp = np.subtract(*xvals)
        numer = tt[np.newaxis, :] - xvals[1]
        frac = np.zeros_like(numer)
        idx = (temp != 0.0)
        frac[idx] = numer[idx] / temp[idx]

        vals = dict()
        # Interpolate each target parameter
        for pp in pars:
            lin_interp_flag = (pp in self._LIN_INTERP_PARS) if lin_interp is None else lin_interp
            # Either (N, M) or (N, M, 2)
            _data = getattr(self, pp)
            if _data is None:
                vals[pp] = None
                continue

            reshape = False
            use_cut = cut
            use_frac = frac
            # Sometimes there is a third dimension for the 2 binaries (e.g. `mass`)
            #    which will have shape, (N, T, 2) --- calling this "double-data"
            if np.ndim(_data) != 2:
                if (np.ndim(_data) == 3) and (np.shape(_data)[-1] == 2):
                    # Keep the interpolation axis last (N, T, 2) ==> (N, 2, T)
                    _data = np.moveaxis(_data, -1, -2)
                    # Expand other arrays appropriately
                    use_cut = cut[:, :, np.newaxis]
                    use_frac = frac[:, np.newaxis, :]
                    reshape = True
                else:
                    raise ValueError("Unexpected shape of data: {}!".format(np.shape(_data)))

            if not lin_interp_flag:
                _data = np.log10(_data)

            if rev:
                _data = _data[..., ::-1]
            # (2, N, T) for scalar data or (2, N, 2, T) for "double-data"
            data = [np.take_along_axis(_data, cc, axis=-1) for cc in use_cut]
            # Interpolate by `frac` for each binary   (N, T) or (N, 2, T) for "double-data"
            new = data[1] + (np.subtract(*data) * use_frac)
            # In the "double-data" case, move the doublet back to the last dimension
            #    (N, T) or (N, T, 2)
            if reshape:
                new = np.moveaxis(new, 1, -1)
            # Set invalid binaries to nan
            new[inval, ...] = np.nan

            # if np.any(~np.isfinite(new[valid, ...])):
            #     raise ValueError("Non-finite values after interpolation of '{}'".format(pp))

            # fill return dictionary
            if not lin_interp_flag:
                new = 10.0 ** new
            if squeeze:
                new = new.squeeze()
            vals[pp] = new

            # if np.any(~np.isfinite(new[valid, ...])):
            #     raise ValueError("Non-finite values after exponentiation of '{}'".format(pp))

        return vals

    def sample_full_population(self, freqs, DOWN=None):
        """Construct a full universe of binaries based on resampling this population.

        !**WARNING**!
        NOTE: This function should be cleaned up / improved for public use.

        Parameters
        ----------
        freqs : array_like,
            Target observer-frame frequencies at which to sample population.
        DOWN : None or float,
            Factor by which to downsample the resulting population.
            For example, `10.0` will produce 10x fewer output binaries.

        Returns
        -------
        samples


        """
        PARAMS = ['mass', 'sepa', 'dadt', 'scafa']
        fobs = kale.utils.spacing(freqs, scale='log', dex=10, log_stretch=0.1)
        log.info(f"Converted input freqs ({kale.utils.stats_str(freqs)}) ==> {kale.utils.stats_str(fobs)}")
        data_fobs = self.at('fobs', fobs, pars=PARAMS)

        # Only examine binaries reaching the given locations before redshift zero (other redz=infinite)
        redz = data_fobs['scafa']
        redz = cosmo.a_to_z(redz)
        valid = np.isfinite(redz) & (redz > 0.0)
        # nvalid = np.count_nonzero(valid)
        # print(f"Valid indices: {nvalid}/{valid.size}={nvalid/valid.size:.4e}")

        frst = utils.frst_from_fobs(fobs[np.newaxis, :], redz)
        dcom = cosmo.z_to_dcom(redz)

        # `mass` has shape (Binaries, Frequencies, 2)
        #    convert to (2, B, F), then separate into m1, m2 each with shape (B, F)
        m1, m2 = np.moveaxis(data_fobs['mass'], -1, 0)
        # mchirp = utils.chirp_mass(m1, m2)
        dfdt, _ = utils.dfdt_from_dadt(data_fobs['dadt'], data_fobs['sepa'], freq_orb=frst)
        _tres = frst / dfdt

        vfac = 4.0*np.pi*SPLC * (redz+1.0) * dcom**2 / self._sample_volume   # * thub
        tfac = _tres  # / thub

        # ---- Get the "Lambda"/Poisson weighting factor ----
        # Calculate weightings
        #    Sesana+08, Eq.10
        lambda_factor = vfac * tfac
        # print(f"`lambda_factor` = {lambda_factor.shape}, {utils.stats(lambda_factor)}")

        mt, mr = utils.mtmr_from_m1m2(m1[valid], m2[valid])
        fo = (fobs[np.newaxis, :] * np.ones_like(redz))[valid]
        redz = redz[valid]
        weights = lambda_factor[valid]

        if DOWN is not None:
            prev_sum = weights.sum()
            weights /= DOWN
            next_sum = weights.sum()
            log.warning(f"DOWNSAMPLING ARTIFICIALLY!!  DOWN={DOWN:g} :: {prev_sum:.4e}==>{next_sum:.4e}")

        # vals = [mt, mr, redz, fo]
        # TODO/FIX: Consider sampling in comoving-volume instead of redz (like in sam.py)
        #           can also return dcom instead of redz for easier strain calculation
        vals = [np.log10(mt), np.log10(mr), np.log10(redz), np.log10(fo)]
        nsamp = np.random.poisson(weights.sum())
        reflect = [None, [None, 0.0], None, np.log10([0.95*fobs[0], fobs[-1]*1.05])]
        # print(f"{nsamp=:.4e}, {np.shape(vals)=}")
        samples = kale.resample(vals, size=nsamp, reflect=reflect, weights=weights, bw_rescale=0.5)
        # print(f"{samples.shape=}")

        samples = np.asarray([10.0 ** ss for ss in samples])
        # hs, fo = holo.sam._strains_from_samples(samples)
        return samples

    # ==== Internal Methods

    def _init_step_zero(self):
        """Set the initial conditions of the binaries at the 0th step.

        Transfers attributes from the stored :class:`holodeck.population._Population_Discrete`
        instance to the 0th index of the evolution arrays.  The attributes are [`sepa`, `scafa`,
        `mass`, and optionally `eccen`].  The hardening model is also used to calculate the 0th
        hardening rates `dadt` and `dedt`.  The initial lookback time, `tlbk` is also set.

        """
        pop = self._pop
        size, nsteps = self.shape

        # ---- Initialize basic parameters

        # Initialize ALL separations ranging from initial to mutual-ISCO, for each binary
        rad_isco = utils.rad_isco(*pop.mass.T)
        # (2, N)
        sepa = np.log10([pop.sepa, rad_isco])
        # Get log-space range of separations for each of N ==> (N, S), for S steps
        sepa = np.apply_along_axis(lambda xx: np.logspace(*xx, nsteps), 0, sepa).T
        self.sepa[:, :] = sepa
        if (pop.eccen is not None):
            self.eccen[:, 0] = pop.eccen

        self.scafa[:, 0] = pop.scafa
        redz = cosmo.a_to_z(pop.scafa)
        tlbk = cosmo.z_to_tlbk(redz)
        self.tlbk[:, 0] = tlbk
        # `pop.mass` has shape (N, 2), broadcast to (N, S, 2) for `S` steps
        self.mass[:, :, :] = pop.mass[:, np.newaxis, :]

        # ---- Initialize hardening rate at first step
        dadt_init, dedt_init = self._hardening_rate(step=0)

        self.dadt[:, 0] = dadt_init
        if (pop.eccen is not None):
            self.dedt[:, 0] = dedt_init

        return

    def _take_next_step(self, step):
        """Integrate the binary population forward (to smaller separations) by one step.

        For an integration step `s`, we are moving from index `s-1` to index `s`.  These correspond
        to the 'left' and 'right' edges of the step.  The evolutionary trajectory values have
        already been calculated on the left edges (during either the previous time step, or the
        initial time step).  Each subsequent integration step then proceeds as follows:

        (1) The hardening rate is calculated at the right edge of the step.
        (2) The time it takes to move from the left to right edge is calculated using a trapezoid
            rule in log-log space.
        (3) The right edge evolution values are stored and updated.

        Parameters
        ----------
        step : int
            The destination integration step number, i.e. `step=1` means integrate from 0 to 1.

        """
        # ---- Initialize
        size, nsteps = self.shape
        left = step - 1     # the previous time-step (already completed)
        right = step        # the next     time-step

        # ---- Hardening rates at the right-edge of the step
        # calculate
        dadt_r, dedt_r = self._hardening_rate(right)
        # store
        self.dadt[:, right] = dadt_r
        if self.eccen is not None:
            self.dedt[:, right] = dedt_r

        # ---- Calculate time between edges

        # get the $dt/da$ rate on both edges of the step
        dtda = 1.0 / - self.dadt[:, (left, right)]   # NOTE: `dadt` is negative, convert to positive
        # get the deparation $a$ on both edges
        sepa = self.sepa[:, (right, left)]   # sepa is decreasing, so switch left-right order
        # use trapezoid rule to find total time for this step
        dt = utils.trapz_loglog(dtda, sepa, axis=-1).squeeze()   # this should come out positive
        if np.any(dt < 0.0):
            utils.error(f"Negative time-steps found at step={step}!")

        # ---- Update right-edge values
        # NOTE/ENH: this would be a good place to make a function `_update_right_edge()` (or something like that),
        # that stores the updated right edge values, and also performs any additional updates, such as mass evolution

        # Update lookback time based on duration of this step
        tlbk = self.tlbk[:, left] - dt
        self.tlbk[:, right] = tlbk
        # update scale-factor for systems at z > 0.0 (i.e. a < 1.0 and tlbk > 0.0)
        val = (tlbk > 0.0)
        self.scafa[val, right] = cosmo.z_to_a(cosmo.tlbk_to_z(tlbk[val]))
        # set systems after z = 0 to scale-factor of unity
        self.scafa[~val, right] = 1.0

        # update eccentricity if it's being evolved
        if self.eccen is not None:
            dedt = self.dedt[:, (left, right)]
            time = self.tlbk[:, (right, left)]   # tlbk is decreasing, so switch left-right order
            # decc = utils.trapz_loglog(dedt, time, axis=-1).squeeze()
            decc = utils.trapz(dedt, time, axis=-1).squeeze()
            self.eccen[:, right] = self.eccen[:, left] + decc
            if self._debug:
                bads = ~np.isfinite(decc)
                if np.any(bads):
                    utils.print_stats(print_func=log.error, dedt=dedt, time=time, decc=decc)
                    err = f"Non-finite changes in eccentricity found in step {step}!"
                    log.exception(err)
                    raise ValueError(err)

        return

    def _hardening_rate(self, step):
        """Calculate the net hardening rate for the given integration step.

        The hardening rates (:class:`_Hardening` subclasses) stored in the :attr:`Evolution._hard`
        attribute are called in sequence, their :meth:`_Hardening.dadt_dedt` methods are called,
        and the $da/dt$ and $de/dt$ hardening rates are added together.
        NOTE: the da/dt and de/dt values are added together to get the net rate, this is an
        approximation.

        Parameters
        ----------
        step : int
            Current step number (the destination of the current step, i.e. step=1 is for integrating
            from 0 to 1.)

        Returns
        -------
        dadt : np.ndarray
            The hardening rate in separation, $da/dt$, in units of [cm/s].
            The shape is (N,) where N is the number of binaries.
        dedt : np.ndarray or None
            If eccentricity is not being evolved, this is `None`.  If eccentricity is being evolved,
            this is the hardening rate in eccentricity, $de/dt$, in units of [1/s].
            In this case, the shape is (N,) where N is the number of binaries.

        """
        dadt = np.zeros(self.shape[0])
        dedt = None if self.eccen is None else np.zeros_like(dadt)

        for ii, hard in enumerate(self._hard):
            _ar, _er = hard.dadt_dedt(self, step)
            if self._debug:
                log.debug(f"hard={hard} : dadt = {utils.stats(_ar)}")
                # Store individual hardening rates
                getattr(self, f"_dadt_{ii}")[:, step] = _ar[...]
                # Raise error on invalid entries
                if not np.all(np.isfinite(_ar)) or np.any(_ar > 0.0):
                    utils.error(f"invalid `dadt` for hard={hard}!")

            dadt[:] += _ar
            if (self.eccen is not None) and (_er is not None):
                dedt[:] += _er

        return dadt, dedt

    def _check(self):
        """Perform basic diagnostics on parameter validity after evolution.
        """
        _check_var_names = ['sepa', 'scafa', 'mass', 'tlbk', 'dadt']
        _check_var_names_eccen = ['eccen', 'dedt']

        def check_vars(names):
            for cv in names:
                vals = getattr(self, cv)
                if np.any(~np.isfinite(vals)):
                    err = "Found non-finite '{}' !".format(cv)
                    raise ValueError(err)

        check_vars(_check_var_names)

        if self.eccen is None:
            return

        check_vars(_check_var_names_eccen)

        return

    def _finalize(self):
        """Perform any actions after completing all of the integration steps.
        """
        # Set a flag to record that evolution has been completed
        self._evolved = True
        # Apply any modifiers
        self.modify()
        # Run diagnostics
        self._check()
        return

    def _update_derived(self):
        """Update any derived quantities after modifiers are applied.
        """
        pass

    # ==== Properties and generic functionality

    @property
    def shape(self):
        """The number of binaries and number of steps (N, S).
        """
        return self._shape

    @property
    def coal(self):
        """Indices of binaries that coalesce before redshift zero.
        """
        if self._coal is None:
            self._coal = (self.redz[:, -1] > 0.0)
        return self._coal

    @property
    def freq_orb_rest(self):
        """Rest-frame orbital frequency. [1/s]
        """
        if self._freq_orb_rest is None:
            self._check_evolved()
            mtot = self.mass.sum(axis=-1)
            self._freq_orb_rest = utils.kepler_freq_from_sepa(mtot, self.sepa)
        return self._freq_orb_rest

    @property
    def freq_orb_obs(self):
        """Observer-frame orbital frequency. [1/s]
        """
        redz = cosmo.a_to_z(self.scafa)
        fobs = self.freq_orb_rest / (1.0 + redz)
        return fobs

    def _check_evolved(self):
        """Raise an error if this instance has not yet been evolved.
        """
        if self._evolved is not True:
            raise RuntimeError("This instance has not been evolved yet!")
        return


# =================================================================================================
# ====    Hardening Classes    ====
# =================================================================================================


class _Hardening(abc.ABC):
    """Base class for binary-hardening models, providing the `dadt_dedt(evo, step, ...)` method.



    """

    @abc.abstractmethod
    def dadt_dedt(self, evo, step, *args, **kwargs):
        pass


class Hard_GW(_Hardening):
    """Gravitational-wave driven binary hardening.
    """

    @staticmethod
    def dadt_dedt(evo, step):
        m1, m2 = evo.mass[:, step, :].T    # (Binaries, Steps, 2) ==> (2, Binaries)
        sepa = evo.sepa[:, step]
        eccen = evo.eccen[:, step] if (evo.eccen is not None) else None
        dadt = utils.gw_hardening_rate_dadt(m1, m2, sepa, eccen=eccen)

        if eccen is None:
            dedt = None
        else:
            dedt = utils.gw_dedt(m1, m2, evo.sepa, evo.eccen)

        return dadt, dedt

    @staticmethod
    def dadt(mt, mr, sepa):
        m1, m2 = utils.m1m2_from_mtmr(mt, mr)
        dadt = utils.gw_hardening_rate_dadt(m1, m2, sepa, eccen=None)
        return dadt


class Sesana_Scattering(_Hardening):
    """Binary-Hardening Rates calculated based on the Sesana stellar-scattering model.

    This module uses the stellar-scattering rate constants from the fits in [Sesana2006]_ using the
    `_SHM06` class.  Scattering is assumed to only be effective once the binary is bound.  An
    exponential cutoff is imposed at larger radii.

    Parameters
    ----------
    gamma_dehnen : scalar  or  (N,) array-like of scalar
        Dehnen stellar-density profile inner power-law slope.
        Fiducial Dehnen inner density profile slope gamma=1.0 is used in [Chen2017]_.
    gbh : _Galaxy_Blackhole_Relation class/instance  or  `None`
        Galaxy-Blackhole Relation used for calculating stellar parameters.
        If `None` the default is loaded.

    """

    def __init__(self, gamma_dehnen=1.0, gbh=None):
        gbh = _get_galaxy_blackhole_relation(gbh)
        self._gbh = gbh
        self._gamma_dehnen = gamma_dehnen

        self._shm06 = _SHM06()
        return

    def dadt_dedt(self, evo, step):
        """Stellar scattering hardening rate.

        Parameters
        ----------
        evo : `Evolution` instance
        step : int
            Integration step at which to calculate hardening rates.

        Returns
        -------
        dadt : (N,) array-like of scalar
            Binary hardening rates in units of [cm/s], defined to be negative.
        dedt : (N,) array-like of scalar
            Binary rate-of-change of eccentricity in units of [1/sec].

        """
        mass = evo.mass[:, step, :]
        sepa = evo.sepa[:, step]
        eccen = evo.eccen[:, step] if evo.eccen is not None else None
        dadt, dedt = self._dadt_dedt(mass, sepa, eccen)
        return dadt, dedt

    def _dadt_dedt(self, mass, sepa, eccen):
        """Stellar scattering hardening rate calculated from physical quantities.

        Parameters
        ----------
        mass : (N,2) array-like of scalar
            Masses of each MBH component (0-primary, 1-secondary) in units of [gram].
        sepa : (N,) array-like of scalar
            Binary separation in units of [cm].
        eccen : (N,) array-like of scalar or `None`
            Binary eccentricity.  `None` if eccentricity is not being evolved.

        Returns
        -------
        dadt : (N,) array-like of scalar
            Binary hardening rates in units of [cm/s], defined to be negative.
        dedt : (N,) array-like of scalar  or  `None`
            Binary rate-of-change of eccentricity in units of [1/sec].
            If eccentricity is not being evolved (i.e. `eccen==None`) then `None` is returned.

        """
        mtot, mrat = utils.mtmr_from_m1m2(mass)
        vdisp = self._gbh.vdisp_from_mbh(mtot)
        mbulge = self._gbh.mbulge_from_mbh(mtot)
        dens = _density_at_influence_radius_dehnen(mtot, mbulge, self._gamma_dehnen)

        rhard = _Quinlan1996.radius_hardening(mass[:, 1], vdisp)
        hh = self._shm06.H(mrat, sepa/rhard)
        dadt = _Quinlan1996.dadt(sepa, dens, vdisp, hh)

        rbnd = _radius_influence_dehnen(mtot, mbulge)
        atten = np.exp(-sepa / rbnd)
        dadt = dadt * atten

        if eccen is not None:
            kk = self._shm06.K(mrat, sepa/rhard, eccen)
            dedt = _Quinlan1996.dedt(sepa, dens, vdisp, hh, kk)
        else:
            dedt = None

        return dadt, dedt


class Dynamical_Friction_NFW(_Hardening):
    """Dynamical Friction (DF) hardening module assuming an NFW dark-matter density profile.

    This class calculates densities and orbital velocities based on a NFW profile with parameters based on those of
    each MBH binary.  The `holodeck.observations.NFW` class is used for profile calculations, and the halo parameters
    are calculated from Stellar-mass--Halo-mass relations (see 'arguments' below).  The 'effective-mass' of the
    inspiralling secondary is modeled as a power-law decreasing from the sum of secondary MBH and its stellar-bulge
    (calculated using the `gbh` - Galaxy-Blackhole relation), down to just the bare secondary MBH after 10 dynamical
    times.  This is to model tidal-stripping of the secondary host galaxy.

    Attenuation of the DF hardening rate is typically also included, to account for the inefficiency of DF once the
    binary enters the hardened regime.  This is calculated using the prescription from [BBR1980]_.  The different
    characteristic radii, needed for the attenuation calculation, currently use a fixed Dehnen stellar-density profile
    as in [Chen2017]_, and a fixed scaling relationship to find the characteristic stellar-radius.

    This module does not evolve eccentricity.

    Parameters
    ----------
    gbh : class, instance or None
        Galaxy-blackhole relation (_Galaxy_Blackhole_Relation subclass)
        If `None` the default is loaded.
    smhm : class, instance or None
        Stellar-mass--halo-mass relation (_StellarMass_HaloMass subclass)
        If `None` the default is loaded.
    coulomb : scalar,
        coulomb-logarithm ("log(Lambda)"), typically in the range of 10-20.
        This parameter is formally the log of the ratio of maximum to minimum impact parameters.
    attenuate : bool,
        Whether the DF hardening rate should be 'attenuated' due to stellar-scattering effects at
        small radii.  If `True`, DF becomes significantly less effective for radii < R_hard and R_LC
    rbound_from_density : bool,
        Determines how the binding radius (of MBH pair) is calculated, which is used for attenuation.
        NOTE: this is only used if `attenuate==True`
        If True:  calculate R_bound using an assumed stellar density profile.
        If False: calculate R_bound using a velocity dispersion (constant in radius, from `gbh` instance).

    Notes
    -----
    *   The hardening rate (da/dt) is not allowed to be larger than the orbital/virial velocity of the halo
        (as a function of radius).

    """

    def __init__(self, gbh=None, smhm=None, coulomb=10.0, attenuate=True, rbound_from_density=True):
        gbh = _get_galaxy_blackhole_relation(gbh)
        smhm = _get_stellar_mass_halo_mass_relation(smhm)
        self._gbh = gbh
        self._smhm = smhm
        self._coulomb = coulomb
        self._attenuate = attenuate
        self._rbound_from_density = rbound_from_density

        self._NFW = holo.observations.NFW
        self._time_dynamical = None
        return

    def _dvdt(self, mass_sec_eff, dens, velo):
        """Chandrasekhar dynamical friction formalism providing a deceleration (dv/dt).

        Parameters
        ----------
        mass_sec_eff : (N,) array-like of scalar
            Effective mass (i.e. the mass that should be used in this equation) of the inspiraling
            secondary component in units of [gram].
        dens : (N,) array-like of scalar
            Effective density at the location of the inspiraling secondary in units of [g/cm^3].
        velo : (N,) array-like of scalar
            Effective velocity of the inspiraling secondary in units of [cm/s].

        Returns
        -------
        dvdt (N,) np.ndarray of scalar
            Deceleration rate of the secondary object in units of [cm/s^2].

        """
        dvdt = - 2*np.pi * mass_sec_eff * dens * self._coulomb * np.square(NWTG / velo)
        return dvdt

    def dadt_dedt(self, evo, step):
        """Calculate DF hardening rate given `Evolution` instance, and an integration `step`.

        Parameters
        ----------
        evo : `Evolution` instance
        step : int,
            Integration step at which to calculate hardening rates.

        Returns
        -------
        dadt : (N,) np.ndarray of scalar,
            Binary hardening rates in units of [cm/s].
        dedt : (N,) np.ndarray of scalar, NOTE: always zero
            Rate-of-change of eccentricity, which is not included in this calculation (i.e. zero)

        """
        mass = evo.mass[:, step, :]
        sepa = evo.sepa[:, step]
        dt = evo.tlbk[:, 0] - evo.tlbk[:, step]   # positive time-duration since 'formation'
        # NOTE `scafa` is nan for systems "after" redshift zero (i.e. do not merge before redz=0)
        redz = np.zeros_like(sepa)
        val = (evo.scafa[:, step] > 0.0)
        redz[val] = cosmo.a_to_z(evo.scafa[val, step])

        dadt, dedt = self._dadt_dedt(mass, sepa, redz, dt)

        return dadt, dedt

    def _dadt_dedt(self, mass, sepa, redz, dt, attenuate=None):
        """Calculate DF hardening rate given physical quantities.

        Parameters
        ----------
        mass : (N, 2) array-like of scalar,
            Masses of both MBHs (0-primary, 1-secondary) in units of [grams].
        sepa : (N,) array-like of scalar,
            Binary separation in [cm].
        redz : (N,) array-like of scalar,
            Binary redshifts.

        Returns
        -------
        dadt : (N,) np.ndarray of scalar,
            Binary hardening rates in units of [cm/s].
        dedt : (N,) np.ndarray of scalar, NOTE: always zero
            Rate-of-change of eccentricity, which is not included in this calculation.

        """
        if attenuate is None:
            attenuate = self._attenuate

        # ---- Get Host DM-Halo mass
        # use "bulge-mass" as a proxy for total stellar mass
        mstar = self._gbh.mbulge_from_mbh(mass[:, 0])   # use primary-bh's mass (index 0)
        mhalo = self._smhm.halo_mass(mstar, redz, clip=True)

        # ---- Get effective mass of inspiraling secondary
        m2 = mass[:, 1]
        mstar_sec = self._gbh.mbulge_from_mbh(m2)
        if self._time_dynamical is None:
            self._time_dynamical = self._NFW.time_dynamical(sepa, mhalo, redz) * 10

        # model tidal-stripping of secondary's bulge (see: [Kelley2017a] Eq.6)
        pow = np.clip(1.0 - dt / self._time_dynamical, 0.0, 1.0)
        meff = m2 * np.power((m2 + mstar_sec)/m2, pow)

        dens = self._NFW.density(sepa, mhalo, redz)
        velo = self._NFW.velocity_circular(sepa, mhalo, redz)
        tdyn = self._NFW.time_dynamical(sepa, mhalo, redz)
        # Note: this should be returned as negative values
        dvdt = self._dvdt(meff, dens, velo)

        # convert from deceleration to hardening-rate assuming virialized orbit (i.e. ``GM/a = v^2``)
        dadt = 2 * tdyn * dvdt
        dedt = np.zeros_like(dadt)

        # Hardening rate cannot be larger than orbital/virial velocity
        clip = (np.fabs(dadt) > velo)
        if np.any(clip):
            log.info(f"clipping {utils.frac_str(clip)} `dadt` values to vcirc")
            prev = dadt[:]
            dadt[clip] = - velo[clip]
            log.debug(f"\t{utils.stats(prev*YR/PC)} ==> {utils.stats(dadt*YR/PC)}")
            del prev

        if attenuate:
            atten = self._attenuation_BBR1980(sepa, mass, mstar)
            dadt = dadt / atten

        return dadt, dedt

    def _attenuation_BBR1980(self, sepa, m1m2, mstar):
        """Calculate attentuation factor following [BBR1980]_ prescription.

        Characteristic radii are currently calculated using hard-coded Dehnen stellar-density profiles, and a fixed
        scaling-relationship between stellar-mass and stellar characteristic radius.

        The binding radius can be calculated either using the stellar density profile, or from a velocity dispersion,
        based on the `self._rbound_from_density` flag.  See the 'arguments' section of `docs::Dynamical_Friction_NFW`.

        The attenuation factor is defined as >= 1.0, with 1.0 meaning no attenuation.

        Parameters
        ----------
        sepa : (N,) array-like of scalar,
            Binary separations in units of [cm].
        m1m2 : (N, 2) array-like of scalar,
            Masses of each binary component (0-primary, 1-secondary).
        mstar : (N,) array-like of scalar,
            Mass of the stellar-bulge / stellar-core (ambiguous).

        Returns
        -------
        atten : (N,) np.ndarray of scalar
            Attenuation factor (defined as >= 1.0).

        """

        m1, m2 = m1m2.T
        mbh = m1 + m2

        # characteristic stellar radius in [cm]
        rstar = _radius_stellar_characteristic_dabringhausen_2008(mstar)
        # characteristic hardening-radius in [cm]
        rhard = _radius_hard_BBR1980_dehnen(mbh, mstar)
        # characteristic loss-cone-radius in [cm]
        rlc = _radius_loss_cone_BBR1980_dehnen(mbh, mstar)

        # Calculate R_bound based on stellar density profile (mass enclosed)
        if self._rbound_from_density:
            rbnd = _radius_influence_dehnen(mbh, mstar)
        # Calculate R_bound based on uniform velocity dispersion (MBH scaling relation)
        else:
            vdisp = self._gbh.vdisp_from_mbh(m1)   # use primary-bh's mass (index 0)
            rbnd = NWTG * mbh / vdisp**2

        # Number of stars in the stellar bulge/core
        nstar = mstar / (0.6 * MSOL)
        # --- Attenuation for separations less than the hardening radius
        # [BBR1980] Eq.3
        atten_hard = np.maximum((rhard/sepa) * np.log(nstar), np.square(mbh/mstar) * nstar)
        # use an exponential turn-on at larger radii
        cut = np.exp(-sepa/rhard)
        atten_hard *= cut

        # --- Attenuation for separations less than the loss-cone Radius
        # [BBR1980] Eq.2
        atten_lc = np.power(m2/m1, 1.75) * nstar * np.power(rbnd/rstar, 6.75) * (rlc / sepa)
        atten_lc = np.maximum(atten_lc, 1.0)
        # use an exponential turn-on at larger radii
        cut = np.exp(-sepa/rlc)
        atten_hard *= cut

        atten = np.maximum(atten_hard, atten_lc)
        # Make sure that attenuation is always >= 1.0 (i.e. this never _increases_ the hardening rate)
        atten = np.maximum(atten, 1.0)

        return atten


class Fixed_Time(_Hardening):
    """
    """

    _INTERP_NUM_POINTS = 1e4             # number of random data points used to construct interpolant
    _INTERP_THRESH_PAD_FACTOR = 5.0      #
    _TIME_TOTAL_RMIN = 1.0e-5 * PC       # minimum radius [cm] used to calculate inspiral time

    @classmethod
    def from_pop(cls, pop, time, **kwargs):
        return cls(time, *pop.mtmr, pop.redz, pop.sepa, **kwargs)

    @classmethod
    def from_sam(cls, sam, time, sepa_init=1e4*PC, **kwargs):
        mtot, mrat, redz = [gg.ravel() for gg in sam.grid]
        return cls(time, mtot, mrat, redz, sepa_init, **kwargs)

    def __init__(self, time, mtot, mrat, redz, sepa, rchar=100.0*PC, gamma_sc=-1.0, gamma_df=+2.5, progress=True):
        """

        Parameters
        ----------
        pop : `_Population` instance
        time : scalar, callable  or  array_like[scalar]
            Total merger time of binaries, units of [sec], specifiable in the following ways:
            *   scalar : uniform merger time for all binaries
            *   callable : function `time(mtot, mrat, redz)` which returns the total merger time
            *   array_like : (N,) matching the shape of `mtot` (etc) giving the merger time for
                each binary
        rchar : scalar  or  callable
            Characteristic radius dividing two power-law regimes, in units of [cm]:
            *   scalar : uniform radius for all binaries
            *   callable : function `rchar(mtot, mrat, redz)` which returns the radius
        gamma_sc : scalar
            Power-law of hardening timescale in the stellar-scattering regime
            (small separations: r < rchar)
        gamma_df : scalar
            Power-law of hardening timescale in the dynamical-friction regime
            (large separations: r > rchar)

        """
        self._progress = progress

        # mtot, mrat = utils.mtmr_from_m1m2(pop.mass)
        # sepa = pop.sepa
        # redz = cosmo.a_to_z(pop.scafa)

        # ---- Initialize / Sanitize arguments

        # Ensure `time` is ndarray matching binary variables
        if np.isscalar(time):
            time = time * np.ones_like(mtot)
        elif callable(time):
            time = time(mtot, mrat, redz)
        elif np.shape(time) != np.shape(mtot):
            utils.error(f"Shape of `time` ({np.shape(time)}) does not match `mtot` ({np.shape(mtot)})!")

        # `rchar` must be a function of only mtot, mrat; or otherwise a fixed value
        # This is because it is not being used as an interpolation variable, only an external parameter
        # FIX: either an ndarray could be allowed when interpolation is not needed (i.e. small numbers of systems)
        #      or `rchar` could be added as an explicit interpolation variable
        if callable(rchar):
            rchar = rchar(mtot, mrat, redz)
        elif not np.isscalar(rchar):
            utils.error("`rchar` must be a scalar or callable: (`rchar(mtot, mrat)`)!")

        # ---- Calculate normalization parameter
        mtot, mrat, time, sepa = np.broadcast_arrays(mtot, mrat, time, sepa)
        if mtot.ndim != 1:
            utils.error(f"Error in input shapes (`mtot.shpae={np.shape(mtot)})")

        # If there are lots of points, construct and use an interpolant
        if len(mtot) > self._INTERP_THRESH_PAD_FACTOR * self._INTERP_NUM_POINTS:
            log.info("constructing hardening normalization interpolant")
            # both are callable as `interp(args)`, with `args` shaped (N, 4),
            # the 4 parameters are:      [log10(M/MSOL), log10(q), time/Gyr, log10(Rmax/PC)]
            # the interpolants return the log10 of the norm values
            interp, backup = self._calculate_norm_interpolant(rchar, gamma_sc, gamma_df)
            # self._interp = interp
            # self._interp_backup = backup

            points = [np.log10(mtot/MSOL), np.log10(mrat), time/GYR, np.log10(sepa/PC)]
            points = np.array(points)
            norm = interp(points.T)
            bads = ~np.isfinite(norm)
            if np.any(bads):
                msg = f"Normal interpolant failed on {utils.frac_str(bads, 4)} points.  Using backup interpolant"
                log.info(msg)
                bp = points.T[bads]
                # If scipy throws an error on the shape here, see: https://github.com/scipy/scipy/issues/4123
                # or https://stackoverflow.com/a/26806707/230468
                norm[bads] = backup(bp)
                bads = ~np.isfinite(norm)
                if np.any(bads):
                    utils.error(f"Backup interpolant failed on {utils.frac_str(bads, 4)} points!")

            norm = 10.0 ** norm

        # For small numbers of points, calculate the normalization directly
        else:
            norm = self._get_norm_chunk(time, mtot, mrat, rchar, gamma_sc, gamma_df, sepa, progress=progress)

        self._gamma_sc = gamma_sc
        self._gamma_df = gamma_df
        self._norm = norm
        self._rchar = rchar
        return

    def dadt_dedt(self, evo, step):
        mass = evo.mass[:, step, :]
        sepa = evo.sepa[:, step]
        mt, mr = utils.mtmr_from_m1m2(mass)
        dadt, dedt = self._dadt_dedt(mt, mr, sepa, self._norm, self._rchar, self._gamma_sc, self._gamma_df)
        return dadt, dedt

    def dadt(self, mt, mr, sepa):
        dadt, dedt = self._dadt_dedt(mt, mr, sepa, self._norm, self._rchar, self._gamma_sc, self._gamma_df)
        return dadt

    @classmethod
    def _dadt_dedt(cls, mt, mr, sepa, norm, rchar, g1, g2):
        m1, m2 = utils.m1m2_from_mtmr(mt, mr)
        dadt_gw = utils.gw_hardening_rate_dadt(m1, m2, sepa)

        xx = sepa / rchar
        dadt = cls.function(norm, xx, g1, g2)
        dadt = dadt + dadt_gw

        dedt = None
        return dadt, dedt

    @classmethod
    def function(cls, norm, xx, g1, g2):
        dadt = - norm * np.power(1.0 + xx, -g2-1) / np.power(xx, g1-1)
        return dadt

    @classmethod
    def _time_total(cls, norm, mt, mr, rchar, g1, g2, rmax, num=100):
        norm = np.atleast_1d(norm)
        args = [norm, mt, mr, rchar, g1, g2, rmax, cls._TIME_TOTAL_RMIN]
        args = np.broadcast_arrays(*args)
        norm, mt, mr, rchar, g1, g2, rmax, rmin = args
        if np.ndim(norm) != 1:
            raise

        rextr = np.log10([rmin, rmax]).T
        rads = np.linspace(0.0, 1.0, num)[np.newaxis, :]

        rads = rextr[:, 0, np.newaxis] + rads * np.diff(rextr, axis=1)
        rads = 10.0 ** rads

        args = [norm, mt, mr, rchar, g1, g2]
        args = [aa[:, np.newaxis] for aa in args]
        norm, mt, mr, rchar, g1, g2 = args

        dadt, _ = cls._dadt_dedt(mt, mr, rads, norm, rchar, g1, g2)

        tt = utils.trapz_loglog(- 1.0 / dadt, rads, axis=-1)
        tt = tt[:, -1]
        return tt

    @classmethod
    def _get_norm(cls, tau, *args, guess=1e0):
        def integ(norm):
            return cls._time_total(norm, *args)

        g0 = guess * np.ones_like(tau)
        test = integ(g0)
        guess = g0 * (test / tau)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rv = sp.optimize.newton(lambda xx: integ(xx) - tau, guess)
        return rv

    @classmethod
    def _get_norm_chunk(cls, tau, *args, guess=1e0, chunk=1e3, progress=True):
        if np.ndim(tau) != 1:
            raise
        chunk = int(chunk)
        size = np.size(tau)
        if size <= chunk * cls._INTERP_THRESH_PAD_FACTOR:
            return cls._get_norm(tau, *args, guess=guess)

        args = [tau, *args]
        args = np.broadcast_arrays(*args)
        tau, *args = args

        num = int(np.ceil(size / chunk))
        sol = np.zeros_like(tau)
        step_iter = range(num)
        step_iter = utils.tqdm(step_iter, desc='calculating hardening normalization') if progress else step_iter
        for ii in step_iter:
            lo = ii * chunk
            hi = np.minimum((ii + 1) * chunk, size)
            cut = slice(lo, hi)
            sol[cut] = cls._get_norm(tau[cut], *[aa[cut] for aa in args], guess=guess)

        return sol

    @classmethod
    def _calculate_norm_interpolant(cls, rchar, gamma_one, gamma_two):
        mt = [1e6, 1e11]
        mr = [1e-5, 1.0]
        td = [0.0, 20.0]
        rm = [1e3, 1e5]

        num_points = int(cls._INTERP_NUM_POINTS)
        mt = 10.0 ** np.random.uniform(*np.log10(mt), num_points) * MSOL
        mr = 10.0 ** np.random.uniform(*np.log10(mr), mt.size)
        td = np.random.uniform(*td, mt.size+1)[1:] * GYR
        rm = 10.0 ** np.random.uniform(*np.log10(rm), mt.size) * PC

        norm = cls._get_norm_chunk(td, mt, mr, rchar, gamma_one, gamma_two, rm)

        valid = np.isfinite(norm) & (norm > 0.0)
        if not np.all(valid):
            err = f"Invalid normalizations!  {utils.frac_str(valid, 4)}"
            log.error(err)
            raise ValueError(err)

        points = [mt, mr, td, rm]
        units = [MSOL, 1.0, GYR, PC]
        logs = [True, True, False, True]
        points = [pp/uu for pp, uu in zip(points, units)]
        points = [np.log10(pp) if ll else pp for pp, ll in zip(points, logs)]
        points = np.array(points).T
        interp = sp.interpolate.LinearNDInterpolator(points, np.log10(norm))
        backup = sp.interpolate.NearestNDInterpolator(points, np.log10(norm))
        return interp, backup


class Fixed_Time_SAM(Fixed_Time):

    def __init__(self, sam, time, **kwargs):
        pass


# =================================================================================================
# ====    Utility Classes and Functions    ====
# =================================================================================================


class _Quinlan1996:
    """Hardening rates from stellar scattering parametrized as in [Quinlan1996]_.

    Fits from scattering experiments must be provided as `hparam` and `kparam`.

    """

    @staticmethod
    def dadt(sepa, rho, sigma, hparam):
        """Binary hardening rate from stellar scattering.

        [Sesana2010]_ Eq.8

        Parameters
        ----------
        sepa : (N,) array-like of scalar
            Binary separation in units of [cm].
        rho : (N,) array-like of scalar
            Effective stellar-density at binary separation in units of [g/cm^3].
        sigma : (N,) array-like of scalar
            Stellar velocity-dispersion at binary separation in units of [cm/s].
        hparam : (N,) array-like of scalar
            Binary hardening efficiency parameter "H" (unitless).

        Returns
        -------
        rv : (N,) np.ndarray of scalar
            Binary hardening rate in units of [cm/s].

        """
        rv = - (sepa ** 2) * NWTG * rho * hparam / sigma
        return rv

    @staticmethod
    def dedt(sepa, rho, sigma, hparam, kparam):
        """Binary rate-of-change of eccentricity from stellar scattering.

        [Sesana2010]_ Eq.9

        Parameters
        ----------
        sepa : (N,) array-like of scalar
            Binary separation in units of [cm].
        rho : (N,) array-like of scalar
            Effective stellar-density at binary separation in units of [g/cm^3].
        sigma : (N,) array-like of scalar
            Stellar velocity-dispersion at binary separation in units of [cm/s].
        hparam : (N,) array-like of scalar
            Binary hardening efficiency parameter "H" (unitless).
        kparam : (N,) array-like of scalar
            Binary eccentricity-change efficiency parameter "K" (unitless).

        Returns
        -------
        rv : (N,) np.ndarray of scalar
            Change of eccentricity rate in units of [1/s].

        """
        rv = sepa * NWTG * rho * hparam * kparam / sigma
        return rv

    @staticmethod
    def radius_hardening(msec, sigma):
        """
        [Sesana2010]_ Eq. 10
        """
        rv = NWTG * msec / (4 * sigma**2)
        return rv


class _SHM06:
    """Fits to stellar-scattering hardening rates from [Sesana2006]_, based on the [Quinlan1996]_ formalism.

    Parameters describe the efficiency of hardening as a function of mass-ratio (`mrat`) and separation (`sepa`).

    """

    def __init__(self):
        self._bound_H = [0.0, 40.0]    # See [Sesana2006]_ Fig.3
        self._bound_K = [0.0, 0.4]     # See [Sesana2006]_ Fig.4

        # Get the data filename
        fname = os.path.join(_PATH_DATA, _SCATTERING_DATA_FILENAME)
        if not os.path.isfile(fname):
            err = f"file ({fname}) not does exist!"
            log.error(err)
            raise FileNotFoundError(err)

        # Load Data
        data = json.load(open(fname, 'r'))
        self._data = data['SHM06']
        # 'H' : Hardening Rate
        self._init_h()
        # 'K' : Eccentricity growth
        self._init_k()
        return

    def H(self, mrat, sepa_rhard):
        """Hardening rate efficiency parameter.

        Parameters
        ----------
        mrat : (N,) array-like of scalar
            Binary mass-ratio (q = M2/M1 <= 1.0).
        sepa_rhard : (N,) array-like of scalar
            Binary separation in *units of hardening radius (r_h)*.

        Returns
        -------
        hh : (N,) np.ndarray of scalar
            Hardening parameter.

        """
        xx = sepa_rhard / self._H_a0(mrat)
        hh = self._H_A(mrat) * np.power(1.0 + xx, self._H_g(mrat))
        hh = np.clip(hh, *self._bound_H)
        return hh

    def K(self, mrat, sepa_rhard, ecc):
        """Eccentricity hardening rate efficiency parameter.

        Parameters
        ----------
        mrat : (N,) array-like of scalar
            Binary mass-ratio (q = M2/M1 <= 1.0).
        sepa_rhard : (N,) array-like of scalar
            Binary separation in *units of hardening radius (r_h)*.
        ecc : (N,) array-like of scalar
            Binary eccentricity.

        Returns
        -------
        kk : (N,) np.ndarray of scalar
            Eccentricity change parameter.

        """
        use_a = (sepa_rhard / self._K_a0(mrat, ecc))
        A = self._K_A(mrat, ecc)
        g = self._K_g(mrat, ecc)
        B = self._K_B(mrat, ecc)

        # `interp2d` return a matrix of X x Y results... want diagonal of that
        # NOTE: FIX: this could be improved!!
        use_a = use_a.diagonal()
        A = A.diagonal()
        g = g.diagonal()
        B = B.diagonal()

        kk = A * np.power((1 + use_a), g) + B
        kk = np.clip(kk, *self._bound_K)
        return kk

    def _init_k(self):
        """Initialize and store the interpolants for calculating the K parameter.
        """
        data = self._data['K']
        # Get all of the mass ratios (ignore other keys)
        _kq_keys = list(data.keys())
        kq_keys = []
        for kq in _kq_keys:
            try:
                int(kq)
                kq_keys.append(kq)
            except (TypeError, ValueError):
                pass

        nq = len(kq_keys)
        if nq < 2:
            raise ValueError("Something is wrong... `kq_keys` = '{}'\ndata:\n{}".format(kq_keys, data))
        k_mass_ratios = 1.0/np.array(sorted([int(kq) for kq in kq_keys]))
        k_eccen = np.array(data[kq_keys[0]]['e'])
        ne = len(k_eccen)
        k_A = np.zeros((ne, nq))
        k_a0 = np.zeros((ne, nq))
        k_g = np.zeros((ne, nq))
        k_B = np.zeros((ne, nq))

        for ii, kq in enumerate(kq_keys):
            _dat = data[kq]
            k_A[:, ii] = _dat['A']
            k_a0[:, ii] = _dat['a0']
            k_g[:, ii] = _dat['g']
            k_B[:, ii] = _dat['B']

        self._K_A = sp.interpolate.interp2d(k_mass_ratios, k_eccen, k_A, kind='linear')
        self._K_a0 = sp.interpolate.interp2d(k_mass_ratios, k_eccen, k_a0, kind='linear')
        self._K_g = sp.interpolate.interp2d(k_mass_ratios, k_eccen, k_g, kind='linear')
        self._K_B = sp.interpolate.interp2d(k_mass_ratios, k_eccen, k_B, kind='linear')
        return

    def _init_h(self):
        """Initialize and store the interpolants for calculating the H parameter.
        """
        _dat = self._data['H']
        h_mass_ratios = 1.0/np.array(_dat['q'])
        h_A = np.array(_dat['A'])
        h_a0 = np.array(_dat['a0'])
        h_g = np.array(_dat['g'])

        self._H_A = sp.interpolate.interp1d(h_mass_ratios, h_A, kind='linear', fill_value='extrapolate')
        self._H_a0 = sp.interpolate.interp1d(h_mass_ratios, h_a0, kind='linear', fill_value='extrapolate')
        self._H_g = sp.interpolate.interp1d(h_mass_ratios, h_g, kind='linear', fill_value='extrapolate')
        return


def _get_galaxy_blackhole_relation(gbh=None):
    if gbh is None:
        gbh = holo.observations.Kormendy_Ho_2013

    if inspect.isclass(gbh):
        gbh = gbh()
    elif not isinstance(gbh, holo.observations._Galaxy_Blackhole_Relation):
        err = "`gbh` must be an instance or subclass of `holodeck.observations._Galaxy_Blackhole_Relation`!"
        log.error(err)
        raise ValueError(err)

    return gbh


def _get_stellar_mass_halo_mass_relation(smhm=None):
    if smhm is None:
        smhm = holo.observations.Behroozi_2013

    if inspect.isclass(smhm):
        smhm = smhm()
    elif not isinstance(smhm, holo.observations._StellarMass_HaloMass):
        err = "`smhm` must be an instance or subclass of `holodeck.observations._StellarMass_HaloMass`!"
        log.error(err)
        raise ValueError(err)

    return smhm


def _radius_stellar_characteristic_dabringhausen_2008(mstar, gamma=1.0):
    """
    [Chen2017]_ Eq.27 - from [Dabringhausen+2008]
    """
    rchar = 239 * PC * (np.power(2.0, 1.0/(3.0 - gamma)) - 1.0)
    rchar *= np.power(mstar / (1e9*MSOL), 0.596)
    return rchar


def _radius_influence_dehnen(mbh, mstar, gamma=1.0):
    """
    [Chen2017]_ Eq.25
    """
    rchar = _radius_stellar_characteristic_dabringhausen_2008(mstar, gamma)
    rinfl = np.power(2*mbh/mstar, 1.0/(gamma - 3.0))
    rinfl = rchar / (rinfl - 1.0)
    return rinfl


def _density_at_influence_radius_dehnen(mbh, mstar, gamma=1.0):
    """
    [Chen2017]_ Eq.26
    """
    # [Chen2017] Eq.27 - from [Dabringhausen+2008]
    rchar = _radius_stellar_characteristic_dabringhausen_2008(mstar, gamma)
    dens = mstar * (3.0 - gamma) / np.power(rchar, 3.0) / (4.0 * np.pi)
    dens *= np.power(2*mbh / mstar, gamma / (gamma - 3.0))
    return dens


def _radius_hard_BBR1980_dehnen(mbh, mstar, gamma=1.0):
    """
    [Kelley2017a]_ paragraph below Eq.8 - from [BBR1980]_
    """
    rbnd = _radius_influence_dehnen(mbh, mstar, gamma=gamma)
    rstar = _radius_stellar_characteristic_dabringhausen_2008(mstar, gamma)
    rhard = rstar * (rbnd/rstar) ** 3
    return rhard


def _radius_loss_cone_BBR1980_dehnen(mbh, mstar, gamma=1.0):
    """
    [Kelley2017a]_ Eq.9 - from [BBR1980]_
    """
    mass_of_a_star = 0.6 * MSOL
    rbnd = _radius_influence_dehnen(mbh, mstar, gamma=gamma)
    rstar = _radius_stellar_characteristic_dabringhausen_2008(mstar, gamma)
    rlc = np.power(mass_of_a_star / mbh, 0.25) * np.power(rbnd/rstar, 2.25) * rstar
    return rlc
