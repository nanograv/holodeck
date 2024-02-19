"""Module for binary evolution from the time of formation/galaxy-merger until BH coalescence.

#!NOTE: much of this documentation needs to be updated to reflect that much of the material in this
#!      file was moved to `holodeck.hardening`.

In `holodeck`, initial binary populations are typically defined near the time of galaxy-galaxy
merger, when two MBHs come together at roughly kiloparsec scales.  Environmental 'hardening'
mechanisms are required to dissipate orbital energy and angular momentum, allowing the binary
separation to shrink ('harden'). Typically *dynamical friction (DF)* is most important at large
scales ($\\sim \\mathrm{kpc}$).  Near where the pair of MBHs become a gravitationally-bound binary,
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
*   General

    *   evolution modifiers should act at each step, instead of after all steps?  This would be
        a way to implement a changing accretion rate, for example; or set a max/min hardening rate.
    *   re-implement "magic" hardening models that coalesce in zero change-of-redshift or fixed
        amounts of time.

*   Evolution

    *   `_sample_universe()` : sample in comoving-volume instead of redshift

References
----------
* [BBR1980]_ Begelman, Blandford & Rees 1980.
* [Chen2017]_ Chen, Sesana, & Del Pozzo 2017.
* [Kelley2017a]_ Kelley, Blecha & Hernquist 2017.
* [Quinlan1996]_ Quinlan 1996.
* [Sesana2006]_ Sesana, Haardt & Madau et al. 2006.
* [Sesana2010]_ Sesana 2010.

"""
from __future__ import annotations

# ! ===============================================================================================!
# ! -- UPDATE `_take_next_step` -- !
# ! -- write a function to set right-edge values, call it with estimate from left edge -- !
# ! -- then call it again with a revision to the estimate, using right-edge values -- !
# ! ===============================================================================================!

import numpy as np

# import kalepy as kale

# import holodeck as holo
from holodeck import utils, cosmo, log
from holodeck.discrete import population
# from holodeck.constants import PC
from holodeck.hardening import _Hardening
# from holodeck import accretion

_MAX_ECCEN_ONE_MINUS = 1.0e-6


# =================================================================================================
# ====    Evolution Class    ====
# =================================================================================================


class Evolution:
    """Base class to evolve discrete binary populations forward in time.

    NOTE: This class is primary built to be used with :class:`holodeck.population.Pop_Illustris`.

    The `Evolution` class is instantiated with a :class:`holodeck.population._Population_Discrete`
    instance, and a particular binary hardening model (subclass of :class:`_Hardening`).  It then
    numerically integrates each binary from their initial separation to coalescence, determining
    how much time that process takes, and thus the rate of redshift/time evolution.  NOTE: at
    initialization, a regular range of binary separations are chosen for each binary being evolved,
    and the integration calculates the time it takes for each step to complete.  This is unlike most
    types of dynamical integration in which there is a prescribed time-step, and the amount of
    distance (etc) traveled over that time is then calculated.

    **Initialization**: all attributes are set to empty arrays of the appropriate size.
    NOTE: the 0th step is *not* initialized at this time, it happens in :meth:`Evolution.evolve()`.

    **Evolution**: binary evolution is performed by running the :meth:`Evolution.evolve()` function.
    This function first calls :meth:`Evolution._init_step_zero()`, which sets the 0th step values,
    and then iterates over each subsequent step, calling :meth:`Evolution._take_next_step()`.  Once
    all steps are taken (integration is completed), then :meth:`Evolution._finalize()` is called,
    at which points any stored modifiers (:class:`utils._Modifier` subclasses, in the
    :attr:`Evolution._mods` attribute) are applied.

    NOTE: whenever `frequencies` are used (rest-frame or observer-frame), they refer to **orbital**
    frequencies, not GW frequencies.  For circular binaries, GW-freq = 2 * orb-freq.

    Additional Notes
    ----------------
    acc: Instance of accretion class. This supplies the method by which total accretion
         rates are divided into individual accretion rates for each BH.
         By default, accretion rates are calculated at every step as a fraction of
         the Eddington limit.
         If acc contains a path to an accretion rate file which already stores
         total accretion rates at every timestep, then we omit the step where we
         calculate mdot_total as a fraction of the Eddington limit.
         This gives the flexibility to include accretion rates motivated by e.g. Illustris
         or other cosmological simulations.

    """

    _EVO_PARS = ['mass', 'sepa', 'eccen', 'scafa', 'tlook', 'dadt', 'dedt']
    _LIN_INTERP_PARS = ['eccen', 'scafa', 'tlook', 'dadt', 'dedt']
    _SELF_CONSISTENT = None
    _STORE_FROM_POP = ['_sample_volume']

    def __init__(self, pop, hard, nsteps: int = 100, mods=None, debug: bool = False, acc=None):
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
        self._acc = acc

        # Store hardening instances as a list
        if not np.iterable(hard):
            hard = [hard, ]
        self._hard = hard

        # Make sure types look right
        if not isinstance(pop, population._Population_Discrete):
            err = f"`pop` is {pop}, must be subclass of `population._Population_Discrete`!"
            log.exception(err)
            raise TypeError(err)

        for hh in self._hard:
            good = isinstance(hh, _Hardening) or issubclass(hh, _Hardening)
            if not good:
                err = f"hardening instance is {hh}, must be subclass of `{_Hardening}`!"
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
        self.tlook = np.zeros(shape)           #: lookback time [sec], NOTE: negative after redshift zero
        self.sepa = np.zeros(shape)            #: semi-major axis (separation) [cm]
        self.mass = np.zeros(shape + (2,))     #: mass of BHs [g], 0-primary, 1-secondary
        self.mdot = np.zeros(shape + (2,))     #: accretion rate onto each component of binary [g/s]
        self.dadt = np.zeros(shape)            #: hardening rate in separation [cm/s]
        self.eccen = eccen                     #: eccentricity [], `None` if not being evolved
        self.dedt = dedt                       #: eccen evolution rate [1/s], `None` if not evolved

        # Derived and internal parameters
        self._freq_orb_rest = None
        self._evolved = False
        self._coal = None

        return

    # ==== API and Core Functions

    def evolve(self, progress=False):
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

    def at(self, xpar, targets, params=None, coal=False, lin_interp=None):
        """Interpolate evolution to the given target locations in either separation or frequency.

        The variable of interpolation is specified with the `xpar` argument.  The evolutionary
        tracks are interpolated in that variable, to the new target locations given by `targets`.
        We use 'x' to refer to the independent variable, and 'y' to refer to the dependent variable
        that is being interpolated.  Which values are interpolated are specified with the `params`
        argument.

        The behavior of this function is broken into three sub-functions, that are only used here:

        * :meth:`Evolution._at__inputs` : parse the input arguments.

        * :meth:`Evolution._at__index_frac` : find the indices in the evolutionary tracks bounding
          the target interpolation locations, and also the fractional distance to interpolate
          between them.

        * :meth:`Evolution._at__interpolate_array` : actually interpolate each parameter to a
          the target location.

        Parameters
        ----------
        xpar : str, in ['fobs', 'sepa']
            String specifying the variable of interpolation.
        targets : array_like,
            Locations to interpolate to:

            * if ``xpar == sepa`` : binary separation, units of [cm],
            * if ``xpar == fobs`` : binary orbital freq, observer-frame, units of [1/sec],

        params : None or (list of str)
            Names of the parameters that should be interpolated.
            If `None`, defaults to :attr:`Evolution._EVO_PARS` attribute.
        coal : bool,
            Only store evolution values for binaries coalescing before redshift zero.
            Interpolated values for other binaries are set to `np.nan`.
        lin_interp : None or bool,
            Interpolate parameters in linear space.

            * True : all parameters interpolated in lin-lin space.
            * False: all parameters interpolated in log-log space.
            * None : parameters are interpolated in log-log space, unless they're included in the
              :attr:`Evolution._LIN_INTERP_PARS` attribute.

        Returns
        -------
        vals : dict,
            Dictionary of arrays for each interpolated parameter.
            The returned shape is (N, T), where `T` is the number of target locations to interpolate
            to, and `N` is the total number of binaries.
            Each data array is filled with `np.nan` values if the targets are outside of its
            evolution track.  If ``coal=True``, then binaries that do *not* coalesce before redshift
            zero also have their data array values fillwed with `np.nan`.

        Notes
        -----
        * Out of bounds values are set to `np.nan`.
        * Interpolation is 1st-order in log-log space in general, but properties which are in the
          `_LIN_INTERP_PARS` array are interpolated at 1st-order in lin-lin space.  Parameters
          which can be negative should be interpolated in linear space.  Passing a boolean for the
          `lin_interp` parameter will override the behavior (see `Parameters` above).

        """
        # parse/sanitize input arguments
        xnew, xold, params, lin_interp_list, rev, squeeze = self._at__inputs(xpar, targets, params, lin_interp)

        # (N, M); scale-factors; make sure direction matches that of `xold`
        scafa = self.scafa[:, ::-1] if rev else self.scafa[...]

        # find indices between which to interpolate, and the fractional distance to go between them
        cut_idx, interp_frac, valid = self._at__index_frac(xnew, xold)

        # if we only want coalescing systems, set non-coalescing (stalling) systems to invalid
        if coal:
            valid = valid & self.coal[:, np.newaxis]

        # Valid binaries must be valid at both `bef` and `aft` indices
        # BUG: is this actually doing what it's supposed to be doing?
        for cc in cut_idx:
            valid = valid & np.isfinite(np.take_along_axis(scafa, cc, axis=-1))

        invalid = ~valid

        data = dict()
        # Interpolate each parameter to the given locations, store to `dict`
        for par in params:
            # Load the raw evolution data for this parameter, can be None or ndarray shaped (N, M) or (N, M, 2)
            yold = getattr(self, par)
            if yold is None:
                data[par] = None
                continue

            # Reverse data to match x-values, if needed
            if rev:
                yold = yold[..., ::-1]

            # interpolate
            lin_interp_flag = (par in lin_interp_list)
            ynew = self._at__interpolate_array(yold, cut_idx, interp_frac, lin_interp_flag)

            # fill 'invalid' (i.e. out of bounds, or non-coalescing binaries if ``coal==True``)
            ynew[invalid, ...] = np.nan
            # remove excess dimensions if a single target was requested (i.e. ``T=1``)
            if squeeze:
                ynew = ynew.squeeze()
            # store
            data[par] = ynew

        return data

    def _at__inputs(self, xpar, targets, params, lin_interp):
        """Parse/sanitize the inputs of the :meth:`Evolution.at` method.

        Parameters
        ----------
        xpar : str, in ['fobs', 'sepa']
            String specifying the variable of interpolation.
        targets : array_like,
            Locations to interpolate to.  One of:
            * if ``xpar == sepa``  : binary separation, units of [cm],
            * if ``xpar == fobs`` : binary orbital-frequency, observer-frame, units of [1/sec].
        params : None or list[str]
            Names of parameters that should be interpolated.
            If `None`, defaults to :attr:`Evolution._EVO_PARS` attribute.

        Returns
        -------
        xnew : np.ndarray
            (T,) Log10 of the target locations to interpolate to, i.e. ``log10(targets)``.
        xold : np.ndarray
            (N, M) Log10 of the x-values at which to evaluate the target interpolation points.
            Either ``log10(sepa)`` or ``log10(freq_orb_obs)``.
            NOTE: these values will be returned in *increasing* order.  If they have been reversed,
            then ``rev=True``.
        params : list[str]
            Names of parameters that should be interpolated.
        rev : bool
            Whether or not the `xold` array has been reversed.
        squeeze : bool
            Whether or not the `targets` were a single scalar value (i.e. ``T=1``, as opposed to an
            iterable).  If `targets` were a scalar, then the data returned from :meth:`Evolution.at`
            will be shaped as (N,) instead of (N,T); since in this case, T=1.

        """
        # Raise an error if this instance has not been evolved yet
        self._check_evolved()

        _allowed = ['sepa', 'fobs']
        if xpar not in _allowed:
            raise ValueError("`xpar` must be one of '{}'!".format(_allowed))

        if params is None:
            params = self._EVO_PARS
        if np.isscalar(params):
            params = [params]

        if lin_interp is None:
            lin_interp_list = self._LIN_INTERP_PARS
        elif isinstance(lin_interp, list):
            lin_interp_list = lin_interp
            lin_interp = None
            for ll in lin_interp_list:
                if ll not in params:
                    err = f"`lin_interp` value {ll} not in parameters list {params}!"
                    raise ValueError(err)
        elif lin_interp is True:
            lin_interp_list = params
        elif lin_interp is False:
            lin_interp_list = []
        else:
            err = f"`lin_interp` ({lin_interp}) must be `None`, boolean, or a list of parameter names!"
            raise ValueError(err)

        squeeze = False
        if np.isscalar(targets):
            targets = np.atleast_1d(targets)
            squeeze = True

        size, nsteps = self.shape

        # Observer-frame orbital frequency, units of [1/sec] = [Hz]
        if xpar == 'fobs':
            # frequency is already increasing (must be true for interplation later)
            xold = np.log10(self.freq_orb_obs)
            xnew = np.log10(targets)
            rev = False
        # Binary-Separation, units of [cm]
        elif xpar == 'sepa':
            # separation is decreasing, reverse to increasing (for interpolation)
            xold = np.log10(self.sepa)[:, ::-1]
            xnew = np.log10(targets)
            rev = True
        else:   # nocov
            # This should never be reached, we already checked `xpar` is valid above
            raise ValueError("Bad `xpar` {}!".format(xpar))

        # Make sure target values are within bounds
        textr = utils.minmax(xnew)
        xextr = utils.minmax(xold)
        if (textr[1] < xextr[0]) | (textr[0] > xextr[1]):
            err = "`targets` extrema ({}) outside `xvals` extema ({})!  Bad units?".format(
                (10.0**textr), (10.0**xextr))
            raise ValueError(err)

        return xnew, xold, params, lin_interp_list, rev, squeeze

    def _at__index_frac(self, xnew, xold):
        """Find indices bounding target locations, and the fractional distance to go between them.

        Parameters
        ----------
        xnew : np.ndarray
            Target locations to interplate to.  Shape (T,).
        xold : np.ndarray
            Values of the x-coordinate between which to interpolate.  Shape (N, M).
            These are the x-values of either `sepa` or `fobs` from the evolutionary tracks of each
            binary.

        Returns
        -------
        cut_idx : np.ndarray
            For each binary, the step-number indices between which to interpolate, for each target
            interpolation point.  shape (2, N, T); where the 0th dimension, the 0th value is the
            low/before index, and the 1th value is the high/after index.
            i.e. for binary 'i' and target 'j', we need to interpolate between indices given by
            [0, i, j] and [1, i, j].
        interp_frac : np.ndarray
            The fractional distance between the low value and the high value, to interpolate to.
            Shape (2, N, M).  For binary 'i' and target 'j', `interp_frac[i, j]` is how the
            fraction of the way, from index `cut_idx[0, i, j]` to `cut_idx[1, i, j]` to interpolate
            to, in the `data` array.
        valid : np.ndarray
            Array of boolean values, giving whether or not the target interpolation points are
            within the bounds of each binary's evolution track.  Shape (N, T).

        """
        # ---- For every binary, find the step index immediately following each target value
        # (N, T, M) | `xnew` is (T,) for T-targets,  `xold` is (N, M) for N-binaries and M-steps
        select = (xnew[np.newaxis, :, np.newaxis] <= xold[:, np.newaxis, :])
        # (N, T), xvalue index [0, M-1] following each target point (T,), for each binary (N,)
        aft = np.argmax(select, axis=-1)

        # ---- Determine which locations are 'valid' (i.e. within the evolutionary tracks)
        # zero values in `aft` mean no `xold` after the targets were found; these are 'invalid',
        # these will be converted to `np.nan` later
        valid = (aft > 0)

        # ---- get the x-value index immediately preceding each target point
        bef = np.copy(aft)
        bef[valid] -= 1
        # (2, N, T)
        cut_idx = np.array([aft, bef])

        # Get the x-values before and after the target locations  (2, N, T)
        xold_temp = [np.take_along_axis(xold, cc, axis=-1) for cc in cut_idx]
        # Find how far to interpolate between values (in log-space; `xold` was already log10'd
        #     (N, T)
        denom = np.subtract(*xold_temp)
        numer = xnew[np.newaxis, :] - xold_temp[1]
        interp_frac = np.zeros_like(numer)
        idx = (denom != 0.0)
        interp_frac[idx] = numer[idx] / denom[idx]

        return cut_idx, interp_frac, valid

    def _at__interpolate_array(self, yold, cut_idx, interp_frac, lin_interp_flag):
        """Interpolate a parameter to a fraction between integration steps.

        Parameters
        ----------
        yold : np.ndarray
            The data to be interpolated.  This is the raw evolution data, for each binary and
            each step.  Shaped either as (N, M) or (N, M, 2) if parameter is mass.
        cut_idx : np.ndarray
            For each binary, the step-number indices between which to interpolate, for each target
            interpolation point.  shape (2, N, T); where the 0th dimension, the 0th value is the
            low/before index, and the 1th value is the high/after index.
            i.e. for binary 'i' and target 'j', we need to interpolate between indices given by
            [0, i, j] and [1, i, j].
        interp_frac : np.ndarray
            The fractional distance between the low value and the high value, to interpolate to.
            Shape (2, N, M).  For binary 'i' and target 'j', `interp_frac[i, j]` is how the
            fraction of the way, from index `cut_idx[0, i, j]` to `cut_idx[1, i, j]` to interpolate
            to, in the `data` array.
        lin_interp_flag : bool,
            Whether data should be interpolated in lin-lin space (True), or log-log space (False).

        Returns
        -------
        ynew : np.ndarray
            The input `data` interpolated to the new target locations.
            Shape is (N, T) or (N, T, 2) for N-binaries, T-target points.  A third dimension is
            present if the input `data` was 3D.

        """

        reshape = False
        cut = cut_idx
        frac = interp_frac
        # Sometimes there is a third dimension for the 2 binaries (e.g. `mass`)
        #    which will have shape, (N, T, 2) --- calling this "double-data"
        if np.ndim(yold) != 2:
            if (np.ndim(yold) == 3) and (np.shape(yold)[-1] == 2):
                # Keep the interpolation axis last (N, T, 2) ==> (N, 2, T)
                yold = np.moveaxis(yold, -1, -2)
                # Expand other arrays appropriately
                cut = cut[:, :, np.newaxis]
                frac = frac[:, np.newaxis, :]
                reshape = True
            else:   # nocov
                raise ValueError("Unexpected shape of yold: {}!".format(np.shape(yold)))

        if not lin_interp_flag:
            yold = np.log10(yold)

        # (2, N, T) for scalar data or (2, N, 2, T) for "double-data"
        yold = [np.take_along_axis(yold, cc, axis=-1) for cc in cut]
        # Interpolate by `frac` for each binary   (N, T) or (N, 2, T) for "double-data"
        ynew = yold[1] + (np.subtract(*yold) * frac)
        # In the "double-data" case, move the doublet back to the last dimension
        #    (N, T) or (N, T, 2)
        if reshape:
            ynew = np.moveaxis(ynew, 1, -1)

        # fill return dictionary
        if not lin_interp_flag:
            ynew = 10.0 ** ynew

        return ynew

    def sample_universe(self, fobs_orb_edges, down_sample=None):
        """Construct a full universe of binaries based on resampling this population.

        Parameters
        ----------
        fobs : array_like,
            Observer-frame *orbital*-frequencies at which to sample population. Units of [1/sec].
        down_sample : None or float,
            Factor by which to downsample the resulting population.
            For example, `10.0` will produce 10x fewer output binaries.

        Returns
        -------
        names : (4,) list of str,
            Names of the returned binary parameters (i.e. each array in `samples` and `vals`).
        samples : (4, S) ndarray,
            Sampled binary data.  For each binary samples S, 4 parameters are returned:
            ['mtot', 'mrat', 'redz', 'fobs'] (these are listed in the `names` returned value.)
            NOTE: `fobs` is *observer*-frame *orbital*-frequencies.
            These values correspond to all of the binaries in an observer's Universe
            (i.e. light-cone), within the given frequency bins.  The number of samples `S` is
            roughly the sum of the `weights` --- but the specific number is drawn from a Poisson
            distribution around the sum of the `weights`.
        vals : (4,) list of (V,) ndarrays or float
            Binary parameters (log10 of parameters specified in the `names` return values) at each
            frequency bin.  Binaries not reaching the target frequency bins before redshift zero,
            or before coalescing, are not returned.  Thus the number of values `V` may be less than
            F*N for F frequency bins and N binaries.
        weights : (V,) ndarray of float
            The weight of each binary-frequency sample.  i.e. number of observer-universe binaries
            corresponding to this binary in the simulation, at the target frequency.

        To-Do
        -----
        * This should sample in volume instead of `redz`, see how it's done in sam module.

        """

        # these are `log10(values)` where values are in CGS units
        # names = ['mtot', 'mrat', 'redz', 'fobs']
        names, vals, weights = self._sample_universe__at_values_weights(fobs_orb_edges)

        samples = self._sample_universe__resample(fobs_orb_edges, vals, weights, down_sample)

        # Convert back to normal-space
        samples = np.asarray([10.0 ** ss for ss in samples])
        vals = np.asarray([10.0 ** vv for vv in vals])
        return names, samples, vals, weights

    def _sample_universe__at_values_weights(self, fobs_orb_edges):
        """Interpolate binary histories to target frequency bins, obtaining parameters and weights.

        The `weights` correspond to the number of binaries in an observer's Universe (light-cone)
        corresponding to each simulated binary sample.

        Arguments
        ---------
        fobs_orb_edges : (F+1,) arraylike
            Edges of target frequency bins to sample population.  These are observer-frame orbital
            frequencies.  Binaries are interpolated to frequency bin centers, calculated from the
            midpoints of the provided bin edges.

        Returns
        -------
        names : (4,) list of str,
            Names of the returned binary parameters (i.e. each array in `vals`).
        vals : (4,) list of (V,) ndarrays or float
            Binary parameters (log10 of parameters specified in the `names` return values) at each
            frequency bin.  Binaries not reaching the target frequency bins before redshift zero,
            or before coalescing, are not returned.  Thus the number of values `V` may be less than
            F*N for F frequency bins and N binaries.
        weights : (V,) ndarray of float
            The weight of each binary-frequency sample.  i.e. number of observer-universe binaries
            corresponding to this binary in the simulation, at the target frequency.

        """
        fobs_orb_cents = utils.midpoints(fobs_orb_edges, log=False)
        dlnf = np.diff(np.log(fobs_orb_edges))

        # Interpolate binaries to given frequencies; these are the needed parameters
        PARAMS = ['mass', 'sepa', 'dadt', 'scafa']
        # each array within `data_fobs` is shaped (N, F) for N-binaries and F-frequencies (`fobs`)
        data_fobs = self.at('fobs', fobs_orb_cents, params=PARAMS)

        # Only examine binaries reaching the given locations before redshift zero (other redz=infinite)
        redz = cosmo.a_to_z(data_fobs['scafa'])
        valid = (redz > 0.0)
        log.debug(f"After interpolation, valid binary-targets: {utils.frac_str(valid)}")

        # Get rest-frame orbital-frequency [1/s]
        frst_orb_cents = utils.frst_from_fobs(fobs_orb_cents[np.newaxis, :], redz)
        # Comoving distance [cm]
        dcom = cosmo.z_to_dcom(redz)

        # `mass` has shape (Binaries, Frequencies, 2), units [gram]
        #    convert to (2, B, F), then separate into m1, m2 each with shape (B, F)
        m1, m2 = np.moveaxis(data_fobs['mass'], -1, 0)
        dfdt, _ = utils.dfdt_from_dadt(data_fobs['dadt'], data_fobs['sepa'], frst_orb=frst_orb_cents)

        _lambda_factor = utils.lambda_factor_dlnf(frst_orb_cents, dfdt, redz, dcom=dcom) / self._sample_volume
        num_binaries = _lambda_factor * dlnf[np.newaxis, :]

        # select only valid entries
        mt, mr = utils.mtmr_from_m1m2(m1[valid], m2[valid])
        # broadcast `fobs` to match the shape of binaries, then select valid entries
        fo = (fobs_orb_cents[np.newaxis, :] * np.ones_like(redz))[valid]
        redz = redz[valid]
        weights = num_binaries[valid]
        log.debug(f"Weights (lambda values) at targets: {utils.stats(weights)}")

        # Convert to log-space
        vals = [np.log10(mt), np.log10(mr), np.log10(redz), np.log10(fo)]
        names = ['mtot', 'mrat', 'redz', 'fobs']
        return names, vals, weights

    def _sample_universe__resample(self, fobs_orb_edges, vals, weights, down_sample):
        import kalepy as kale
        # down-sample weights to decrease the number of sample points
        prev_sum = weights.sum()
        log.info(f"Total weights (number of binaries in the universe): {prev_sum:.8e}")
        if down_sample is not None:
            weights = weights / down_sample
            next_sum = weights.sum()
            msg = f"downsampling artificially: down_sample={down_sample:g} :: total: {prev_sum:.4e}==>{next_sum:.4e}"
            log.warning(msg)

        # TODO/FIX: Consider sampling in comoving-volume instead of redz (like in sam.py)
        #           can also return dcom instead of redz for easier strain calculation
        nsamp = np.random.poisson(weights.sum())
        reflect = [None, [None, 0.0], None, np.log10([fobs_orb_edges[0], fobs_orb_edges[-1]])]
        samples = kale.resample(vals, size=nsamp, reflect=reflect, weights=weights, bw_rescale=0.5)
        # samples = np.power(10.0, samples)
        num_samp = samples[0].size
        log.debug(f"Sampled {num_samp:.8e} binaries in the universe")
        return samples

    # ==== Internal Methods

    def _init_step_zero(self):
        """Set the initial conditions of the binaries at the 0th step.

        Transfers attributes from the stored :class:`holodeck.population._Population_Discrete`
        instance to the 0th index of the evolution arrays.  The attributes are [`sepa`, `scafa`,
        `mass`, and optionally `eccen`].  The hardening model is also used to calculate the 0th
        hardening rates `dadt` and `dedt`.  The initial lookback time, `tlook` is also set.

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
        tlook = cosmo.z_to_tlbk(redz)
        self.tlook[:, 0] = tlook
        # `pop.mass` has shape (N, 2), broadcast to (N, S, 2) for `S` steps
        # self.mass[:, :, :] = pop.mass[:, np.newaxis, :]
        self.mass[:, 0, :] = pop.mass
        # HERE INITIAL MASSES ARE COPIED FOR EVERY STEP
        self.mass[:, :, :] = self.mass[:, 0, np.newaxis, :]

        if self._debug:    # nocov
            for ii, hard in enumerate(self._hard):
                # Store individual hardening rates
                setattr(self, f"_dadt_{ii}", np.zeros_like(self.dadt))
                setattr(self, f"_dedt_{ii}", np.zeros_like(self.dadt))

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

        # get the separation $a$ on both edges
        sepa = self.sepa[:, (right, left)]   # sepa is decreasing, so switch left-right order

        # ! ====================================================================
        # ---- Hardening rates at the left-edge of the step
        # calculate
        dadt_l, dedt_l = self._hardening_rate(left, store_debug=False)
        da = np.diff(sepa, axis=-1)
        da = da[:, 0]
        dt = da / -dadt_l
        if np.any(dt < 0.0):    # nocov
            err = f"Negative time-steps found at step={step}!"
            log.exception(err)
            raise ValueError(err)

        if self.eccen is not None:
            de = dedt_l * dt
            ecc_r = self.eccen[:, left] + de
            ecc_r = np.clip(ecc_r, 0.0, 1.0 - _MAX_ECCEN_ONE_MINUS)
            self.eccen[:, right] = ecc_r

        # Update lookback time based on duration of this step
        tlook = self.tlook[:, left] - dt
        self.tlook[:, right] = tlook
        # update scale-factor for systems at z > 0.0 (i.e. a < 1.0 and tlook > 0.0)
        val = (tlook > 0.0)
        self.scafa[val, right] = cosmo.z_to_a(cosmo.tlbk_to_z(tlook[val]))
        # set systems after z = 0 to scale-factor of unity
        self.scafa[~val, right] = 1.0
        # ! ====================================================================

        # ---- Hardening rates at the right-edge of the step
        # calculate
        dadt_r, dedt_r = self._hardening_rate(right, store_debug=True)

        # store
        self.dadt[:, right] = dadt_r
        if self.eccen is not None:
            self.dedt[:, right] = dedt_r

        # ---- Calculate time between edges

        # get the $dt/da$ rate on both edges of the step
        dtda = 1.0 / - self.dadt[:, (left, right)]   # NOTE: `dadt` is negative, convert to positive
        # use trapezoid rule to find total time for this step
        dt = utils.trapz_loglog(dtda, sepa, axis=-1).squeeze()   # this should come out positive
        if np.any(dt < 0.0):    # nocov
            err = f"Negative time-steps found at step={step}!"
            log.exception(err)
            raise ValueError(err)

        # ---- Update right-edge values
        # NOTE/ENH: this would be a good place to make a function `_update_right_edge()` (or something like that),
        # that stores the updated right edge values, and also performs any additional updates, such as mass evolution

        # Update lookback time based on duration of this step
        tlook = self.tlook[:, left] - dt
        self.tlook[:, right] = tlook
        # update scale-factor for systems at z > 0.0 (i.e. a < 1.0 and tlook > 0.0)
        val = (tlook > 0.0)
        self.scafa[val, right] = cosmo.z_to_a(cosmo.tlbk_to_z(tlook[val]))
        # set systems after z = 0 to scale-factor of unity
        self.scafa[~val, right] = 1.0

        # update eccentricity if it's being evolved
        if self.eccen is not None:
            dedt = self.dedt[:, (left, right)]
            time = self.tlook[:, (right, left)]   # tlook is decreasing, so switch left-right order
            # decc = utils.trapz_loglog(dedt, time, axis=-1).squeeze()
            decc = utils.trapz(dedt, time, axis=-1).squeeze()
            ecc_r = self.eccen[:, left] + decc
            ecc_r = np.clip(ecc_r, 0.0, 1.0 - _MAX_ECCEN_ONE_MINUS)
            self.eccen[:, right] = ecc_r
            if self._debug:    # nocov
                bads = ~np.isfinite(decc)
                if np.any(bads):
                    utils.print_stats(print_func=log.error, dedt=dedt, time=time, decc=decc)
                    err = f"Non-finite changes in eccentricity found in step {step}!"
                    log.exception(err)
                    raise ValueError(err)

        if self._acc is not None:
            """ An instance of the accretion class has been supplied,
                and binary masses are evolved through accretion
                First, get total accretion rates """

            mdot_t = self._acc.mdot_total(self, step)
            """ A preferential accretion model is called to divide up
                total accretion rates into primary and secondary accretion rates """
            self.mdot[:,step-1,:] = self._acc.pref_acc(mdot_t, self, step)
            """ Accreted mass is calculated and added to primary and secondary masses """
            self.mass[:, step, 0] = self.mass[:, step-1, 0] + dt * self.mdot[:,step-1,0]
            self.mass[:, step, 1] = self.mass[:, step-1, 1] + dt * self.mdot[:,step-1,1]

        return

    def _hardening_rate(self, step, store_debug=True):
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
            _hard_dadt, _ecc = hard.dadt_dedt(self, step)
            dadt[:] += _hard_dadt
            if self._debug:    # nocov
                log.debug(f"{step} hard={hard} : dadt = {utils.stats(_hard_dadt)}")
                # Store individual hardening rates
                if store_debug:
                    getattr(self, f"_dadt_{ii}")[:, step] = _hard_dadt[...]
                # Raise error on invalid entries
                bads = ~np.isfinite(_hard_dadt) | (_hard_dadt > 0.0)
                if np.any(bads):
                    log.error(f"{step} hard={hard} : dadt = {utils.stats(_hard_dadt)}")
                    err = f"invalid `dadt` for hard={hard}  (bads: {utils.frac_str(bads)})!"
                    log.exception(err)
                    log.error(f"BAD dadt = {_hard_dadt[bads]}")
                    log.error(f"BAD sepa = {self.sepa[bads, step]}")
                    log.error(f"BAD mass = {self.sepa[bads, step]}")
                    raise ValueError(err)

            if (self.eccen is not None):
                if _ecc is None:
                    log.warning(f"`Evolution.eccen` is not None, but `dedt` is None!  {step} {hard}")
                    continue
                dedt[:] += _ecc
                if self._debug:    # nocov
                    log.debug(f"{step} hard={hard} : dedt = {utils.stats(_ecc)}")
                    # Raise error on invalid entries
                    if not np.all(np.isfinite(_ecc)):
                        err = f"invalid `dedt` for hard={hard}!"
                        log.exception(err)
                        raise ValueError(err)
                    # Store individual hardening rates
                    if store_debug:
                        getattr(self, f"_dedt_{ii}")[:, step] = _ecc[...]

        return dadt, dedt

    def _check(self):
        """Perform basic diagnostics on parameter validity after evolution.
        """
        _check_var_names = ['sepa', 'scafa', 'mass', 'tlook', 'dadt']
        _check_var_names_eccen = ['eccen', 'dedt']

        def check_vars(names):
            for cv in names:
                vals = getattr(self, cv)
                if np.any(~np.isfinite(vals)):    # pragma: no cover
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
        """The number of binaries and number of steps (N, S)."""
        return self._shape

    @property
    def size(self):
        """The number of binaries"""
        return self._shape[0]

    @property
    def steps(self):
        """The number of evolution steps"""
        return self._shape[1]

    @property
    def coal(self):
        """Indices of binaries that coalesce before redshift zero.
        """
        if self._coal is None:
            self._coal = (self.scafa[:, -1] < 1.0)
        return self._coal

    @property
    def tage(self):
        """Age of the universe [sec] for each binary-step.

        Derived from :attr:`Evolution.tlook`.

        Returns
        -------
        ta : np.ndarray,
            (B, S).  Age of the universe.

        """
        ta = cosmo.age(0.0).cgs.value - self.tlook
        return ta

    @property
    def mtmr(self):
        """Total-mass and mass-ratio.

        Returns
        -------
        mt : np.ndarray
            Total mass ($M = m_1 + m_2$) in [gram].
        mr : np.ndarray
            Mass ratio ($q = m_2/m_1 \\leq 1.0$).

        """
        mass = np.moveaxis(self.mass, -1, 0)   # (N, M, 2) ==> (2, N, M)
        mt, mr = utils.mtmr_from_m1m2(*mass)
        return mt, mr

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
