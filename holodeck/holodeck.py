"""
"""

import abc
import enum

import numpy as np
import tqdm

# import holodeck as holo
from holodeck import utils, cosmo
from holodeck.constants import MSOL, PC, GYR, SPLC

_DEF_ECCS_DIST = (1.0, 0.2)
_DEF_TIME_DELAY = (5.0*GYR, 0.2)
_CALC_MC_PARS = ['mass', 'sepa', 'dadt', 'time', 'eccs']


# ============    Population    ============


class _Binary_Population(abc.ABC):

    def __init__(self, fname, *args, mods=None, check=True, **kwargs):
        self._fname = fname

        # Initial binary values (i.e. at time of formation)
        self.time = None    # scale factor        (N,)
        self.sepa = None    # binary separation a (N,)
        self.mass = None    # blackhole masses    (N, 2)
        self._size = None

        self.eccs = None
        self._sample_volume = None

        # Initialize the population
        self._init_from_file(fname)
        # Apply modifications (using `Modifer` instances)
        self.modify(mods)
        # Perform diagnostics
        if check:
            self._check()
        return

    @abc.abstractmethod
    def _init_from_file(self, fname):
        pass

    @abc.abstractmethod
    def _update_derived(self):
        pass

    @property
    def size(self):
        return self._size

    def modify(self, mods=None):
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

    def _check(self):
        return


class BP_Illustris(_Binary_Population):

    def _init_from_file(self, fname):
        header, data = utils.load_hdf5(fname)
        self._sample_volume = header['box_volume_mpc'] * (1e6*PC)**3

        # Select the stellar radius
        part_names = header['part_names'].tolist()
        gal_rads = data['SubhaloHalfmassRadType']
        st_idx = part_names.index('star')
        gal_rads = gal_rads[:, st_idx, :]
        # Set initial separation to sum of stellar half-mass radii
        self.sepa = np.sum(gal_rads, axis=-1)

        self.mbulge = data['SubhaloMassInRadType'][:, st_idx, :]
        self.vdisp = data['SubhaloVelDisp']
        self.mass = data['SubhaloBHMass']
        self.time = data['time']
        return

    def _update_derived(self):
        self._size = self.sepa.size
        return


class _Modifier(abc.ABC):

    def __call__(self, base):
        self.modify(base)
        return

    @abc.abstractmethod
    def modify(self, base):
        pass


class Population_Modifier(_Modifier):
    pass


class PM_Eccentricity(Population_Modifier):

    def __init__(self, eccs_dist=_DEF_ECCS_DIST):
        self.eccs_dist = eccs_dist
        return

    def modify(self, pop):
        eccs_dist = self.eccs_dist
        size = pop.size
        eccs = eccs_func(*eccs_dist, size)
        pop.eccs = eccs
        return


class PM_Resample(Population_Modifier):

    _DEF_ADDITIONAL_KEYS = ['vdisp', 'mbulge']

    def __init__(self, resample=10.0, plot=False, additional_keys=True):
        self.resample = resample
        self._plot = plot
        self._old_data = None
        self._new_data = None

        if additional_keys is True:
            additional_keys = self._DEF_ADDITIONAL_KEYS
        elif additional_keys in [False, None]:
            additional_keys = []
        self._additional_keys = additional_keys
        return

    def modify(self, pop):
        import kalepy as kale

        mt, mr = utils.mtmr_from_m1m2(pop.mass)
        labels = ['mtot', 'mrat', 'redz', 'sepa']
        old_data = [
            np.log10(mt / MSOL),
            np.log10(mr),
            pop.time,      # resample linearly in scale-factor
            np.log10(pop.sepa / PC)
        ]
        reflect = [
            None,
            [None, 0.0],
            [0.0, 1.0],
            None,
        ]

        eccs = pop.eccs
        if eccs is not None:
            labels.append('eccs')
            old_data.append(eccs)
            reflect.append([0.0, 1.0])

        opt_idx = []
        for ii, opt in enumerate(self._additional_keys):
            vals = getattr(pop, opt, None)
            if vals is not None:
                idx = len(labels) + ii
                opt_idx.append(idx)
                labels.append(opt)
                for kk in range(2):
                    old_data.append(np.log10(vals[:, kk]))
                    reflect.append(None)
            else:
                opt_idx.append(None)

        resample = self.resample
        old_size = pop.size
        new_size = old_size * resample

        kde = kale.KDE(old_data, reflect=reflect)
        new_data = kde.resample(new_size)

        mt = MSOL * 10**new_data[0]
        mr = 10**new_data[1]

        pop.mass = utils.m1m2_from_mtmr(mt, mr).T
        pop.time = new_data[2]
        pop.sepa = PC * 10**new_data[3]
        pop.eccs = None if (eccs is None) else new_data[4]

        for opt, idx in zip(self._additional_keys, opt_idx):
            if idx is None:
                continue
            size = new_data[idx].size
            temp = np.zeros((size, 2))
            for kk in range(2):
                temp[:, kk] = np.power(10.0, new_data[idx+kk])
            setattr(pop, opt, temp)

        pop._sample_volume *= resample

        if self._plot:
            self._labels = labels
            self._old_data = old_data
            self._new_data = new_data

        return

    def plot(self):
        import kalepy as kale
        dold = self._old_data
        dnew = self._new_data
        labels = self._labels
        if (dold is None) or (dnew is None):
            raise ValueError("Stored data is empty!")

        corner = kale.Corner(len(dold), labels=labels)
        kw = dict(scatter=False, contour=True, probability=True)
        corner.plot_clean(dnew, color='blue', **kw)
        corner.plot_clean(dold, color='red', **kw)
        return corner.fig


class PM_Mass_Reset(Population_Modifier):
    """
    """

    FITS = {}
    NORM = {}
    _VALID_RELATIONS = ['vdisp', 'mbulge']

    def __init__(self, relation, alpha=None, beta=None, eps=None, scatter=1.0):
        relation = relation.strip().lower()

        if relation not in self._VALID_RELATIONS:
            err = f"`relation` {relation} must be one of '{self._VALID_RELATIONS}'!"
            raise ValueError(err)

        self.relation = relation
        if scatter in [None, False]:
            scatter = 0.0
        elif scatter is True:
            scatter = 1.0

        self.scatter = scatter
        fits = self.FITS[relation]
        if alpha is None:
            alpha = fits['alpha']
        if beta is None:
            beta = fits['beta']
        if eps is None:
            eps = fits['eps']

        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        return

    def modify(self, pop):
        relation = self.relation
        vals = getattr(pop, relation, None)
        if vals is None:
            err = (
                f"relation is set to '{relation}', "
                f"but value is not set in population instance!"
            )
            raise ValueError(err)

        shape = (pop.size, 2)
        scatter = self.scatter
        alpha = self.alpha
        beta = self.beta
        eps = self.eps

        norm = self.NORM[relation]
        x0 = norm['x']
        y0 = norm['y']

        params = [alpha, beta, [0.0, eps]]
        for ii, vv in enumerate(params):
            if (scatter > 0.0):
                vv = np.random.normal(vv[0], vv[1]*scatter, size=shape)
            else:
                vv = vv[0]

            params[ii] = vv

        alpha, beta, eps = params
        mass = alpha + beta * np.log10(vals/x0) + eps
        mass = np.power(10.0, mass) * y0
        # Store old version
        pop._mass = pop.mass
        pop.mass = mass
        return


class PM_MM13(PM_Mass_Reset):
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


class PM_KH13(PM_Mass_Reset):
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


# ============    Evolution    ============


@enum.unique
class EVO(enum.Enum):
    CONT = 1
    END = -1


class _Binary_Evolution(abc.ABC):

    _EVO_PARS = ['mass', 'sepa', 'eccs', 'redz', 'dadt', 'tlbk']
    _LIN_INTERP_PARS = ['eccs', 'redz', 'tlbk']
    _SELF_CONSISTENT = None

    def __init__(self, bin_pop, nsteps=100, mods=None, check=True):
        self._bin_pop = bin_pop
        self._nsteps = nsteps
        self._mods = mods

        size = bin_pop.size
        shape = (size, nsteps)

        self._shape = shape
        self.time = np.zeros(shape)
        self.tlbk = np.zeros(shape)
        self.sepa = np.zeros(shape)
        self.mass = np.zeros(shape + (2,))

        if bin_pop.eccs is not None:
            self.eccs = np.zeros(shape)
            self.dedt = np.zeros(shape)
        else:
            self.eccs = None
            self.dedt = None

        # NOTE: these values should be stored as positive values
        self.dadt = np.zeros(shape)

        # Derived parameters
        self._freq_orb_rest = None
        self._evolved = False
        self._coal = None

        return

    @property
    def shape(self):
        return self._shape

    def evolve(self):
        # ---- Initialize Integration Step Zero
        self._init_step_zero()

        # ---- Iterate through all integration steps
        size, nsteps = self.shape
        steps_list = range(1, nsteps)
        for step in steps_list:
            rv = self._take_next_step(step)
            if rv is EVO.END:
                self._evolved = True
                break
            elif rv not in EVO:
                raise ValueError("Recieved bad `rv` ({}) after step {}!".format(rv, step))

        # ---- Finalize
        self._finalize()
        return

    @property
    def coal(self):
        if self._coal is None:
            self._coal = (self.redz[:, -1] > 0.0)
        return self._coal

    def _init_step_zero(self):
        bin_pop = self._bin_pop
        _, nsteps = self.shape

        # Initialize ALL separations ranging from initial to mutual-ISCO, for each binary
        rad_isco = utils.rad_isco(*bin_pop.mass.T)
        # (2, N)
        sepa = np.log10([bin_pop.sepa, rad_isco])
        # Get log-space range of separations for each of N ==> (N, nsteps)
        sepa = np.apply_along_axis(lambda xx: np.logspace(*xx, nsteps), 0, sepa).T
        self.sepa[:, :] = sepa

        self.time[:, 0] = bin_pop.time
        redz = utils.a_to_z(bin_pop.time)
        tlbk = cosmo.z_to_tlbk(redz)
        self.tlbk[:, 0] = tlbk
        self.mass[:, 0, :] = bin_pop.mass

        if (bin_pop.eccs is not None):
            self.eccs[:, 0] = bin_pop.eccs

        return

    @abc.abstractmethod
    def _take_next_step(self, ii):
        pass

    def _check(self):
        _check_var_names = ['sepa', 'time', 'mass', 'tlbk', 'dadt']
        _check_var_names_eccs = ['eccs', 'dedt']

        def check_vars(names):
            for cv in names:
                vals = getattr(self, cv)
                if np.any(~np.isfinite(vals)):
                    err = "Found non-finite '{}' !".format(cv)
                    raise ValueError(err)

        check_vars(_check_var_names)

        if self.eccs is None:
            return

        check_vars(_check_var_names_eccs)

        return

    def _finalize(self):
        self._evolved = True

        self.modify()

        # Run diagnostics
        self._check()

        return

    def modify(self, mods=None):
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

    def _update_derived(self):
        pass

    @property
    def freq_orb_rest(self):
        if self._freq_orb_rest is None:
            self._check_evolved()
            mtot = self.mass.sum(axis=-1)
            self._freq_orb_rest = utils.kepler_freq_from_sep(mtot, self.sepa)
        return self._freq_orb_rest

    @property
    def freq_orb_obs(self):
        redz = utils.a_to_z(self.time)
        fobs = self.freq_orb_rest / (1.0 + redz)
        return fobs

    def _check_evolved(self):
        if self._evolved is not True:
            raise RuntimeError("This instance has not been evolved yet!")

        return

    def at(self, xpar, targets, pars=None, coal=False):
        """Interpolate evolution to the given observed, orbital frequency.

        Arguments
        ---------
        xpar : str, one of ['fobs', 'sepa']
            String specifying the variable to interpolate to.
        targets : array of scalar
            Locations to interpolate to.
            `sepa` : units of cm
            `fobs` : units of 1/s [Hz]

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
            time = self.time[:, :]
            tt = np.log10(targets)
            rev = False
        # Binary-Separation, units of pc
        elif xpar == 'sepa':
            # separation is decreasing, reverse to increasing
            _xvals = np.log10(self.sepa)[:, ::-1]
            time = self.time[:, ::-1]
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
            select = select & (time[:, np.newaxis, :] > 0.0) & (time[:, np.newaxis, :] < 1.0)

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
            valid = valid & np.isfinite(np.take_along_axis(time, cc, axis=-1))

        inval = ~valid
        # Get the x-values before and after the target locations  (2, N, T)
        xvals = [np.take_along_axis(_xvals, cc, axis=-1) for cc in cut]
        # Find how far to interpolate between values (in log-space)
        #     (N, T)
        frac = (tt[np.newaxis, :] - xvals[1]) / np.subtract(*xvals)

        vals = dict()
        # Interpolate each target parameter
        for pp in pars:
            lin_interp = (pp in self._LIN_INTERP_PARS)
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

            if not lin_interp:
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

            if np.any(~np.isfinite(new[valid, ...])):
                raise ValueError("Non-finite values after interpolation of '{}'".format(pp))

            # fill return dictionary
            if not lin_interp:
                new = 10.0 ** new
            if squeeze:
                new = new.squeeze()
            vals[pp] = new

            if np.any(~np.isfinite(new[valid, ...])):
                raise ValueError("Non-finite values after exponentiation of '{}'".format(pp))

        return vals


class BE_Magic_Delay(_Binary_Evolution):

    _SELF_CONSISTENT = False

    def __init__(self, *args, time_delay=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._time_delay = _parse_log_norm_pars(time_delay, self.shape[0], default=_DEF_TIME_DELAY)
        return

    def _init_step_zero(self):
        super()._init_step_zero()
        nbins, nsteps = self.shape
        mass = self.mass[:, 0, :]
        tlbk = self.tlbk[:, 0]
        sepa = self.sepa
        m1, m2 = mass.T

        # Time delay (N,)
        dtime = self._time_delay
        # Instantly step-forward this amount, discontinuously
        tlbk = np.clip(tlbk - dtime, 0.0, None)
        self.tlbk[:, 1:] = tlbk[:, np.newaxis]
        # calculate new scalefactors
        redz = cosmo.tlbk_to_z(tlbk)
        self.time[:, 1:] = utils.z_to_a(redz)[:, np.newaxis]

        # Masses don't evolve
        self.mass[:, :, :] = mass[:, np.newaxis, :]
        self.dadt[...] = - utils.gw_hardening_rate_dadt(m1[:, np.newaxis], m2[:, np.newaxis], sepa)
        return

    def _take_next_step(self, step):
        return EVO.END


# ============    Gravitational Waves    ==============


class GWB:

    def __init__(self, bin_evo, freqs, box_vol_cgs, nharms=30, nreals=100, calculate=True):
        self.freqs = freqs
        self.nharms = nharms
        self.nreals = nreals
        self._box_vol_cgs = box_vol_cgs
        self._bin_evo = bin_evo

        if calculate:
            self.calculate(bin_evo)

        return

    def calculate(self, bin_evo, eccen=None, stats=False, progress=True, nloudest=5):
        freqs = self.freqs
        nharms = self.nharms
        nreals = self.nreals
        bin_evo = self._bin_evo
        box_vol = self._box_vol_cgs

        if eccen is None:
            eccen = (bin_evo.eccs is not None)

        if eccen not in [True, False]:
            raise ValueError("`eccen` '{}' is invalid!".format(eccen))

        eccen_fore = np.zeros((freqs.size, nreals))
        eccen_back = np.zeros((freqs.size, nreals))
        eccen_both = np.zeros((freqs.size, nreals))
        circ_fore = np.zeros((freqs.size, nreals))
        circ_back = np.zeros((freqs.size, nreals))
        circ_both = np.zeros((freqs.size, nreals))
        loudest = np.zeros((freqs.size, nloudest, nreals))
        sa_eccen = np.zeros_like(freqs)
        sa_circ = np.zeros_like(freqs)

        if eccen:
            harm_range = range(1, nharms+1)
        else:
            harm_range = [2]

        for ii, fobs in tqdm.tqdm(enumerate(freqs), total=len(freqs)):
            rv = _calc_mc_at_fobs(
                fobs, harm_range, nreals, bin_evo, box_vol,
                loudest=nloudest
            )
            mc_ecc, mc_circ, ret_sa_ecc, ret_sa_circ, loud = rv
            eccen_fore[ii, :] = mc_ecc[0]
            eccen_back[ii, :] = mc_ecc[1]
            eccen_both[ii, :] = mc_ecc[2]
            circ_fore[ii, :] = mc_circ[0]
            circ_back[ii, :] = mc_circ[1]
            circ_both[ii, :] = mc_circ[2]
            sa_eccen[ii] = ret_sa_ecc
            sa_circ[ii] = ret_sa_circ
            loudest[ii, :] = loud

        self.eccen_fore = np.sqrt(eccen_fore)
        self.eccen_back = np.sqrt(eccen_back)
        self.eccen_both = np.sqrt(eccen_both)

        self.circ_fore = np.sqrt(circ_fore)
        self.circ_back = np.sqrt(circ_back)
        self.circ_both = np.sqrt(circ_both)

        self.sa_eccen = np.sqrt(sa_eccen)
        self.sa_circ = np.sqrt(sa_circ)
        self.loudest = np.sqrt(loudest)

        return


def _calc_mc_at_fobs(fobs, harm_range, nreals, bin_evo, box_vol, loudest=5):
    """
    """

    # ---- Interpolate data to all harmonics of this frequency
    harm_range = np.asarray(harm_range)
    # Each parameter will be (N, H) = (binaries, harmonics)
    data_harms = bin_evo.at('fobs', fobs / harm_range, pars=_CALC_MC_PARS)

    # Only examine binaries reaching the given locations before redshift zero (other redz=inifinite)
    redz = data_harms['time']
    redz = utils.a_to_z(redz)
    valid = np.isfinite(redz) & (redz > 0.0)

    # Broadcast harmonics numbers to correct shape
    harms = np.ones_like(redz, dtype=int) * harm_range[np.newaxis, :]
    # Select the elements corresponding to the n=2 (circular) harmonic, to use later
    sel_n2 = np.zeros_like(redz, dtype=int)
    sel_n2[(harms == 2)] = 1
    # Select only the valid elements, also converts to 1D, i.e. (N, H) ==> (V,)
    sel_n2 = sel_n2[valid]
    harms = harms[valid]
    redz = redz[valid]
    # If there are eccentricities, calculate the freq-dist-function
    eccs = data_harms['eccs']
    if eccs is None:
        gne = 1
    else:
        gne = utils.gw_freq_dist_func(harms, ee=eccs[valid])
        # BUG: FIX: NOTE: this fails for zero eccentricities (at times?) fix manually!
        sel_e0 = (eccs[valid] == 0.0)
        gne[sel_e0] = 0.0
        gne[sel_n2 & sel_e0] = 1.0

    # Calculate required parameters for valid binaries (V,)
    dlum = cosmo.z_to_dlum(redz)
    zp1 = redz + 1
    frst_orb = fobs * zp1 / harms
    mchirp = data_harms['mass'][valid]
    mchirp = utils.chirp_mass(*mchirp.T)
    # NOTE: `dadt` is stored as positive values
    dfdt = utils.dfdt_from_dadt(
        -data_harms['dadt'][valid], data_harms['sepa'][valid], freq_orb=frst_orb)
    _tres = frst_orb / dfdt

    # Calculate strains from each source
    hs2 = utils.gw_strain_source(mchirp, dlum, frst_orb)**2
    # Calculate resampling factors
    vfac = 4.0*np.pi*SPLC * dlum**2 / box_vol   # * thub
    tfac = _tres  # / thub

    # Calculate weightings
    #    Sesana+08, Eq.10
    num_frac = vfac * tfac * zp1
    try:
        num_pois = np.random.poisson(num_frac, (nreals, num_frac.size)).T
    except:
        print(f"{dlum=}")
        print(f"{redz=}")
        print(f"{vfac=}")
        print(f"{tfac=}")
        print(f"{zp1=}")
        print(f"{num_frac=}")
        raise

    # --- Calculate GW Signals
    temp = hs2 * gne * (2.0 / harms)**2
    mc_ecc_both = np.sum(temp[:, np.newaxis] * num_pois, axis=0)
    mc_circ_both = np.sum(temp[:, np.newaxis] * num_pois * sel_n2[:, np.newaxis], axis=0)

    sa_ecc = np.sum(temp * num_frac, axis=0)
    sa_circ = np.sum(temp * num_frac * sel_n2, axis=0)

    if np.count_nonzero(num_pois) > 0:
        # Find the L loudest binaries in each realizations
        loud = np.sort(temp[:, np.newaxis] * (num_pois > 0), axis=0)[::-1, :]
        mc_ecc_fore = loud[0, :]
        loud = loud[:loudest, :]

        mc_circ_fore = np.max(temp[:, np.newaxis] * (num_pois > 0) * sel_n2[:, np.newaxis], axis=0)
    else:
        mc_ecc_fore = np.zeros_like(mc_ecc_both)
        mc_circ_fore = np.zeros_like(mc_circ_both)
        loud = np.zeros((loudest, nreals))

    mc_ecc_back = mc_ecc_both - mc_ecc_fore
    mc_circ_back = mc_circ_both - mc_circ_fore

    # Package and return
    mc_ecc = [mc_ecc_fore, mc_ecc_back, mc_ecc_both]
    mc_circ = [mc_circ_fore, mc_circ_back, mc_circ_both]

    return mc_ecc, mc_circ, sa_ecc, sa_circ, loud


# ============    Utlity    ============


def eccs_func(norm, std, size):
    eccs = utils.log_normal_base_10(1.0/norm, std, size=size)
    eccs = 1.0 / (eccs + 1.0)
    return eccs


def _parse_log_norm_pars(vals, nbins, default=None):
    """
    vals:
        ()   ==> (N,)
        (2,) ==> (N,) log_normal(vals)
        (N,) ==> (N,)

    """
    if (vals is None):
        if default is None:
            return None
        vals = default

    if np.isscalar(vals):
        vals = vals * np.ones(nbins)
    elif (isinstance(vals, tuple) or isinstance(vals, list)) and (len(vals) == 2):
        vals = utils.log_normal_base_10(*vals, size=nbins)
    elif np.shape(vals) != (nbins,):
        err = "`vals` must be scalar, (2,) of scalar, or array (nbins={},) of scalar!".format(nbins)
        raise ValueError(err)

    return vals
