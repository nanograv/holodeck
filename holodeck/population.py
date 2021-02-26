"""
"""

import abc

import numpy as np

from holodeck import utils, log
from holodeck.constants import PC, MSOL

_DEF_ECCEN_DIST = (1.0, 0.2)


class _Binary_Population(abc.ABC):

    def __init__(self, fname, *args, mods=None, check=True, **kwargs):
        self._fname = fname
        self._check_flag = check

        # Initial binary values (i.e. at time of formation)
        self.time = None    # scale factor        (N,)
        self.sepa = None    # binary separation a (N,)
        self.mass = None    # blackhole masses    (N, 2)
        self.eccen = None    # eccentricities      (N,) [optional]

        self._size = None
        self._sample_volume = None

        # Initialize the population
        self._init_from_file(fname)
        # Apply modifications (using `Modifer` instances), run `_finalize` and `_check()`
        self.modify(mods)
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
        updated = False
        for mod in mods:
            if mod is not None:
                mod(self)
                self._update_derived()
                updated = True

        if not updated:
            self._update_derived()

        self._finalize()
        if self._check_flag:
            self._check()
        return

    def _finalize(self):
        pass

    def _check(self):
        ErrorType = ValueError
        msg = "{}._check() Failed!  ".format(self.__class__.__name__)
        array_names = ['time', 'sepa', 'mass', 'eccen']
        two_dim = ['mass']
        allow_none = ['eccen']

        size = self.size
        if size is None:
            err = msg + " `self.size` is 'None'!"
            raise ErrorType(err)

        for name in array_names:
            vals = getattr(self, name)
            shape = np.shape(vals)
            msg = None if (vals is None) else shape
            log.debug(f"{name:>10s} :: {msg}")
            if vals is None:
                if name in allow_none:
                    continue
                err = msg + "`{}` is 'None'!".format(name)
                raise ErrorType(err)

            bad_shape = False
            if (len(shape) == 0) or (size != shape[0]):
                bad_shape = True

            if name in two_dim:
                if (len(shape) != 2) or (shape[1] != 2):
                    bad_shape = True
            elif len(shape) != 1:
                bad_shape = True

            if bad_shape:
                err = msg + "`{}` has invalid shape {} (size: {})!".format(name, shape, size)
                raise ErrorType(err)

            bads = ~np.isfinite(vals)
            if np.any(bads):
                bads = np.where(bads)
                print("bad entries: ", bads)
                for nn in array_names:
                    print("{}: {}".format(nn, getattr(self, nn)[bads]))
                err = msg + "`{}` has {} non-finite values!".format(name, len(bads[0]))
                raise ErrorType(err)

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


class BP_Continuous(_Binary_Population):

    def _init_from_file(self, fname):
        data = np.load(fname)
        mt = data['mtot'] * MSOL
        mr = data['mrat']
        sc = utils.z_to_a(data['redz'])
        ww = data['pops'][..., 0]
        self._mtot = mt
        self._mrat = mr
        self._redz = data['redz']

        mt, mr, sc = [xx.flatten() for xx in np.meshgrid(mt, mr, sc, indexing='ij')]
        self.mtot = mt
        self.mrat = mr
        self.time = sc
        self.weight = ww.flatten()
        self.sepa = 1e5 * PC * np.ones_like(mt)
        self.mass = utils.m1m2_from_mtmr(self.mtot, self.mrat).T

        return

    def _update_derived(self):
        self._size = self.mtot.size
        return


class Population_Modifier(utils._Modifier):
    pass


class PM_Eccentricity(Population_Modifier):

    def __init__(self, eccen_dist=_DEF_ECCEN_DIST):
        self.eccen_dist = eccen_dist
        return

    def modify(self, pop):
        eccen_dist = self.eccen_dist
        size = pop.size
        eccen = eccen_func(*eccen_dist, size)
        pop.eccen = eccen
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

        eccen = pop.eccen
        if eccen is not None:
            labels.append('eccen')
            old_data.append(eccen)
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

        kde = kale.KDE(old_data, reflect=reflect, bw_rescale=0.25)
        new_data = kde.resample(new_size)

        mt = MSOL * 10**new_data[0]
        mr = 10**new_data[1]

        pop.mass = utils.m1m2_from_mtmr(mt, mr).T
        pop.time = new_data[2]
        pop.sepa = PC * 10**new_data[3]
        pop.eccen = None if (eccen is None) else new_data[4]

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


def eccen_func(norm, std, size):
    eccen = utils.log_normal_base_10(1.0/norm, std, size=size)
    eccen = 1.0 / (eccen + 1.0)
    return eccen
