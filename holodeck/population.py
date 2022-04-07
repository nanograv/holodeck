"""MBH Binary Populations and related tools.

"""

import abc
import inspect
import os

import numpy as np

import holodeck as holo
from holodeck import utils, log, _PATH_DATA, cosmo
from holodeck.constants import PC, MSOL

_DEF_ECCEN_DIST = (1.0, 0.2)
_DEF_ILLUSTRIS_FNAME = "illustris-galaxy-mergers_L75n1820FP_gas-100_dm-100_star-100_bh-000.hdf5"


class _Population_Discrete(abc.ABC):

    def __init__(self, *args, mods=None, check=True, **kwargs):
        self._check_flag = check
        # Initialize the population
        self._init()
        # Apply modifications (using `Modifer` instances), run `_finalize` and `_check()`
        self.modify(mods)
        return

    def _init(self):
        # Initial binary values (i.e. at time of formation)
        self.mass = None    # blackhole masses    (N, 2)
        self.sepa = None    # binary separation a (N,)
        self.scafa = None    # scale factor        (N,)

        self.eccen = None   # eccentricities      (N,) [optional]
        self.weight = None  # weight of each binary as a sample  (N,) [optional]

        self._size = None
        self._sample_volume = None
        return

    @abc.abstractmethod
    def _update_derived(self):
        pass

    @property
    def size(self):
        return self._size

    @property
    def mtmr(self):
        return utils.mtmr_from_m1m2(self.mass)

    @property
    def redz(self):
        return cosmo.a_to_z(self.scafa)

    def modify(self, mods=None):
        # Sanitize
        if mods is None:
            mods = []
        elif not np.iterable(mods):
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
        array_names = ['scafa', 'sepa', 'mass', 'eccen']
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


class Pop_Illustris(_Population_Discrete):

    def __init__(self, fname=None, **kwargs):
        if fname is None:
            fname = _DEF_ILLUSTRIS_FNAME
            fname = os.path.join(_PATH_DATA, fname)

        self._fname = fname
        super().__init__(**kwargs)
        return

    def _init(self):
        super()._init()
        fname = self._fname
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
        self.scafa = data['time']
        return

    def _update_derived(self):
        self._size = self.sepa.size
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
            # pop.scafa,      # resample linearly in scale-factor
            cosmo.a_to_z(pop.scafa),
            np.log10(pop.sepa / PC)
        ]
        reflect = [
            None,
            [None, 0.0],
            # [0.0, 1.0],   # scafa
            [0.0, None],   # redz
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
        # pop.scafa = new_data[2]
        pop.scafa = cosmo.z_to_a(new_data[2])
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
    """Reset the masses of a target population based on a given M-Host relation.
    """

    def __init__(self, mhost, scatter=True):
        """

        Parameters
        ----------
        mhost : class or instance of `holodeck.relations._MHost_Relation`
            The Mbh-MHost scaling relationship with which to reset population masses.
        scatter : bool, optional
            Include random scatter when resetting masses.
            The amount of scatter is specified in the `mhost.SCATTER_DEX` parameter.

        """
        # if `mhost` is a class (not an instance), then instantiate it
        if inspect.isclass(mhost):
            mhost = mhost()

        if not isinstance(mhost, holo.relations._MHost_Relation):
            err = (
                f"`mhost` ({mhost.__class__}) must be an instance"
                f" or subclass of `holodeck.relations._MHost_Relation`!"
            )
            utils.error(err)

        self.mhost = mhost
        self._scatter = scatter
        return

    def modify(self, pop):
        # relation = self.relation
        host = {}
        for requirement in self.mhost.requirements():
            vals = getattr(pop, requirement, None)
            if vals is None:
                err = (
                    f"population modifier requires '{requirement}', "
                    f"but value is not set in population instance (class: {pop.__class__})!"
                )
                utils.error(err)
            if requirement == 'redz':
                vals = vals[:, np.newaxis] # need to duplicate values for proper broadcasting in calculation
            host[requirement] = vals
        scatter = self._scatter
        # Store old version
        pop._mass = pop.mass
        # if `scatter` is `True`, then it is set to the value in `mhost.SCATTER_DEX`
        pop.mass = self.mhost.mbh_from_host(host, scatter)
        return


def eccen_func(norm, std, size):
    eccen = utils.log_normal_base_10(1.0/norm, std, size=size)
    eccen = 1.0 / (eccen + 1.0)
    return eccen
