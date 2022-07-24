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
    """Base class for representing discrete binaries, e.g. from cosmo hydrodynamic simulations.
    """

    def __init__(self, *args, mods=None, check=True, **kwargs):
        """Initialize discrete population.

        Typically the initializer will be overridden, calling `super().__init__()`.
        Runs the `self._init()` method, and the `self.modify()` method.

        Arguments
        ---------
        args : additional arguments
        mods : None or (list of `utils._Modifiers`),
            Population modifiers to apply to this population.
        check : bool,
            Perform diagnostic checks.
        kwargs : dict, additional keyword-arguments

        """
        self._check_flag = check
        # Initialize the population
        self._init()
        # Apply modifications (using `Modifer` instances), run `_finalize` and `_check()`
        self.modify(mods)
        return

    def _init(self):
        """Initialize basic binary parameters.

        This function should typically be overridden in subclasses.

        """
        # Initial binary values (i.e. at time of formation)
        self.mass = None    #: blackhole masses    (N, 2)
        self.sepa = None    #: binary separation `a` (N,)
        self.scafa = None   #: scale factor of the universe (N,)

        self.eccen = None   #: binary eccentricities      (N,) [optional]
        self.weight = None  #: weight of each binary as a sample point  (N,) [optional]

        self._size = None   #: number of binaries
        self._sample_volume = None    #: comoving volume containing the binary population [cm^3]
        return

    @abc.abstractmethod
    def _update_derived(self):
        """Set or reset any derived quantities.
        """
        pass

    @property
    def size(self):
        """Number of binaries in descrete population.

        Returns
        -------
        int
            Number of binaries.

        """
        return self._size

    @property
    def mtmr(self):
        """Total mass and mass-ratio of each binary.

        Returns
        -------
        ndarray (2, N)
            The total-mass (0) and mass-ratio (1) of each binary.

        """
        return utils.mtmr_from_m1m2(self.mass)

    @property
    def redz(self):
        """Redshift at formation of each binary.

        Returns
        -------
        ndarray (N,)
            Redshift.

        """
        return cosmo.a_to_z(self.scafa)

    def modify(self, mods=None):
        """Apply any population modifiers to this population.

        Parameters
        ----------
        mods : None or (list of `Population_Modifer`)
            Population modifiers to apply to this population.

        """
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
        """Method called after all population modifers have been applied, in the `modify()` method.
        """
        pass

    def _check(self):
        """Perform diagnostic/sanity checks on the binary population.
        """
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
    """Discrete population derived from the Illustris cosmological hydrodynamic simulations.

    Takes as input a data file that includes BH and subhalo data for BH and/or galaxy mergers.

    NOTES
    -----
    * Parameters required in input hdf5 file:
        * `box_volume_mpc`:
        * `part_names`:
        * `time`:
        * `SubhaloHalfmassRadType`:
        * `SubhaloMassInRadType`:
        * `SubhaloVelDisp`:
        * `SubhaloBHMass`:

    """

    def __init__(self, fname=None, **kwargs):
        """Initialize a binary population using data in the given filename.

        Parameters
        ----------
        fname : None or str,
            Filename for input data.
            * `None`: default value `_DEF_ILLUSTRIS_FNAME` is used.
        kwargs : dict,
            Additional keyword-arguments passed to `super().__init__`.

        """
        if fname is None:
            fname = _DEF_ILLUSTRIS_FNAME
            fname = os.path.join(_PATH_DATA, fname)

        self._fname = fname             #: Filename for binary data
        super().__init__(**kwargs)
        return

    def _init(self):
        """Set the population parameters using an input simulation file.
        """
        super()._init()
        fname = self._fname
        header, data = utils.load_hdf5(fname)
        self._sample_volume = header['box_volume_mpc'] * (1e6*PC)**3   #: comoving-volume of sim [cm^3]

        # Select the stellar radius
        part_names = header['part_names'].tolist()
        st_idx = part_names.index('star')
        gal_rads = data['SubhaloHalfmassRadType']
        gal_rads = gal_rads[:, st_idx, :]
        # Set initial separation to sum of stellar half-mass radii
        self.sepa = np.sum(gal_rads, axis=-1)       #: Initial binary separation [cm]
        # Get the stellar mass, and take that as bulge mass
        self.mbulge = data['SubhaloMassInRadType'][:, st_idx, :]   #: Stellar mass / stellar-bulge mass [grams]
        self.vdisp = data['SubhaloVelDisp']    #: Velocity dispersion of galaxy [?cm/s?]
        self.mass = data['SubhaloBHMass']      #: BH Mass in subhalo [grams]
        self.scafa = data['time']              #: scale-factor at time of 'merger' event in sim []
        return

    def _update_derived(self):
        """Reset any derived quantities.

        This is called after modifiers are applied, which may change class attributes.

        """
        self._size = self.sepa.size            #: Number of binaries
        return


class Population_Modifier(utils._Modifier):
    """Base class for constructing Modifiers that are applied to `_Discrete_Population` instances.
    """
    pass


class PM_Eccentricity(Population_Modifier):
    """Population Modifier to implement eccentricity in the binary population.
    """

    def __init__(self, eccen_dist=_DEF_ECCEN_DIST):
        """Initialize eccentricity modifier.

        Parameters
        ----------
        eccen_dist : (2,) array_like of float
            Parametrization of the center and width of the desired eccentricity distribution.
            Passed as the 0th and 1th arguments to the `holodeck.utils.eccen_func` function.

        """
        self.eccen_dist = eccen_dist        #: Two parameter specification for eccentricity distribution
        return

    def modify(self, pop):
        """Add eccentricity to the given population.

        Passes the `self.eccen_dist` attribute to the `holodeck.utils.eccen_func` function.

        Parameters
        ----------
        pop : instance of `_Population_Discrete` or subclass,
            Binary population to be modified.

        """
        eccen_dist = self.eccen_dist
        size = pop.size
        # Draw eccentricities from the `eccen_func` defined to be [0.0, 1.0]
        eccen = utils.eccen_func(*eccen_dist, size)
        pop.eccen = eccen
        return


class PM_Resample(Population_Modifier):
    """Population Modifier to resample a population instance to a new number of binaries.

    Uses `kalepy` kernel density estimation (KDE) to resample the original population into a new
    one, changing the total number of binaries by some factor (usually increasing the population).

    Notes
    -----
    **Resampled Parameters**: By default, four or five parameters are resampled:
    1) `mtot`: total mass; 2) `mrat`: mass ratio; 3) `redz`: redshift; and 4) `sepa`: separation;
    and if the `eccen` attrbute of the population is not `None`, then it is also resampled.
    Additional parameters can be specified using the `_DEF_ADDITIONAL_KEYS` attribute which is a
    list of str giving the parameter names, which must then be accessible attributes of the
    population instance being resampled.
    **Sample Volume**: `_Discrete_Population` instances refer to some finite (comoving) volume of
    the Universe, which is given in the population's `_sample_volume` attribute.  When resampling
    is performed, this volume is also multiplied by the same resampling factor.  For example, if
    the population is resampled 10x (i.e. ``resample==10.0``), then the `_sample_volume` attribute
    is also multiplied by 10x.

    """

    # Additional variables to be resampled
    # NOTE: mtot, mrat, redz, sepa, eccen (if not None) are all resampled automatically
    _DEF_ADDITIONAL_KEYS = ['vdisp', 'mbulge']

    def __init__(self, resample=10.0, plot=False, additional_keys=True):
        """Initialize `PM_Resample` instance.

        Parameters
        ----------
        resample : float,
            Factor by which to resample the population.  For example, if ``resample==10.0``, the new
            population will have 10x more elements than the initial population.
        plot : bool,
            If `True`, store old properties and some additional parameters for plotting comparisons
            of the old and new population.
        additional_keys : bool, or list of strings
            Whether or not to resample additional attributes of the population.
            `True`: the parameters in `_DEF_ADDITIONAL_KEYS` are also resampled.
            `False` or `None`: no additional parameters are resampled.
            `list[str]`: the provided additional parameters are resampled, which must be attributes
                of the population instance being resampled.

        """
        self.resample = resample
        self._plot = plot
        self._old_data = None      #: Version of the previous population stored for plotting purposes
        self._new_data = None      #: Version of the updated population stored for plotting purposes

        # Set which additional attributes will be resampled
        if additional_keys is True:
            additional_keys = self._DEF_ADDITIONAL_KEYS
        elif additional_keys in [False, None]:
            additional_keys = []
        self._additional_keys = additional_keys   #: Additional parameter names to be resampled
        return

    def modify(self, pop):
        """Resample the binaries from the given population to achieve a new number of binaries.

        Parameters
        ----------
        pop : instance of `_Population_Discrete` or subclass,
            Binary population to be modified.

        """
        import kalepy as kale

        # ---- Package data for resampling

        # Store basic quantities
        mt, mr = utils.mtmr_from_m1m2(pop.mass)
        labels = ['mtot', 'mrat', 'redz', 'sepa']
        old_data = [
            np.log10(mt / MSOL),
            np.log10(mr),
            # pop.redz,            # linear redshift
            np.log10(pop.redz),    # log redshift
            np.log10(pop.sepa / PC)
        ]
        # set KDE reflection properties
        reflect = [
            None,
            [None, 0.0],
            # [0.0, None],   # linear redz
            None,            # log redz
            None,
        ]

        # Add eccentricity if it's defined (not `None`)
        eccen = pop.eccen
        if eccen is not None:
            labels.append('eccen')
            old_data.append(eccen)
            reflect.append([0.0, 1.0])

        # Add optional variables specified in `_additional_keys` (by default, from `_DEF_ADDITIONAL_KEYS`)
        opt_idx = []
        for ii, opt in enumerate(self._additional_keys):
            # Load value
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

        # construct a `kalepy` Kernel Density Estimator instance
        kde = kale.KDE(old_data, reflect=reflect, bw_rescale=0.25)
        # resample the population data
        new_data = kde.resample(new_size)

        # Convert back to desired quantities
        mt = MSOL * 10**new_data[0]
        mr = 10**new_data[1]
        pop.mass = utils.m1m2_from_mtmr(mt, mr).T
        # stored variable is scale-factor `scafa` (redz is calculated from that), convert from redz
        redz = new_data[2]            # linear red
        redz = 10.0 ** new_data[2]    # log redz
        pop.scafa = cosmo.z_to_a(redz)
        pop.sepa = PC * 10**new_data[3]
        pop.eccen = None if (eccen is None) else new_data[4]

        # store 'additional' parameters
        for opt, idx in zip(self._additional_keys, opt_idx):
            if idx is None:
                continue
            size = new_data[idx].size
            temp = np.zeros((size, 2))
            for kk in range(2):
                temp[:, kk] = np.power(10.0, new_data[idx+kk])
            setattr(pop, opt, temp)

        # increase size of sample volume to account for resampling factor
        pop._sample_volume *= resample

        # store data for plotting
        if self._plot:
            self._labels = labels
            self._old_data = old_data
            self._new_data = new_data

        return

    def plot(self):
        """Plot a comparison of the old and new data, before and after resampling.

        A `kalepy.Corner` plot is generated.

        Returns
        -------
        `mpl.figure.Figure` instance,
            The figure object containing the plot.

        """
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
        """Initialize the modifier.

        Parameters
        ----------
        mhost : class or instance of `holodeck.relations._MHost_Relation`
            The Mbh-MHost scaling relationship with which to reset population masses.
        scatter : bool, optional
            Include random scatter when resetting masses.
            The amount of scatter is specified in the `mhost.SCATTER_DEX` parameter.

        """
        # if `mhost` is a class (not an instance), then instantiate it; make sure its a subclass
        # of `_MHost_Relation`
        mhost = utils._get_subclass_instance(mhost, None, holo.relations._MHost_Relation)
        # store attributes
        self.mhost = mhost         #: Scaling relationship between host and MBH (`holo.relations._MHost_relation`)
        self._scatter = scatter    #: Bool determining whether resampled masses should include statistical scatter
        return

    def modify(self, pop):
        """Reset the BH masses of the given population.

        Parameters
        ----------
        pop : instance of `_Population_Discrete` or subclass,
            Binary population to be modified.

        """
        scatter = self._scatter
        # Store old version
        pop._mass = pop.mass
        # if `scatter` is `True`, then it is set to the value in `mhost.SCATTER_DEX`
        pop.mass = self.mhost.mbh_from_host(pop, scatter)
        return
