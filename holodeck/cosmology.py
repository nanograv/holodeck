"""Cosmology related code.

[WMAP9] = Hinshaw et al. 2013 (1212.5226)
    Nine-year Wilkinson Microwave Anisotropy Probe (WMAP) Observations: Cosmological Parameter Results
    https://ui.adsabs.harvard.edu/abs/2013ApJS..208...19H/abstract

"""

import astropy as ap
import astropy.cosmology  # noqa
import scipy as sp
import numpy as np


class Cosmology(ap.cosmology.FlatLambdaCDM):
    """Class for calculating (and inverting) cosmological distance measures.

    Based on the `Astropy.cosmology` methods and classes.

    The base class provides methods to convert from redshift to desired cosmological parameters,
    e.g. `luminosity_distance(z)`.  These methods are used to construct a numerical grid of values
    as a function of redshift.  To calculate the redshift (and other cosmological parameters) when
    given a derived quantitiy (i.e. to come from d_L to redshift), the numerical grid can then be
    inverted and interpolated to find the corresponding redshift.

    API Methods
    -----------

    """

    # These are WMAP9 parameters, see [WMAP9], Table 3, WMAP+BAO+H0
    Omega0 = 0.2880
    OmegaBaryon = 0.0472
    HubbleParam = 0.6933
    _H0_sub = HubbleParam * 100.0  # NOTE: this *cannot* be `H0` or `_H0` --- conflicts with astropy internals
    SPLC = 29979245800.0

    # z=0.0 is added automatically
    _Z_GRID = [1e4, 100.0, 10.0, 4.0, 2.0, 1.0, 0.5, 0.1, 0.01]
    _INTERP_POINTS = 20

    def __init__(self):
        # Initialize parent class
        super().__init__(H0=self._H0_sub, Om0=self.Omega0, Ob0=self.OmegaBaryon)

        # Create grids for interpolations
        # -------------------------------
        #    Create a grid in redshift
        zgrid = self._init_interp_grid(self._Z_GRID, self._INTERP_POINTS)
        self._grid_z = zgrid
        self._sort_z = np.argsort(zgrid)
        self._grid_a = self.z_to_a(zgrid)
        self._sort_a = np.argsort(self._grid_a)
        # Calculate corresponding values in desired parameters
        #    Ages in seconds
        self._grid_age = self.age(zgrid).cgs.value
        self._sort_age = np.argsort(self._grid_age)
        self._grid_lbk = self.lookback_time(zgrid).cgs.value
        self._sort_lbk = np.argsort(self._grid_lbk)
        #    Comoving distances in centimeters
        self._grid_dcom = self.comoving_distance(zgrid).cgs.value
        self._sort_dcom = np.argsort(self._grid_dcom)
        #    Comoving distances in centimeters
        self._grid_dlum = self.luminosity_distance(zgrid).cgs.value
        self._sort_dlum = np.argsort(self._grid_dlum)
        return

    def __str__(self):
        rstr = "cosmopy.cosmology.Cosmology :: H0 = {}, Om0 = {}, Ob0 = {}".format(
            self.H0, self.Omega0, self.OmegaBaryon)
        return rstr

    @staticmethod
    def _init_interp_grid(z_pnts, num_pnts):
        """Create a grid in redshift for interpolation.

        Arguments
        ---------
        z_pnts : (N,) array_like of scalar
            Monotonically decreasing redshift values greater than zero.  Inbetween each pair of
            values, `num_pnts` points are evenly added in log-space.  Also, `num_pnts` are added
            from the minimum given value down to redshift `0.0`.
        num_pnts : int
            Number of grid points to insert between each pair of values in `z_pnts`.

        Returns
        -------
        zgrid : (M,) array of scalar
            The total length of the final array will be ``N * num_pnts + 1``; that is, `num_pnts`
            for each value given in `z_pnts`, plus redshift `0.0` at the end.

        """
        # Make sure grid is monotonically decreasing
        if not np.all(np.diff(z_pnts) < 0.0):
            err_str = "Non-monotonic z_grid = {}".format(z_pnts)
            raise ValueError(err_str)

        z0 = z_pnts[0]
        zgrid = None
        # Create a log-spacing of points between each pair of values in the given points
        for z1 in z_pnts[1:]:
            temp = np.logspace(*np.log10([z0, z1]), num=num_pnts, endpoint=False)
            if zgrid is None:
                zgrid = temp
            else:
                zgrid = np.append(zgrid, temp)
            z0 = z1

        # Linearly space from the last point up to `0.0` (cant reach `0.0` with log-spacing)
        zgrid = np.append(zgrid, np.linspace(z0, 0.0, num=num_pnts))
        return zgrid

    @staticmethod
    def a_to_z(sf):
        """Convert from scale-factor to redshift.
        """
        sf = np.asarray(sf)
        # NOTE: this does not check for `nan`
        if np.any((sf > 1.0) | (sf < 0.0)):
            raise ValueError("Scale-factor must be [0.0, 1.0]")

        return (1.0/sf) - 1.0

    @staticmethod
    def z_to_a(redz):
        """Convert from redshift to scale-factor.
        """
        redz = np.asarray(redz)
        # NOTE: this does not check for `nan`
        if np.any(redz < 0.0):
            raise ValueError("Redshift must be [0.0, +inf)")

        return 1.0/(redz + 1.0)

    @staticmethod
    def _interp(vals, xx, yy, inds=None):
        if inds is None:
            inds = np.argsort(xx)
        # PchipInterpolate guarantees monotonicity with higher order
        arrs = sp.interpolate.PchipInterpolator(xx[inds], yy[inds], extrapolate=False)(vals)
        return arrs

    def tage_to_z(self, age):
        """Convert from age of the universe [seconds] to redshift.
        """
        zz = self._interp(age, self._grid_age, self._grid_z, self._sort_age)
        return zz

    def tage_to_a(self, age):
        """Convert from age of the universe [seconds] to redshift.
        """
        aa = self._interp(age, self._grid_age, self._grid_a, self._sort_age)
        return aa

    def z_to_tage(self, redz):
        """Convert from age of the universe [seconds] to redshift.
        """
        tage = self._interp(redz, self._grid_z, self._grid_age, self._sort_z)
        return tage

    def a_to_tage(self, sca):
        """Convert from age of the universe [seconds] to redshift.
        """
        tage = self._interp(sca, self._grid_a, self._grid_age, self._sort_a)
        return tage

    def tlbk_to_z(self, lbk):
        """Convert from lookback time [seconds] to redshift.
        """
        zz = self._interp(lbk, self._grid_lbk, self._grid_z, self._sort_lbk)
        return zz

    def z_to_tlbk(self, redz):
        """Convert from redshift to lookback time [seconds].
        """
        zz = self._interp(redz, self._grid_z, self._grid_lbk, self._sort_z)
        return zz

    def dcom_to_z(self, dc):
        """Convert from comoving-distance [cm] to redshift.
        """
        zz = self._interp(dc, self._grid_dcom, self._grid_z, self._sort_dcom)
        return zz

    def dlum_to_z(self, dl):
        """Convert from luminosity-distance [cm] to redshift.
        """
        zz = self._interp(dl, self._grid_dlum, self._grid_z, self._sort_dlum)
        return zz

    def z_to_dcom(self, zz):
        """Convert from comoving-distance [cm] to redshift.
        """
        dc = self._interp(zz, self._grid_z, self._grid_dcom, self._sort_z)
        return dc

    def z_to_dlum(self, zz):
        """Convert from luminosity-distance [cm] to redshift.
        """
        dl = self._interp(zz, self._grid_z, self._grid_dlum, self._sort_z)
        return dl

    def get_grid(self):
        """Return an array of the grid of interpolation points for each cosmological parameter.

        Returns
        -------
        grid : (N,M,) of scalar
            Grid of `N` values for each of `M` cosmological parameters.
            For example, the redshift values are `grid[:, 0]`.
        names : (M,) str
            The names of each cosmological parameter in the grid.
        units : (M,) str
            The units used for each cosmological parameter in the grid.

        """
        _var_keys = [
            "_grid_z", "_grid_a", "_grid_dcom", "_grid_dlum", "_grid_lbk", "_grid_age"]
        _var_names = ["z", "a", "dc", "dl", "tl", "ta"]
        _var_types = [None, None, ['cm', 'Mpc'], ['cm', 'Mpc'], ['s', 'Gyr'], ['s', 'Gyr']]

        ngrid = self._grid_z.size
        npars = len(_var_keys)
        grid = np.zeros((ngrid, npars))
        names = []
        units = []
        for ii, (vk, un) in enumerate(zip(_var_keys, _var_types)):
            vals = getattr(self, vk)
            nam = _var_names[ii]
            # Convert units if desired
            if un is not None:
                vals = ap.units.Quantity(vals, unit=un[0]).to(un[1]).value
                # nam += '[' + un[1] + ']'
                units.append(un[1])
            else:
                units.append('-')
            grid[:, ii] = vals
            names.append(nam)

        return grid, names, units

    def dVcdz(self, zz, cgs=True):
        """Differential comoving volume of the universe.

        From Hogg1999 Eq. 28
        """
        # This is 4*pi*D_h
        efac = self.efunc(zz)
        if cgs:
            retval = (4.0*np.pi*self.SPLC/self.H(0.0).cgs.value)
            com_dist = self.comoving_distance(zz).cgs.value
        else:
            retval = (4.0*np.pi*ap.constants.c/self.H(0.0)).decompose()
            com_dist = self.comoving_distance(zz)

        retval = retval * np.square(com_dist) / efac
        # Using `astropy` this gives a `m * Mpc^2` object for some reason; simplify
        if not cgs:
            retval = retval.to('Mpc3')
        return retval

    def dtdz(self, zz):
        """Differential lookback time of the Universe.

        From Hogg1999 Eq. 30

        Returns
        -------
        retval : array_like of scalar, units of [sec]
            dt/dz at the given redshifts

        """
        efac = self.efunc(zz)
        time_hub = self.hubble_time.to('s').value

        retval = time_hub / (1.0 + zz) / efac
        return retval
