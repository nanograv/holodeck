""" Massive Black Hole Binary (MBHB) accretion models to evolve individual Massive Black Hole (MBH)
masses using the illustris accretion rates.

Authors
-------
Magdalena Siwek

"""

import os
import numpy as np
from scipy.interpolate import RectBivariateSpline
from holodeck import _PATH_DATA
from holodeck.constants import SPLC, EDDT


class Accretion:
    """ Preferential Accretion prescription

    Attributes
    ----------
    accmod : {'Basic', 'Proportional', 'Primary', 'Secondary', 'Siwek22', 'Duffell'}, optional
    f_edd : double, optional
    mdot_ext : Any, optional
    eccen : float, optional
    subpc : boolean, optional

    Methods
    -------
    :meth:`mdot_eddington(mass)`
        Calculate the total accretion rate based on masses and a fraction of
        the Eddington limit.
    :meth:`pref_acc(mdot, evol, step)`
        Contains a variety of accretion models to choose from to calculate
        primary vs secondary accretion rates.

    """

    def __init__(self, accmod='Basic', f_edd=0.01, mdot_ext=None, eccen=0.0,
                 subpc=True, **kwargs):
        """ Initializes the Accretion class.

        Parameters
        ----------
        accmod : {'Basic', 'Proportional', 'Primary', 'Secondary', 'Siwek22', 'Duffell'}, optional
            Name of desired accretion model. Defaults to `Basic`.
        f_edd : double, optional

        mdot_ext : Any, optional

        eccen : float, optional
            Eccentricity value.
        subpc : boolean, optional

        Other Parameters
        ----------------
        **kwargs : dict

        """
        self.accmod = accmod
        self.f_edd = f_edd
        self.mdot_ext = mdot_ext
        self.eccen = eccen
        self.subpc = subpc

    def mdot_eddington(self, mass, eps=0.1):
        """Calculate the total accretion rate based on masses and a fraction of the Eddington limit.

        Parameters
        ----------
        mass : float
        eps : float, optional
            Radiative efficiency epsilon. Defaults to 0.1.

        Returns
        -------
        medd : float

        See Also
        --------
        holodeck.constants : constants used for calculation of the accretion rate.

        Notes
        -----
        The limiting Eddington accretion rate is defined as:
        .. math:: `\dot{M}_{\mathsf{Edd}} = \frac{4 \pi GM m_p}{\epsilon c \sigma_{\mathsf{T}}`

        Examples
        --------
        >>> acc = Accretion()
        >>> mass =
        >>> print(acc.mdot_eddington(mass))

        """
        # medd = (4.*np.pi*NWTG*MPRT)/(eps*SPLC*SIGMA_T) * self.mtot
        medd = self.f_edd * (EDDT/(eps * SPLC**2)) * mass
        return(medd)

    def mdot_total(self, evol, step):
        if self.mdot_ext is not None:
            """ accretion rates have been supplied externally """
            mdot = self.mdot_ext[:,step-1]
        else:
            """ Get accretion rates as a fraction (f_edd in self._acc) of the
                Eddington limit from current BH masses """
            total_bh_masses = np.sum(evol.mass[:, step-1, :], axis=1)
            mdot = self.mdot_eddington(total_bh_masses)

        """ Calculate individual accretion rates """
        if self.subpc:
            """ Indices where separation is less than or equal to a parsec """
            ind_sepa = evol.sepa[:, step] <= PC
        else:
            """ Indices where separation is less than or equal to 100 kilo-parsec """
            ind_sepa = evol.sepa[:, step] <= 10**5 * PC

        """ Set total accretion rates to 0 when separation is larger than 1pc or 10kpc,
            depending on subpc switch applied to accretion instance """
        mdot[~ind_sepa] = 0
        return(mdot)

    def pref_acc(self, mdot, evol, step):
        """
        Contains a variety of accretion models to choose from to calculate primary vs secondary
        accretion rates.

        The accretion models are as follows:
        * Basic
        * Primary
        * Secondary
        * Proportional
        * Siwek22
        * Duffell

        Parameters
        ----------
        mdot :
            Gas inflow rate in solar masses. Units of [M/year]
        evol :
            evolution class instance which contains the eccentricities of the current evolution
        step : int
            current timestep

        Returns
        -------
        mdot_arr : ndarray
            Array of accretion rates for each binary in the timestep.

        See Also
        --------
        :meth:`Evolution._take_next_step()` : Relationship

        Notes
        -----
        The instance of the evolution class must also be suppled in case eccentricities need to be
        accessed.

        """
        m1 = evol.mass[:, step - 1, 0]
        m2 = evol.mass[:, step - 1, 1]

        if self.accmod == 'Siwek22':
            # Calculate the mass ratio
            q_b = m2 / m1
            # secondary and primary may swap indices
            # need to account for that and reverse the mass ratio
            inds_rev = q_b > 1
            q_b[inds_rev] = 1./q_b[inds_rev]
            """if evol has an eccentricity distribution,
               we use it, if not, we set each eccentricity to
               the value specified in __init__ """
            if evol.eccen is not None:
                e_b = evol.eccen[:, step-1]
            else:
                e_b = self.eccen
            """ Now interpolate to get lambda at [q,e] """
            def lambda_qe_interp_2d(fp="data/preferential_accretion/siwek+22/", es=[0.0,0.2,0.4,0.6,0.8]):
                all_lambdas = []
                for e in es:
                    fname = 'preferential_accretion/siwek+22/lambda_e=%.2f.txt' % e
                    fname = os.path.join(_PATH_DATA, fname)
                    lambda_e = np.loadtxt(fname)
                    qs = lambda_e[:, 0]
                    lambdas = lambda_e[:, 1]
                    all_lambdas.append(lambdas)
                # True values of q, e
                x = qs
                y = es
                # Populate the true values of q, e in a meshgrid
                X, Y = np.meshgrid(x, y)
                Z = all_lambdas

                return(RectBivariateSpline(np.array(x), np.array(y),
                                           np.array(Z).T, kx=1, ky=1))

            lamb_interp = lambda_qe_interp_2d()
            # Need to use RectBivariateSpline.ev to evaluate the interpolation
            # at points, allowing q_b and e_b to be in non-ascending order
            lamb_qe = lamb_interp.ev(q_b, e_b)
            mdot_1 = 1./(np.array(lamb_qe) + 1.) * mdot
            mdot_2 = np.array(lamb_qe)/(np.array(lamb_qe) + 1.) * mdot

            # After calculating the primary and secondary accretion rates,
            # they need to be placed at the correct index into `mdot_arr`, to
            # account for primary/secondary being at 0-th OR 1-st index
            mdot_arr = np.zeros(np.shape(evol.mass[:, step-1, :]))
            inds_m1_primary = m1 >= m2  # where first mass is actually primary
            inds_m2_primary = m2 >= m1  # where second mass is actually primary
            mdot_arr[:, 0][inds_m1_primary] = mdot_1[inds_m1_primary]
            mdot_arr[:, 0][~inds_m1_primary] = mdot_2[~inds_m1_primary]
            mdot_arr[:, 1][inds_m2_primary] = mdot_1[inds_m2_primary]
            mdot_arr[:, 1][~inds_m2_primary] = mdot_2[~inds_m2_primary]
            # `mdot_arr` is then passed to `_take_next_step()` in evolution.py
            return(mdot_arr)

        if self.accmod == 'Basic':
            mdot_1 = mdot_2 = 0.5 * mdot
            mdot_arr = np.array([mdot_1, mdot_2]).T
            return(mdot_arr)

        if self.accmod == 'Proportional':
            """ Get primary and secondary masses """
            m1 = self.m1
            m2 = self.m2
            # Calculate ratio of accretion rates so that mass ratio stays
            # constant through accretion
            mdot_ratio = m2/(m1 + m2)
            mdot_2 = mdot_ratio * mdot
            mdot_1 = (1. - mdot_ratio) * mdot
            mdot_arr = np.array([mdot_1, mdot_2]).T
            return(mdot_arr)

        if self.accmod == 'Primary':
            mdot_ratio = 0.
            mdot_2 = mdot_ratio * mdot
            mdot_1 = (1. - mdot_ratio) * mdot
            mdot_arr = np.array([mdot_1, mdot_2]).T
            return(mdot_arr)

        if self.accmod == 'Secondary':
            mdot_ratio = 1.
            mdot_2 = mdot_ratio * mdot
            mdot_1 = (1. - mdot_ratio) * mdot
            mdot_arr = np.array([mdot_1, mdot_2]).T
            return(mdot_arr)

        if self.accmod == 'Duffell':
            # Taken from Paul's paper: http://arxiv.org/abs/1911.05506
            def f(q): return 1./(0.1 + 0.9 * q)
            q = m2/m1
            mdot_1 = mdot/(1. + f(q))
            mdot_2 = f(q) * mdot_1
            mdot_arr = np.array([mdot_1, mdot_2]).T
            return(mdot_arr)

        # Catch any weirdness if no model is selected.
        if self.accmod is None:
            raise TypeError("'None' value provided for accretion model." +
                            "An accretion model is required.")
