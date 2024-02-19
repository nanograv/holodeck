""" Massive Black Hole Binary (MBHB) accretion models to evolve individual Massive Black Hole (MBH)
masses using the illustris accretion rates.

Authors
-------
Magdalena Siwek

"""

import os
import numpy as np
from scipy.interpolate import RectBivariateSpline
from holodeck import _PATH_DATA, utils
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
                 subpc=True, edd_lim=None, evol_mass=True, **kwargs):
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
        edd_lim : None, or numerical value. This limits the accretion to a factor edd_lim the Eddington limit.

        Other Parameters
        ----------------
        **kwargs : dict

        """
        self.accmod = accmod
        self.f_edd = f_edd
        self.mdot_ext = mdot_ext
        self.eccen = eccen
        self.subpc = subpc
        self.edd_lim = edd_lim
        self.evol_mass = evol_mass

        """ PRE-CALCULATE PREFERENTIAL ACCRETION FUNCTION """
        if self.accmod == 'Siwek22':
            """ interpolate to get lambda at [q,e] """
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
            
            self.swk_acc = lambda_qe_interp_2d()

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
        from holodeck.constants import SPLC, EDDT

        # medd = (4.*np.pi*NWTG*MPRT)/(eps*SPLC*SIGMA_T) * self.mtot
        medd = self.f_edd * (EDDT/(eps * SPLC**2)) * mass
        return(medd)

    def mdot_total(self, evol, bin, step):
        from holodeck.constants import PC

        if self.mdot_ext is not None:
            """ accretion rates have been supplied externally """
            raise NotImplementedError("THIS HASNT BEEN UPDATED TO NEW TIME-BASED INTEGRATION!")
            mdot = self.mdot_ext[:, step-1]

        else:
            """ Get accretion rates as a fraction (f_edd in self._acc) of the
                Eddington limit from current BH masses """
            if bin == None: 
                total_bh_masses = np.sum(evol.mass[:, step, :], axis=-1)
            else:
                total_bh_masses = np.sum(evol.mass[step, :])
            mdot = self.mdot_eddington(total_bh_masses)

        """ Calculate individual accretion rates """
        if self.subpc:
            """ Indices where separation is less than or equal to a parsec """
            if bin == None:
                #we are using old evol method, i.e. calculating mdot for all binaries at a given step
                mdot[evol.sepa[:, step]>PC] = 0.0
            else:
                #new evol method, iterating over individual binaries, so mdot is just a float
                if evol.sepa[step] > PC:
                    mdot = 0.0

        return mdot

    def pref_acc(self, mdot, evol, bin, step):
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
        
        if bin == None:
            m1, m2 = utils.m1m2_ordered(evol.mass[:,step].T[0], evol.mass[:,step].T[1])
        else:
            m1, m2 = utils.m1m2_ordered(*evol.mass[step])


        if self.accmod == 'Siwek22':
            """if evol has an eccentricity distribution,
               we use it, if not, we set each eccentricity to
               the value specified in __init__ """
            if bin == None:
                e_b = evol.eccen[:, step] if (evol.eccen is not None) else None
            else:
                e_b = evol.eccen[step] if (evol.eccen is not None) else None

            lamb_interp = self.swk_acc
            # Need to use RectBivariateSpline.ev to evaluate the interpolation
            # at points, allowing q_b and e_b to be in non-ascending order
            q_b = m2 / m1
            lamb_qe = lamb_interp.ev(q_b, e_b)
            mdot_1 = 1./(np.array(lamb_qe) + 1.) * mdot
            mdot_2 = np.array(lamb_qe)/(np.array(lamb_qe) + 1.) * mdot

        if self.accmod == 'Basic':
            mdot_1 = mdot_2 = 0.5 * mdot
            mdot_arr = np.array([mdot_1, mdot_2]).T

        if self.accmod == 'Proportional':
            # Calculate ratio of accretion rates so that mass ratio stays
            # constant through accretion
            mdot_ratio = m2/(m1 + m2)
            mdot_2 = mdot_ratio * mdot
            mdot_1 = (1. - mdot_ratio) * mdot

        if self.accmod == 'Primary':
            mdot_ratio = 0.
            mdot_2 = mdot_ratio * mdot
            mdot_1 = (1. - mdot_ratio) * mdot

        if self.accmod == 'Secondary':
            mdot_ratio = 1.
            mdot_2 = mdot_ratio * mdot
            mdot_1 = (1. - mdot_ratio) * mdot

        if self.accmod == 'Duffell':
            # Taken from Paul's paper: http://arxiv.org/abs/1911.05506
            def f(q): return 1./(0.1 + 0.9 * q)
            q = m2/m1
            mdot_1 = mdot/(1. + f(q))
            mdot_2 = f(q) * mdot_1

        if self.edd_lim is not None:
            #need to limit the accretion rate to edd_lim times the Eddington limit
            medd_1 = self.edd_lim * (self.mdot_eddington(m1)/self.f_edd)
            mdot_1 = np.minimum(np.array(medd_1), np.array(mdot_1))
            medd_2 = self.edd_lim * (self.mdot_eddington(m2)/self.f_edd)
            mdot_2 = np.minimum(np.array(medd_2), np.array(mdot_2))

        # After calculating the primary and secondary accretion rates,
        # they need to be placed at the correct index into `mdot_arr`, to
        # account for primary/secondary being at 0-th OR 1-st index
        if bin == None:
            inds_m1_primary = evol.mass[:,step].T[0] >= evol.mass[:,step].T[1]
            mdot_arr = np.zeros(np.shape(evol.mass[:, step-1, :]))
            mdot_arr[:, 0][inds_m1_primary] = mdot_1[inds_m1_primary]
            mdot_arr[:, 0][~inds_m1_primary] = mdot_2[~inds_m1_primary]
            inds_m2_primary = evol.mass[:,step-1].T[1] >= evol.mass[:,step-1].T[0]
            mdot_arr[:, 1][inds_m2_primary] = mdot_1[inds_m2_primary]
            mdot_arr[:, 1][~inds_m2_primary] = mdot_2[~inds_m2_primary]
        else:
            if evol.mass[step, 0] >= evol.mass[step, 1]:
                mdot_arr = [mdot_1, mdot_2]
            else:
                mdot_arr = [mdot_2, mdot_1]
        
        # Catch any case where no model is selected.
        if self.accmod is None:
            raise TypeError("'None' value provided for accretion model." +
        
        return np.asarray(mdot_arr)
    
    def ebeq(self, qb):
        import pickle as pkl
        fp_ebeq_pkl = 'cbd_torques/siwek+23/eb_eq_arr.pkl'
        fp_ebeq_pkl = os.path.join(_PATH_DATA, fp_ebeq_pkl)
        fp_ebeq = open(fp_ebeq_pkl, 'rb')
        ebeq_dict = pkl.load(fp_ebeq)

        all_qb = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        all_ebeq = []

        for q in all_qb: 
            all_ebeq.append(ebeq_dict['q=%.2f_sum_grav_acc' %q])
        
        return(np.interp(qb, all_qb, all_ebeq))

