""" MBHB ACCRETION MODELS TO EVOLVE INDIVIDUAL MBH MASSES USING ILLUSTRIS ACCRETION RATES """
import numpy as np

class Accretion:
    """

    Additional Notes
    ----------------

    """
    def __init__(self, accmod = 'Basic', f_edd = 0.01, mdot_ext=None, **kwargs):
        """ First sum masses to get total mass of MBHB """
        self.accmod = accmod
        self.f_edd = f_edd
        self.mdot_ext = mdot_ext

    def mdot_eddington(self, mass):
        """ Calculate the total accretion rate based on masses and a
        fraction of the Eddington limit.
        """
        from holodeck.constants import SIGMA_T, MPRT, NWTG, MSOL, SPLC, EDDT
        #choose radiative efficiency epsilon = 0.1
        #eps = 0.1
        #medd = (4.*np.pi*NWTG*MPRT)/(eps*SPLC*SIGMA_T) * self.mtot
        eps = 0.1
        medd = self.f_edd * (EDDT/(eps*SPLC**2)) * mass
        return(medd)

    def pref_acc(self, mdot, evol, step):
        """ Choose one of the below models to calculate primary vs secondary accretion rates
            We also supply the instance of the evolution class here in case we need to access eccentricities """
        # if self.accmod == 'Siwek22':
        #

        if self.accmod == 'Basic':
            mdot_1 = mdot_2 = 0.5*mdot
            """ Return an array of accretion rates for each binary in this timestep """
            mdot_arr = np.array([mdot_1, mdot_2]).T
            return(mdot_arr)

        if self.accmod == 'Proportional':
            """ Get primary and secondary masses """
            m1 = self.m1
            m2 = self.m2
            """ Calculate ratio of accretion rates so that mass ratio stays constant through accretion """
            mdot_ratio = m2/(m1+m2)
            mdot_2 = mdot_ratio*mdot
            mdot_1 = (1.-mdot_ratio)*mdot
            """ Return an array of accretion rates for each binary in this timestep """
            mdot_arr = np.array([mdot_1, mdot_2]).T
            return(mdot_arr)

        if self.accmod == 'Primary':
            mdot_ratio = 0.
            mdot_2 = mdot_ratio*mdot
            mdot_1 = (1.-mdot_ratio)*mdot
            """ Return an array of accretion rates for each binary in this timestep """
            mdot_arr = np.array([mdot_1, mdot_2]).T
            return(mdot_arr)

        if self.accmod == 'Secondary':
            mdot_ratio = 1.
            mdot_2 = mdot_ratio*mdot
            mdot_1 = (1.-mdot_ratio)*mdot
            """ Return an array of accretion rates for each binary in this timestep """
            mdot_arr = np.array([mdot_1, mdot_2]).T
            return(mdot_arr)

        if self.accmod == 'Duffell':
            #Taken from Paul's paper: http://arxiv.org/abs/1911.05506
            f = lambda q: 1./(0.1 + 0.9*q)
            m1 = self.m1
            m2 = self.m2
            q = m2/m1
            mdot_1 = mdot/(1.+f(q))
            mdot_2 = f(q)*mdot_1
            mdot_arr = np.array([mdot_1, mdot_2]).T
            return(mdot_arr)
