""" MBHB ACCRETION MODELS TO EVOLVE INDIVIDUAL MBH MASSES USING ILLUSTRIS ACCRETION RATES """
import numpy as np

class Accretion:
    """

    Additional Notes
    ----------------

    """
    def __init__(self, evol, step, f_edd = 0.01):
        """ First sum masses to get total mass of MBHB """
        self.mtot = np.sum(evol.mass[:, step-1, :], axis=1)
        self.m1 = evol.mass[:, step-1, 0]
        self.m2 = evol.mass[:, step-1, 1]
        self.f_edd = f_edd

    def mdot_eddington(self):
        from holodeck.constants import SIGMA_T, MPRT, NWTG, MSOL, SPLC, EDDT
        #choose radiative efficiency epsilon = 0.1
        #eps = 0.1
        #medd = (4.*np.pi*NWTG*MPRT)/(eps*SPLC*SIGMA_T) * self.mtot
        eps = 0.1
        medd = (EDDT/(eps*SPLC**2)) * self.mtot
        return(medd)

    def total_mdot(self):
        """ Calculate the total accretion rate based on masses and a
        fraction of the Eddington limit.
        """
        return(self.f_edd * self.mdot_eddington())

    def basic_accretion(self):
        mdot = self.total_mdot()
        mdot_1 = mdot_2 = 0.5*mdot
        """ Return an array of accretion rates for each binary in this timestep """
        mdot_arr = np.array([mdot_1, mdot_2]).T
        return(mdot_arr)

    def proportional_accretion(self):
        mdot = self.total_mdot()
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

    def primary_accretion(self):
        mdot = self.total_mdot()
        mdot_ratio = 0.
        mdot_2 = mdot_ratio*mdot
        mdot_1 = (1.-mdot_ratio)*mdot
        """ Return an array of accretion rates for each binary in this timestep """
        mdot_arr = np.array([mdot_1, mdot_2]).T
        return(mdot_arr)

    def secondary_accretion(self):
        mdot = self.total_mdot()
        mdot_ratio = 1.
        mdot_2 = mdot_ratio*mdot
        mdot_1 = (1.-mdot_ratio)*mdot
        """ Return an array of accretion rates for each binary in this timestep """
        mdot_arr = np.array([mdot_1, mdot_2]).T
        return(mdot_arr)

    def duffell_accretion(self):
        mdot = self.total_mdot()
        #Taken from Paul's paper: http://arxiv.org/abs/1911.05506
        f = lambda q: 1./(0.1 + 0.9*q)
        m1 = self.m1
        m2 = self.m2
        q = m2/m1
        mdot_1 = mdot/(1.+f(q))
        mdot_2 = f(q)*mdot_1
        mdot_arr = np.array([mdot_1, mdot_2]).T
        return(mdot_arr)
