""" MBHB ACCRETION MODELS TO EVOLVE INDIVIDUAL MBH MASSES USING ILLUSTRIS ACCRETION RATES """
import numpy as np
import os
from holodeck import utils, cosmo, log, _PATH_DATA
from scipy import interpolate
from scipy.interpolate import RectBivariateSpline


class Accretion:
    """

    Preferential Accretion prescription
    ----------------

    """
    def __init__(self, accmod = 'Basic', f_edd = 0.01, mdot_ext=None, eccen=0.0, subpc=True, **kwargs):
        """ First sum masses to get total mass of MBHB """
        self.accmod = accmod
        self.f_edd = f_edd
        self.mdot_ext = mdot_ext
        self.eccen = eccen
        self.subpc = subpc

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
        m1 = evol.mass[:, step-1, 0]
        m2 = evol.mass[:, step-1, 1]

        if self.accmod == 'Siwek22':
            """ Calculate the mass ratio"""
            q_b = m2/m1
            """ secondary and primary can swap indices. need to account for that and reverse the mass ratio """
            inds_rev = q_b > 1
            q_b[inds_rev] = 1./q_b[inds_rev]
            #if evol has eccen, then do below, if not, set e_b = 0.
            #e_b = evol.eccen[:, step-1]
            e_b = self.eccen
            """ Now interpolate to get lambda at [q,e] """
            def lambda_qe_interp_2d(fp="data/preferential_accretion/siwek+22/", es=[0.0,0.2,0.4,0.6,0.8]):
                all_lambdas = []
                for e in es:
                    fname = 'preferential_accretion/siwek+22/lambda_e=%.2f.txt' %e
                    fname = os.path.join(_PATH_DATA, fname)
                    lambda_e = np.loadtxt(fname)
                    qs = lambda_e[:,0]
                    lambdas = lambda_e[:,1]
                    all_lambdas.append(lambdas)
                """ True values of q, e """
                x = qs
                y = es
                """ True values of q, e in a meshgrid """
                X, Y = np.meshgrid(x, y)
                Z = all_lambdas

                """ Need to use RectBivariateSpline since interp2d is deprecated
                    in SciPy 1.10 and will be removed in SciPy 1.12.0,
                    causing an issue when merging into the dev branch """
                lamb_qe_interp = RectBivariateSpline(np.array(x), np.array(y), np.array(Z).T, kx = 1, ky = 1)
                return(lamb_qe_interp)

            lamb_interp = lambda_qe_interp_2d()
            """ Need to use RectBivariateSpline.ev to evaluate the interpolation at points,
                allowing q_b and e_b to be in non-ascending order """
            lamb_qe = lamb_interp.ev(q_b, e_b)
            mdot_1 = 1./(np.array(lamb_qe) + 1.) * mdot
            mdot_2 = np.array(lamb_qe)/(np.array(lamb_qe) + 1.) * mdot

            """ After calculating the primary and secondary accretion rates,
                need to place them at the correct index into mdot_arr, to account
                for primary/secondary being at 0-th OR 1-st index """
            mdot_arr = np.zeros(np.shape(evol.mass[:, step-1, :]))
            inds_m1_primary = m1 >= m2 #where first mass is actually primary
            inds_m2_primary = m2 >= m1 #where second mass is actually primary
            mdot_arr[:, 0][inds_m1_primary] = mdot_1[inds_m1_primary]
            mdot_arr[:, 0][~inds_m1_primary] = mdot_2[~inds_m1_primary]
            mdot_arr[:, 1][inds_m2_primary] = mdot_1[inds_m2_primary]
            mdot_arr[:, 1][~inds_m2_primary] = mdot_2[~inds_m2_primary]
            """ mdot_arr is then passed to _take_next_step() function in evolution.py """
            return(mdot_arr)


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
            q = m2/m1
            mdot_1 = mdot/(1.+f(q))
            mdot_2 = f(q)*mdot_1
            mdot_arr = np.array([mdot_1, mdot_2]).T
            return(mdot_arr)
