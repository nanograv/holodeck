""" MBHB ACCRETION MODELS TO EVOLVE INDIVIDUAL MBH MASSES USING ILLUSTRIS ACCRETION RATES """
import numpy as np
import os
import random
import holodeck as holo
from scipy import interpolate

class Accretion:
    """

    Additional Notes
    ----------------

    """
    def __init__(self, accmod = 'Basic', f_edd = 0.01, mdot_ext=None, eccen = 0, subpc=True, **kwargs):
        """ First sum masses to get total mass of MBHB """
        self.accmod = accmod
        self.f_edd = f_edd
        self.mdot_ext = mdot_ext
        self.eccen = eccen
        self.subpc = subpc

    def mdot_eddington(self, mass):
        """ Calculate the total accretion rate based on masses and a
        fraction of the Eddington limit.
        UNITS = g/s ?
        """
        from holodeck.constants import SIGMA_T, MPRT, NWTG, MSOL, SPLC, EDDT, PC
        #choose radiative efficiency epsilon = 0.1
        #eps = 0.1
        #medd = (4.*np.pi*NWTG*MPRT)/(eps*SPLC*SIGMA_T) * self.mtot
        eps = 0.1
        medd = self.f_edd * (EDDT/(eps*SPLC**2)) * mass
        return(medd)

    def pref_acc(self, mdot, evol, step):
        """ Choose one of the below models to calculate primary vs secondary accretion rates
            We also supply the instance of the evolution class here in case we need to access eccentricities """
        from holodeck.constants import SIGMA_T, MPRT, NWTG, MSOL, SPLC, EDDT, PC
        m1 = evol.mass[:, step-1, 0]
        m2 = evol.mass[:, step-1, 1]

        primary_inds_m1 = m1 >= m2
        primary_inds_m2 = m2 > m1
        secondary_inds_m1 = m1 <= m2
        secondary_inds_m2 = m2 < m1

        primary = np.zeros(np.shape(m1))
        secondary = np.zeros(np.shape(m2))

        primary[primary_inds_m1] = m1[primary_inds_m1]
        primary[primary_inds_m2] = m2[primary_inds_m2]
        secondary[secondary_inds_m1] = m1[secondary_inds_m1]
        secondary[secondary_inds_m2] = m2[secondary_inds_m2]

        if self.accmod == 'Siwek22':
            q_b = secondary/primary
            q_b_inds = q_b > 1
            q_b[q_b_inds] = 1./q_b[q_b_inds]
            #if evol has eccen, then do below, if not, set e_b = 0.
            #e_b = evol.eccen[:, step-1]
            e_b = self.eccen#random.uniform(0.1, 1.0)
            print("e_b = ", e_b)
            """ Now interpolate to get lambda at [q,e] """
            def lambda_qe_interp_2d(fp="data/preferential_accretion/siwek+22/", es=[0.0,0.2,0.4,0.6,0.8]):
                all_lambdas = []
                for e in es:
                    fname = 'preferential_accretion/siwek+22/lambda_e=%.2f.txt' %e
                    fname = os.path.join(holo._PATH_DATA, fname)
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
                """ Interpolation function in 2D grid, should extrapolate outside domain by default """
                lamb_qe_interp = interpolate.interp2d(x, y, Z, kind='linear')
                return(lamb_qe_interp)

            lamb_interp = lambda_qe_interp_2d()
            lamb_qe = lamb_interp(q_b, e_b)
            mdot_1 = 1./(lamb_qe + 1.) * mdot
            mdot_2 = lamb_qe/(lamb_qe + 1.) * mdot

            mdot_1_arr = np.zeros(np.shape(mdot_1))
            mdot_2_arr = np.zeros(np.shape(mdot_2))

            mdot_1_arr[primary_inds_m1] = mdot_1[primary_inds_m1]
            mdot_1_arr[secondary_inds_m1] = mdot_2[secondary_inds_m1]

            mdot_2_arr[secondary_inds_m2] = mdot_2[secondary_inds_m2]
            mdot_2_arr[primary_inds_m2] = mdot_1[primary_inds_m2]

            """ ONLY PASS CBD ACCRETION RATES TO BINARY IF SEPARATION IS <= PARSEC,
                OTHERWISE ACCRETION RATES ARE ZERO """
            if self.subpc:
                ind_above_pc = evol.sepa[:,step-1] > PC
                mdot_1_arr[ind_above_pc] = 0.0
                mdot_2_arr[ind_above_pc] = 0.0
            """ ONLY PASS CBD ACCRETION RATES TO BINARY IF SEPARATION IS <= PARSEC,
                OTHERWISE ACCRETION RATES ARE ZERO """

            mdot_arr = np.array([mdot_1_arr, mdot_2_arr]).T
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
