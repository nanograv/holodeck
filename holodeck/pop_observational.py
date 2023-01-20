"""

To-Do
-----
* Use only `cosmo` (instead of other astropy.cosmology utils e.g. `Planck15`)

"""
# This code is reproduced and modified from the original repository
# https://github.com/morgan-nanez/nanohertz_GWs that is based
# on results from Mingarelli et al. (2017) (https://zenodo.org/badge/latestdoi/90664185)
# Related paper : https://www.nature.com/articles/s41550-017-0299-6

# Computing probabilistic numbers of PTA binaries from MASSIVE and 2MASS, and making realizations of GW skies with
# ILLUSTRIS merger model
# Chiara Mingarelli, mingarelli@mpifr-bonn.mpg.de

from __future__ import division
import numpy as np
from scipy.integrate import quad
from scipy.stats import lognorm
import astropy  # noqa
import astropy.units as u
from astropy.cosmology import Planck15, z_at_value
import os

import holodeck as holo
from holodeck import _PATH_DATA, cosmo, log
from holodeck.constants import MSOL

# physical constants for natural units c = G = 1
c = 2.99792458*(10**8)
G = 6.67428*(10**(-11))
S_MASS = G*(1.98892*10**(30))/(c**3)

# common function shortcuts
log10 = np.log10
pi = np.pi
sqrt = np.sqrt

# _DEF_OBSERVATIONAL_FNAME = "mbhb-pops-continuous_casey-clyde_mingarelli_2021-02-17.npz"
_DEF_OBSERVATIONAL_FNAME = "observational_2mass_galaxy-catalog_extended.npz"


class BP_Observational(holo.population._Population_Discrete):

    FREQ_MIN = 1e-9    # Hz, minimum of PTA band of interest

    def __init__(self, fname=None, *args, **kwargs):
        if fname is None:
            fname = _DEF_OBSERVATIONAL_FNAME
        if not os.path.isfile(fname):
            fname = os.path.join(_PATH_DATA, fname)

        self._fname = fname
        super().__init__(*args, **kwargs)
        return

    def _init(self):
        super()._init()
        fname = self._fname
        data = np.load(fname)
        k_mag = data['k_mag'][:1000]
        log.warning("WARNING: TRUNCATING `k_mag` FOR TESTING!")

        # Construct population
        no_of_bhb, prim_BHmass_min, prim_BHmass_max, binaries = single_realization(k_mag, self.FREQ_MIN)
        pop, edges = continuous_pop(no_of_bhb, *np.log10([prim_BHmass_min, prim_BHmass_max]))
        self._single_realization = binaries
        self._mbhb_pdf = pop
        self._mbhb_edges = edges

        # Convert from grid to 1D arrays
        self._redz, self._mtot, self._mrat = edges
        redz, mtot, mrat = [xx.flatten() for xx in np.meshgrid(*edges, indexing='ij')]
        mtot = (10.0**mtot) * MSOL
        mass = holo.utils.m1m2_from_mtmr(mtot, mrat).T

        # print(f"{pop.shape=}, {pop.size=}, {redz.shape=}, {mtot.shape=}, {mass.shape=}")

        # Store standardized quantities
        self.scafa = cosmo.z_to_a(redz)
        self.sepa = holo.utils.kepler_sepa_from_freq(mtot, self.FREQ_MIN)
        self.mass = mass
        self.weight = pop.flatten()

        return

    def _update_derived(self):
        self._size = self.sepa.size
        return


def pipeline(freq_min=1e-9):
    """Full calculation pipeline.

    freq_min : [Hz], minimum PTA frequency
    """

    # Main Part of Code

    # Choose a galaxy catalog
    fname = "2mass_galaxies.lst"
    fname = os.path.join(_PATH_DATA, fname)
    catalog = np.loadtxt(fname, usecols=(1, 2, 3, 4))

    fname = "added_Mks.lst"
    fname = os.path.join(_PATH_DATA, fname)
    ext_catalog = np.loadtxt(fname, usecols=(1, 2, 3, 4, 5), skiprows=2)

    k_mag = catalog[:, 3]
    k_mag = np.hstack((k_mag, ext_catalog[:, 4]))

    no_of_bhb, prim_BHmass_min, prim_BHmass_max, binaries = single_realization(k_mag, freq_min)
    log_m_bh_min = np.log10(prim_BHmass_min)
    log_m_bh_max = np.log10(prim_BHmass_max)
    bhb_pop = continuous_pop(no_of_bhb, log_m_bh_min, log_m_bh_max)
    return bhb_pop, binaries


def single_realization(k_mag, f_min=1e-9):
    # creating the realizations
    gal_no = k_mag.size
    # array which holds the probablity of each binary being in PTA band and outputs from prob calcs.
    p_i_vec = np.zeros([gal_no])

    z_loop = np.zeros([gal_no])
    T_zLoop = np.zeros([gal_no])
    mergRate_loop = np.zeros([gal_no])
    t2c_loop = np.zeros([gal_no])
    r_inf_loop = np.zeros([gal_no])
    friction_t_loop = np.zeros([gal_no])
    hardening_t_loop = np.zeros([gal_no])

    # initialize mass arrays

    m_bulge = Mk2mStar(k_mag)  # inferred M* mass from k-band luminosity, Cappellari (2013)
    tot_mass = Mbh2Mbulge(m_bulge)  # M-Mbulge McConnell & Ma

    # Look for galaxies which have dynamical SMBH mass measurements, and replace their M-Mbulge total
    # mass with the dynamically measured one.

    # q_choice = np.zeros([gal_no])
    # for yy in range(gal_no):
    #     q_choice[yy] = np.random.choice(np.logspace(-0.6020599913279624, 0, num=5000))  # random q > 0.25 each time

    # Uniform mass-ratios in logspace in [0.25, 1]
    q_choice = np.random.uniform(*np.log10([0.25, 1.0]), gal_no)
    q_choice = 10.0**q_choice

    # NOTE: not used
    # chirp_mass_vec = np.zeros([gal_no])
    # for xx in range(gal_no):
    #     # chirp mass with that q, M_tot from catalogue
    #     chirp_mass_vec[xx] = mchirp_q(q_choice[xx], tot_mass[xx])/S_MASS
    # NOTE: faster
    # chirp_mass_vec = mchirp_q(q_choice, tot_mass) / S_MASS

    # prob of binary being in PTA band
    for zz in range(gal_no):
        p_i_vec[zz], z_loop[zz], T_zLoop[zz], mergRate_loop[zz], t2c_loop[zz], r_inf_loop[zz], \
            friction_t_loop[zz], hardening_t_loop[zz] = i_prob_Illustris(m_bulge[zz], tot_mass[zz], q_choice[zz], f_min)

    # number of stalled binaries
    binaries = dict(prob=p_i_vec, redz=z_loop, mtot=tot_mass, mrat=q_choice)

    # num_zeros = (p_i_vec == 0).sum()
    pta_sources = np.sum(p_i_vec)

    # What is the prob. of a single galaxy being chosen?
    prob_of_each_gal = p_i_vec/pta_sources
    no_of_samples = int(np.round(pta_sources))

    # from "gal_no" choose "no_of_samples" with a probability of "p". The result is the index of the galaxy.
    gal_choice = np.random.choice(gal_no, no_of_samples, replace=False, p=prob_of_each_gal)

    # number of stalled binaries and their indexs
    # num_stalled = (p_i_vec == 0).sum()
    # prob_of_each_gal_stalled = p_i_vec/num_stalled
    # gal_choice_stalled = [gal for gal in range(gal_no) if p_i_vec[gal] == 0]

    # get the primary masses of all binaries and the PTA sources
    prim_mass = tot_mass/(1+q_choice)
    prim_mass_sources = []
    # collect data for all desires galaxies
    for pr in gal_choice:
        prim_mass_sources.append(prim_mass[pr])

    # Compute min and max primary mass among PTA sources
    prim_BHmass_min = min(prim_mass_sources)
    prim_BHmass_max = max(prim_mass_sources)

    return no_of_samples, prim_BHmass_min, prim_BHmass_max, binaries


def quasar_formation_rate(log_mass, z, log_formation_rate_normalization=-3.830,
                          log_formation_rate_power_law_slope=-4.02,
                          log_mass_break_normalization=8.959,
                          log_mass_break_k_1=1.18, log_mass_break_k_2=-6.68,
                          low_mass_slope=.2,
                          high_mass_slope_normalization=2.86,
                          high_mass_slope_k_1=1.80, high_mass_slope_k_2=-1.13,
                          z_ref=2):
    """Compute the differential quasar formation rate density.

    Differential quasar formation rate density computed per unit
    redshift and logarithmic mass, as a function of redshift and
    logarithmic mass.

    Parameters
    ----------
    log_mass : float
        Base 10 logarithm of the mass coordinate.
    z : float
        Redshift coordinate.
    log_formation_rate_normalization : float, optional
        Base 10 logarithm of quasar formation rate normalization. The
        default is -3.830.
    log_formation_rate_power_law_slope : float, optional
        Power law slope of formation rate normalization evolution over
        redshift. The default is -4.02.
    log_mass_break_normalization : float, optional
        Base 10 logarithm of "break mass". The default is 8.959.
    log_mass_break_k_1 : float, optional
        Linear coefficient in break mass quadratic evolution. The default
        is 1.18.
    log_mass_break_k_2 : float, optional
        Quadratic coefficient in break mass quadratic evolution. The
        default is -6.68.
    low_mass_slope : float, optional
        Low mass slope in the double power law mass function. The default
        is .2.
    high_mass_slope_normalization : float, optional
        Local normalization of the high mass slope in the double power law
        mass function. The default is 2.86.
    high_mass_slope_k_1 : float, optional
        Low redshift slope of high mass slope double power law evolution.
        The default is 1.80.
    high_mass_slope_k_2 : float, optional
        High redshift slope of high mass slope double power law evolution.
        The default is -1.13.
    z_ref : float, optional
        Reference redshift (see Hopkins et al. (2007)[[1]_]). The default
        is 2.

    Returns
    -------
    float
        Differential quasar formation rate per unit log mass and redshift.

    Notes
    -----
    Based on the quasar formation rate model in
    Hopkins et al. (2007)[[1]_], including default parameter values.

    References
    ----------
    .. [1] P. F. Hopkins, G. T. Richards, and L. Hernquist, "An
       Observational Determination of the Bolometric Quasar Luminosity
       Function", The Astrophysical Journal 654, 731 (2007).


    """
    # Hopkins et al. (2007) eq. 10
    xi = np.log10((1 + z) / (1 + z_ref))

    # log form of Hopkins et al. (2007) eq. 25
    z_term = np.where(z <= z_ref, 0, xi * log_formation_rate_power_law_slope)
    log_normalization = log_formation_rate_normalization + z_term

    # Hopkins et al. (2007) eq. 19
    high_mass_slope = (
        2 * high_mass_slope_normalization / ((10 ** (xi * high_mass_slope_k_1)) + (10 ** (xi * high_mass_slope_k_2)))
    )

    # Hopkins et al. (2007) eq. 9
    log_mass_break = (log_mass_break_normalization + (log_mass_break_k_1 * xi) + (log_mass_break_k_2 * (xi ** 2)))

    # no sense computing this more than once
    log_mass_ratio = log_mass - log_mass_break

    # log form of denominator in Hopkins et al. (2007) eq. 24
    low_mass_contribution = 10 ** (log_mass_ratio * low_mass_slope)
    high_mass_contribution = 10 ** (log_mass_ratio * high_mass_slope)
    log_mass_distribution = np.log10(low_mass_contribution + high_mass_contribution)

    # convert from rate to differential redshift density (d/dt to d/dz)
    dtdz = (
        Planck15.H0 * (1 + z) * np.sqrt(Planck15.Om0 * ((1 + z) ** 3) + Planck15.Ok0 * ((1 + z) ** 2) + Planck15.Ode0)
    )
    dtdz = 1 / dtdz.to(u.Gyr ** -1).value

    return (10 ** (log_normalization - log_mass_distribution)) * dtdz


def continuous_pop(num_local_binaries, log_m_min_local, log_m_max_local,
                   log_m_min=7, log_m_max=11, z_max=1.5, q_min=.25, mu_log_q=0,
                   std_log_q=.5, shape=(50, 51, 52)):
    """Continuous differential number density of SMBHBs.

    Taken with respect to unit logarithmic mass, redshift, and mass ratio,
    computed over a range of the same parameters.

    Parameters
    ----------
    num_local_binaries : int
        The number of local (redshift ~0) binaries.
    log_m_min_local : float
        Minimum base 10 logarithmic mass of the local population.
    log_m_max_local : float
        Maximum base 10 logarithmic mass of the local population.
    log_m_min : float, optional
        Minimum base 10 logarithmic mass of the considered SMBHB
        population. The default is 7.
    log_m_max : float, optional
        Maximum base 10 logarithmic mass of the considered SMBHB
        population. The default is 11.
    z_max : float, optional
        Maximum redshift of the considered SMBHB population. The default
        is 1.5.
    q_min : float, optional
        Minimum binary mass ratio to consider (where 0 <= q <= 1). The
        default is .25.
    mu_log_q : float, optional
        Mean log q. Used for assumed log-normal distribution of q. The
        default is 0.
    std_log_q : float, optional
        Standard deviation of log q. Used for assumed log-normal
        distribution of q. The default is .5.

    Returns
    -------
    binary_pop : ndarray
        Differential number density of SMBHB sources over log_mass,
        redshift, and mass ratio. Returned as a 3d array with axes
        corresponding to (log mass, redshift, mass ratio).

    """
    # compute the local binary number density
    distance = 225 * u.Mpc
    z_225 = z_at_value(Planck15.angular_diameter_distance, distance, zmax=.25)
    vol = Planck15.comoving_volume(z_225).value
    local_binary_number_density = num_local_binaries / vol

    # renorm quasar formation rate density to get binary density
    local_agn_number_density = quad(quasar_formation_rate, log_m_min_local,
                                    log_m_max_local, args=(0,))[0]  # at z=0
    binary_norm = local_binary_number_density / local_agn_number_density

    # sample the SMBHB population
    z_range = np.linspace(0, z_max, shape[0])
    log_m_range = np.linspace(log_m_min, log_m_max, shape[1])
    q_range = np.linspace(q_min, 1, shape[2])
    edges = [z_range, log_m_range, q_range]

    p_q = lognorm.pdf(q_range, std_log_q, loc=mu_log_q)
    binary_pop = np.array([[quasar_formation_rate(log_m, z)
                            for log_m in log_m_range]
                           for z in z_range])

    binary_pop *= binary_norm
    binary_pop = binary_pop[..., np.newaxis] * p_q[np.newaxis, np.newaxis, :]
    return binary_pop, edges


def i_prob_Illustris(Mstar, Mtot, q, min_freq):
    """Probability that this galaxy contains a binary in the PTA band
    """
    chirpMass = mchirp_q(q, Mtot)/S_MASS  # in solar mass units
    M1 = Mtot/(1+q)
    M2 = M1*q
    mu_min, mu_max = 0.25, 1.0
    gamma = 1.0  # for Hernquist profile, see Dehen 1993

    # Mstar = Mstar*MzMnow(mu, sigma) # scale M* according to Figure 7 of de Lucia and Blaizot 2007
    MstarZ = 0.7*Mstar
    hardening_t, r_inf_here = t_hard(MstarZ, q, gamma, Mtot)
    friction_t = tfric(MstarZ, M2)
    timescale = hardening_t + friction_t  # Gyrs

    # if timescale is longer than a Hubble time, 0 probability
    # also, if timescale > 12.25 Gyrs (z=4), no merging SMBHs
    # also limit of validity for Rodriguez-Gomez + fit in Table 1.
    if timescale > 12.25:
        return 0, 'nan', timescale*1e9, 'nan', 'nan', r_inf_here, friction_t, hardening_t
    else:
        z = z_at_value(Planck15.age, (13.79-timescale) * u.Gyr)  # redshift of progenitor galaxies
        # print "redshift is ", z
        t2c = time_to_c_wMc(chirpMass, min_freq)  # in years
        mergRate = cumulative_merg_ill(mu_min, mu_max, MstarZ, z)  # rate per Gigayear
        Tz = timescale*1e9
        ans = t2c*mergRate/1e9
        return ans, z, Tz, mergRate, t2c, r_inf_here, friction_t, hardening_t


# Mass functions


def mu(m1, m2):
    return S_MASS*(m1*m2)/(m1+m2)  # reduced mass


def M(m1, m2):
    return S_MASS*(m1+m2)  # total mass


def mchirp(m1, m2):
    return ((mu(m1, m2))**(3./5))*((M(m1, m2))**(2./5))  # chirp mass


def mchirp_q(q, Mtot):
    """Chirp mass in terms of q and M_tot. Answer in seconds.
    """
    ans = (q/(1+q)**2)**(3/5)*Mtot*S_MASS
    return ans


def parsec2sec(d):
    return d*3.08568025e16/299792458


# Functions related to galaxy catalogue

def Mk2mStar(mag):
    """Converting from k-band luminosity to M*

    Eq.2 from Ma et al. (2014), valid for early-type galaxies
    """
    Mstar = 10.58-0.44*(mag + 23)
    return 10**(Mstar)


def Mbh2Mbulge(Mbulge):
    """M_BH - M_bulge relationship.

    Bulge mass to black hole mass (note that M_bulge = Mstar; assume these are the same)
    McConnell and Ma (2013) relation below Figure 3
    Includes scatter in the relation, epsilon = 0.34
    Answer in solar masses.
    """
    # MM2013
    exponent = 8.46+1.05*log10(Mbulge/1e11)
    ans_w_scatter = np.random.normal(exponent, 0.34)
    return 10**ans_w_scatter


# For GWs: strain, GW frequency and time to coalescence


def strain(mass, dist, freq):
    """Strain for an equal mass binary

    Eq. 4 from Schutz and Ma,
    mass in solar masses, freq in Hz, distance in Mpc
    I think this is off by a factor of 2**(-1/3)
    """
    ans = 6.9e-15*(mass/1e9)**(5/3)*(10/dist)*(freq/1e-8)**(2/3)
    return ans


def generic_strain(q_mass_ratio, Mtot, dist, freq):
    strain = sqrt(32./5)*mchirp_q(q_mass_ratio, Mtot)**(5/3)*(pi*freq)**(2/3)/parsec2sec(dist*1e6)
    return strain


def generic_strain_wMc(chirp_mass, dist, freq):
    strain = sqrt(32./5)*(chirp_mass*S_MASS)**(5/3)*(pi*freq)**(2/3)/parsec2sec(dist*1e6)
    return strain


def freq_gw(q, Mtot, tc):
    """GW frquency as a function of time to coalescence in years

    Result from integration of standard df/dt for GWs
    """
    ans = mchirp_q(q, Mtot)**(-5/8)/pi*(256/5*tc*31556926)**(-3/8)
    return ans


def freq_gw_wMc(chirp_mass, tc):
    """GW frquency as a function of time to coalescence in years and chirp mass (directly)

    Result from integration of standard df/dt for GWs
    """
    ans = (chirp_mass*S_MASS)**(-5/8)/pi*(256/5*tc*31556926)**(-3/8)
    return ans


def time_to_c(q, Mtot, freq):
    """Time to coalescence of a binary in years
    """
    ans = (pi*freq)**(-8/3)*mchirp_q(q, Mtot)**(-5/3)*5/256
    return (ans/31556926)


def time_to_c_wMc(chirp_mass, freq):
    """Freq. in Hz, input chirp mass in solar masses, answer in years
    """
    ans = (pi*freq)**(-8/3)*(chirp_mass*S_MASS)**(-5/3)*5/256
    return (ans/31556926)


def i_prob(q, Mtot, min_freq, total_T):
    """Input time in years, Mtot in solar masses
    """
    ans = time_to_c(q, Mtot, min_freq)/total_T
    return ans


def i_prob_wMc(chirpMass, min_freq, total_T):
    """Probability that this galaxy contains a binary in the PTA band

    input time in years, Mtot in solar masses
    """
    ans = time_to_c_wMc(chirpMass, min_freq)/total_T
    return ans


# ## Hardening and Dynamical Friction Timescales

# Black hole merger timescales from galaxy merger timescale; Binney and Tremaine 1987
# "Galactic Dynamics"; also Sesana and Khan 2015
# "a" is computed by equating R_eff from Dabringhausen, Hilker & Kroupa (2008) Eq. 4 and


def R_eff(Mstar):
    """Effective radius, Dabringhausen, Hilker & Kroupa (2008) Eq. 4

    Answer in units of parsecs (pc)
    """
    ans = np.maximum(2.95*(Mstar/1e6)**0.596, 34.8*(Mstar/1e6)**0.399)
    return ans


def r0_sol(Mstar, gamma):
    """R0 solution obtained by equating XX with YY (as in Sesana & Khan 2015)

    answer in parsecs
    """
    ans = R_eff(Mstar)/0.75*(2**(1/(3-gamma))-1)
    return ans


def sigmaVel(Mstar):
    """Velocity dispersion.

    from Zahid et al. 2016 Eq 5 and Table 1 fits; assume massive galaxies with Mb > 10.3
    answer in km/s
    """
    logSigmaB = 2.2969
    alpha2 = 0.299
    Mb = 10**(11)  # solar masses
    logAns = logSigmaB + alpha2*log10(Mstar/Mb)
    # print "sigmaVel is ", (10**logAns)
    return 10**logAns


def tfric(Mstar, M2):
    """Dynamical Friction timescale

    Final eq from https://webhome.weizmann.ac.il/home/iair/astrocourse/tutorial8.pdf
    returns timescale in Gyr
    Mbh should be mass of primary
    """
    # assume log(Lambda) = 10
    vc = sqrt(2)*sigmaVel(Mstar)
    # a = semiMaj_a(Mstar)/1e3 # make sure "a" units are kpc
    a = R_eff(Mstar)/1e3
    ans = 2.64e10*(a/2)**2*(vc/250)*(1e6/M2)
    return ans/1e9


def rho_r(Mstar, gamma, r_var):
    """Stellar density as a function of radius.

    gamma for Dehen profiles; Sesana & Khan 2015, Eq. 1
    r_const = r_0 or "a" in Dehen 1993
    r_var = "r" in Dehen 1993
    answer in seconds^-2
    """
    r_const = parsec2sec(r0_sol(Mstar, gamma))   # parsec to seconds
    r_var = parsec2sec(r_var)
    num = (3-gamma)*(Mstar*S_MASS)*r_const
    deno = 4*pi*(r_var)**gamma*(r_var+r_const)**(4-gamma)
    ans = num/deno
    return ans


def r_inf(Mstar, gamma, Mtot):
    """Influence radius, r_inf, from Sesana & Khan 2015

    answer in parsecs
    """
    num = r0_sol(Mstar, gamma)
    deno = (Mstar/(2*Mtot))**(1/(3-gamma))-1   # units of solar masses cancel out
    rinf = num/deno
    return rinf


def a_StarGW(Mstar, q, Mtot, gamma, H):
    """Characteristic Stellar Radius (??)

    Eq. 6, Sesana & Khan 2015. Assume no eccentricity.
    Answer in seconds
    """
    sigmaInf = sigmaVel(Mstar)*1000/c  # km/s converted to m/s then /c for dimensionless units
    r_inf_loc = r_inf(Mstar, gamma, Mtot)
    rho_inf = rho_r(Mstar, gamma, r_inf_loc)  # rinf in pc, rho_inf func converts
    num = 64*sigmaInf*(q*(Mtot*S_MASS)**3/(1+q)**2)
    deno = 5*H*rho_inf
    ans = (num/deno)**(1/5)
    return ans


def t_hard(Mstar, q, gamma, Mtot):
    """Hardening timescale with stars, Eq. 7 Sesana & Khan 2015

    Answer in Gyrs
    """
    # a_val = parsec2sec(r0_sol(Mstar, gamma))
    H = 15
    aStarGW = a_StarGW(Mstar, q, Mtot, gamma, H)
    sigma_inf = sigmaVel(Mstar)*1000/c
    rinf_val = r_inf(Mstar, gamma, Mtot)
    rho_inf = rho_r(Mstar, gamma, rinf_val)
    ans = sigma_inf/(H*rho_inf*aStarGW)
    return ans/31536000/1e9, rinf_val

# ## Parameters and functions for Illustris
# constants for Illustris, Table 1 of Rodriguez-Gomez et al. (2016), assuming z = 0.


M0 = 2e11  # solar masses
A0 = 10**(-2.2287)  # Gyr^-1
alpha0 = 0.2241
alpha1 = -1.1759
delta0 = 0.7668
beta0 = -1.2595
beta1 = 0.0611
gamma = -0.0477
eta = 2.4644
delta0 = 0.7668
delta1 = -0.4695

# For Illustris galaxy-galaxy merger rate
# functions for Illustris, Table 1 of Rodriguez-Gomez et al. (2016), assuming z != 0.


def A_z(z):
    return A0*(1+z)**eta


def alpha(z):
    return alpha0*(1+z)**alpha1


def beta(z):
    return beta0*(1+z)**beta1


def delta(z):
    return delta0*(1+z)**delta1


def MzMnow(mu, sigma):
    """Scale the value of M* to its value at z=0.3.

    Here mu, sigma = 0.75, 0.05
    This is from de Lucia and Blaizot 2007, Figure 7.
    """
    ans = np.random.normal(mu, sigma)
    return ans


def illus_merg(mustar, Mstar, z):
    """Galaxy-galaxy merger rate from Illustris simulation.

    This is dN_mergers/dmu dt (M, mu*), in units of Gyr^-1
    Table 1 of Rodriguez-Gomez et al. (2016).
    """
    exponent = beta(z) + gamma*np.log10(Mstar/1e10)
    rate = A_z(z)*(Mstar/1e10)**alpha(z)*(1+(Mstar/M0)**delta(z))*mustar**exponent
    return rate


def cumulative_merg_ill(mu_min, mu_max, Mstar, z):
    """Cumulative merger probability over a range of mu^*.

    For major mergers, this is 0.25 to 1.0
    """
    ans, err = quad(illus_merg, mu_min, mu_max, args=(Mstar, z))
    return ans
