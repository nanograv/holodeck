"""Numerical Constants

All constants are used in CGS units, as raw floats.  Most of the holodeck package works in CGS units
whenever possible.  Constants and units should only be added when they are frequently used
(i.e. in multiple files/submodules).

Notes
-----
* [cm] = centimeter
* [g] = gram
* [s] = second
* [erg] = cm^2 * g / s^2
* [Jy] = jansky = [erg/s/cm^2/Hz]
* [fr] franklin = statcoulomb = electro-static unit [esu]
* [K] Kelvin

"""
import numpy as np
import astropy as ap
import astropy.constants  # noqa

# ---- Fundamental Constants
NWTG = ap.constants.G.cgs.value             #: Newton's Gravitational Constant [cm^3/g/s^2]
SPLC = ap.constants.c.cgs.value             #: Speed of light [cm/s]
MELC = ap.constants.m_e.cgs.value           #: Electron Mass [g]
MPRT = ap.constants.m_p.cgs.value           #: Proton Mass [g]
QELC = ap.constants.e.gauss.value           #: Fundamental unit of charge (electron charge) [fr]
KBOLTZ = ap.constants.k_B.cgs.value         #: Boltzmann constant [erg/K]
HPLANCK = ap.constants.h.cgs.value          #: Planck constant [erg/s]
SIGMA_SB = ap.constants.sigma_sb.cgs.value  #: Stefan-Boltzmann constant [erg/cm^2/s/K^4]
SIGMA_T = ap.constants.sigma_T.cgs.value    #: Thomson/Electron -Scattering cross-section [cm^2]

# ---- Typical astronomy units
MSOL = ap.constants.M_sun.cgs.value                                #: Solar Mass [g]
LSOL = ap.constants.L_sun.cgs.value                                #: Solar Luminosity [erg/s]
RSOL = ap.constants.R_sun.cgs.value                                #: Solar Radius [cm]
PC = ap.constants.pc.cgs.value                                     #: Parsec [cm]
AU = ap.constants.au.cgs.value                                     #: Astronomical Unit [cm]
ARCSEC = ap.units.arcsec.cgs.scale                                 #: arcsecond in radians []
YR = ap.units.year.to(ap.units.s)                                  #: year [s]
EVOLT = ap.units.eV.to(ap.units.erg)                               #: Electronvolt in ergs
JY = ap.units.jansky.to(ap.units.g/ap.units.s**2)                  #: Jansky [erg/s/cm^2/Hz]
KMPERSEC = (ap.units.km / ap.units.s).to(ap.units.cm/ap.units.s)   #: km/s [cm/s]

# ---- Derived Constants
SCHW = 2*NWTG/(SPLC*SPLC)                        #: Schwarzschild Constant (2*G/c^2) [cm]
EDDT = 4.0*np.pi*NWTG*SPLC*MPRT/SIGMA_T          #: Eddington Luminosity prefactor factor [erg/s/g]

# Electron-Scattering Opacity ($\kappa_{es} = n_e \sigma_T / \rho = \mu_e \sigma_T / m_p$)
#     Where $\mu_e$ is the mean-mass per electron, for a total mass-density $\rho$.
# KAPPA_ES = SIGMA_T / MPRT                       #: Electron scattering opacity [cm^2/g]

DAY = 86400.0                                   #: Day [s]
MYR = 1.0e6*YR                                  #: Mega-year [s]
GYR = 1.0e9*YR                                  #: Giga-year [s]
KPC = 1.0e3*PC                                  #: Kilo-parsec [cm]
MPC = 1.0e6*PC                                  #: Mega-parsec [cm]
GPC = 1.0e9*PC                                  #: Giga-parsec [cm]
