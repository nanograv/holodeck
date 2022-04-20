"""
"""
import numpy as np
import astropy as ap
import astropy.constants  # noqa

# ---- Fundamental Constants
NWTG = ap.constants.G.cgs.value
SPLC = ap.constants.c.cgs.value
MELC = ap.constants.m_e.cgs.value
MPRT = ap.constants.m_p.cgs.value
QELC = ap.constants.e.gauss.value           # Fundamental unit of charge (electron charge)
KBOLTZ = ap.constants.k_B.cgs.value         # Boltzmann constant
HPLANCK = ap.constants.h.cgs.value          # Planck constant
SIGMA_SB = ap.constants.sigma_sb.cgs.value  # Stefan-Boltzmann constant
SIGMA_T = ap.constants.sigma_T.cgs.value    # Thomson/Electron -Scattering cross-section

# ---- Typical astronomy units
MSOL = ap.constants.M_sun.cgs.value
LSOL = ap.constants.L_sun.cgs.value
RSOL = ap.constants.R_sun.cgs.value
PC = ap.constants.pc.cgs.value
AU = ap.constants.au.cgs.value
ARCSEC = ap.units.arcsec.cgs.scale              # arcsecond in radians
YR = ap.units.year.to(ap.units.s)
EVOLT = ap.units.eV.to(ap.units.erg)            # Electronvolt in ergs
JY = ap.units.jansky.to(ap.units.g/ap.units.s**2)  # Jansky in [erg/s/cm^2/Hz]

# Derived Constants
# -----------------
SCHW = 2*NWTG/(SPLC*SPLC)                        # Schwarzschild Constant (2*G/c^2)
EDDT = 4.0*np.pi*NWTG*SPLC*MPRT/SIGMA_T          # Eddington Luminosity factor [erg/s/g]

# Electron-Scattering Opacity ($\kappa_{es} = n_e \sigma_T / \rho = \mu_e \sigma_T / m_p$)
#     Where $\mu_e$ is the mean-mass per electron, for a total mass-density $\rho$.
KAPPA_ES = SIGMA_T / MPRT

DAY = 86400.0                                   # Day in seconds
KMPERSEC = 1e5                                  # km/s in cm/s

MYR = 1.0e6*YR
GYR = 1.0e9*YR

KPC = 1.0e3*PC
MPC = 1.0e6*PC
GPC = 1.0e9*PC
