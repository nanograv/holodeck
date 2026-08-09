"""Numerical Constants

All constants are used in CGS units, as raw floats.  Most of the holodeck package works in CGS units
whenever possible.  Constants and units should only be added when they are frequently used
(i.e. in multiple files/submodules).

DEVELOPER NOTE:
To preserve fast module import times across holodeck, do NOT import `astropy` or `numpy` 
in this file to calculate constants at module initialization. If you need to add a new constant:

  1. Open a temporary terminal or notebook.
  2. Run your astropy conversion (e.g., `import astropy as ap; print(ap.constants.m_e.cgs.value)`).
  3. Copy the raw float literal value and paste it directly below.

Commented section at the bottom shows how constants were calculated from astropy.constants (ap.constants)
for future reference in case the values are updated.  

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

# ---- Fundamental Constants
NWTG = 6.674299999999999e-08       #: Newton's Gravitational Constant [cm^3/g/s^2]
SPLC = 29979245800.0               #: Speed of light [cm/s]
MELC = 9.1093837015e-28            #: Electron Mass [g]
MPRT = 1.67262192369e-24           #: Proton Mass [g]
QELC = 4.803204712570263e-10       #: Fundamental unit of charge (electron charge) [fr]
KBOLTZ = 1.380649e-16              #: Boltzmann constant [erg/K]
HPLANCK = 6.62607015e-27           #: Planck constant [erg/s]
SIGMA_SB = 5.6703744191844314e-05  #: Stefan-Boltzmann constant [erg/s/cm^2/K^4]


# ---- Derived Constants
# Schwarzschild constant calculated as
# SCHW = 2 * NWTG / (SPLC * SPLC)    #: Schwarzschild Constant (2*G/c^2) [cm]
SCHW = 1.4852320538237328e-28      #: Schwarzschild Constant (2*G/c^2) [cm]
# Thomson scattering cross section calculated as:
# SIGMA_T = (8.0 * math.pi / 3.0) * ((QELC * QELC) / (MELC * SPLC * SPLC))**2
SIGMA_T = 6.6524587321000005e-25   #: Thomson/Electron -Scattering cross-section [cm^2]
# Eddington luminosity calculated as: 
# EDDT = 4.0 * math.pi * NWTG * SPLC * MPRT / SIGMA_T          #: Eddington Luminosity prefactor factor [erg/s/g]
EDDT = 63219.620781981204          #: Eddington Luminosity prefactor factor [erg/s/g]

# ---- Astronomical Constants
MSOL = 1.988409870698051e+33       #: Solar Mass [g]
LSOL = 3.828e+33                   #: Solar Luminosity [erg/s]
RSOL = 69570000000.0               #: Solar Radius [cm]
PC = 3.0856775814913674e+18        #: Parsec [cm]
AU = 14959787070000.0              #: Astronomical Unit [cm]
ARCSEC = 4.84813681109536e-06      #: arcsecond in radians []
YR = 31557600.0                    #: year [s]
EVOLT = 1.6021766339999997e-12     #: Electronvolt in ergs
JY = 1e-23                         #: Jansky [erg/s/cm^2/Hz]
KMPERSEC = 100000.0                #: km/s [cm/s]

DAY = 86400.0                                   #: Day [s]
MYR = 31557600000000.0                          #: Mega [s]
GYR = 3.15576e+16                               #: Giga-year [s]
KPC = 3.0856775814913673e+21                    #: Kilo-parsec [cm]
MPC = 3.0856775814913676e+24                    #: Mega-parsec [cm]
GPC = 3.085677581491367e+27                    #: Giga-parsec [cm]


# ----- Constants as calculated from astropy.constants This is a repetition of the above
# ----- and is provided as a reference for the user in case they want to check the values
# ----- with astropy themselves.
# ---- Fundamental Constants
# NWTG = ap.constants.G.cgs.value             #: Newton's Gravitational Constant [cm^3/g/s^2]
# SPLC = ap.constants.c.cgs.value             #: Speed of light [cm/s]
# MELC = ap.constants.m_e.cgs.value           #: Electron Mass [g]
# MPRT = ap.constants.m_p.cgs.value           #: Proton Mass [g]
# QELC = ap.constants.e.gauss.value           #: Fundamental unit of charge (electron charge) [fr]
# KBOLTZ = ap.constants.k_B.cgs.value         #: Boltzmann constant [erg/K]
# HPLANCK = ap.constants.h.cgs.value          #: Planck constant [erg/s]
# SIGMA_SB = ap.constants.sigma_sb.cgs.value  #: Stefan-Boltzmann constant [erg/cm^2/s/K^4]
# SIGMA_T = ap.constants.sigma_T.cgs.value    #: Thomson/Electron -Scattering cross-section [cm^2]

# # ---- Typical astronomy units
# MSOL = ap.constants.M_sun.cgs.value                                #: Solar Mass [g]
# LSOL = ap.constants.L_sun.cgs.value                                #: Solar Luminosity [erg/s]
# RSOL = ap.constants.R_sun.cgs.value                                #: Solar Radius [cm]
# PC = ap.constants.pc.cgs.value                                     #: Parsec [cm]
# AU = ap.constants.au.cgs.value                                     #: Astronomical Unit [cm]
# ARCSEC = ap.units.arcsec.cgs.scale                                 #: arcsecond in radians []
# YR = ap.units.year.to(ap.units.s)                                  #: year [s]
# EVOLT = ap.units.eV.to(ap.units.erg)                               #: Electronvolt in ergs
# JY = ap.units.jansky.to(ap.units.g/ap.units.s**2)                  #: Jansky [erg/s/cm^2/Hz]
# KMPERSEC = (ap.units.km / ap.units.s).to(ap.units.cm/ap.units.s)   #: km/s [cm/s]

# # ---- Derived Constants
# SCHW = 2*NWTG/(SPLC*SPLC)                        #: Schwarzschild Constant (2*G/c^2) [cm]
# EDDT = 4.0*np.pi*NWTG*SPLC*MPRT/SIGMA_T          #: Eddington Luminosity prefactor factor [erg/s/g]

# # Electron-Scattering Opacity ($\kappa_{es} = n_e \sigma_T / \rho = \mu_e \sigma_T / m_p$)
# #     Where $\mu_e$ is the mean-mass per electron, for a total mass-density $\rho$.
# # KAPPA_ES = SIGMA_T / MPRT                       #: Electron scattering opacity [cm^2/g]

# DAY = 86400.0                                   #: Day [s]
# MYR = 1.0e6*YR                                  #: Mega-year [s]
# GYR = 1.0e9*YR                                  #: Giga-year [s]
# KPC = 1.0e3*PC                                  #: Kilo-parsec [cm]
# MPC = 1.0e6*PC                                  #: Mega-parsec [cm]
# GPC = 1.0e9*PC                                  #: Giga-parsec [cm]
