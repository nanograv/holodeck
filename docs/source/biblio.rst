======================
Annotated Bibliography
======================

Generating references on :ref:`NASA ADS <https://ui.adsabs.harvard.edu/user/libraries/DDrcbmynS-CEJgm24zT0ig>`:

* For short references (e.g. in code files)::

    * [%1.1h%Y]_ %3.1M (%Y).

* For :ref:`annotated bibliography <Bibliography>` section::

    * **%3.1M %Y**, [%1.1h%Y]_ - `%T <%u>`_\n

* For full references (e.g. in the :ref:`References` section below)::

    .. [%1.1h%Y] %3.1M (%Y), %q, %V, %S.\n   %T\n   %u\n


Bibliography
============

* **Begelman, Blandford & Rees 1980**, [BBR1980]_ - `Massive black hole binaries in active galactic nuclei <https://ui.adsabs.harvard.edu/abs/1980Natur.287..307B/abstract>`_
    * The definitive early discussion of massive black-hole binary evolution, outlining the different stages of environmental interaction (dynamical friction, stellar scattering, etc) and mentioning the possibility of stalling in the parsec regime.
    * Includes simplistic, but useful prescriptions for calculating timescales for each regime of evolution.

* **Genel et al. 2014**, [Genel2014]_ - `Introducing the Illustris project: the evolution of galaxy populations across cosmic time <https://ui.adsabs.harvard.edu/abs/2014MNRAS.445..175G>`_
  * One of the standard references for the original Illustris simulations written by the Illustris team.
  * Focuses on the redshift evolution of simulated galaxies.

* **Hogg 1999**, [Hogg1999]_ - `Distance measures in cosmology <https://ui.adsabs.harvard.edu/abs/1999astro.ph..5116H>`_.
    * This is the go-to reference/cheat-sheet for basic cosmological calculations such as distances (comoving, luminosity), volume of the universe, lookback times, etc.

* **Kelley, Blecha, and Hernquist 2017**, [Kelley2017a]_ - `Massive black hole binary mergers in dynamical galactic environments <https://ui.adsabs.harvard.edu/abs/2017MNRAS.464.3131K>`_
  * Describes the MBH-MBH mergers from the Illustris cosmological hydrodynamic simulations.
  * Results include comprehensive semi-analytic models for post-processing the binary mergers at sub-grid scales.

* **Kelley et al. 2017**, [Kelley2017b]_ - `The gravitational wave background from massive black hole binaries in Illustris: spectral features and time to detection with pulsar timing arrays <https://ui.adsabs.harvard.edu/abs/2017MNRAS.471.4508K>`_
  * Uses the MBH-MBH merger catalogs from Illustris, along with comprehensive semi-analytic models of the unresolved binary evolution process, to calculate the expected properties of the GWB and PTA detection prospects.

* **Kelley et al. 2018**, [Kelley2018]_ - `Single sources in the low-frequency gravitational wave sky: properties and time to detection by pulsar timing arrays <https://ui.adsabs.harvard.edu/abs/2018MNRAS.477..964K>`_
  * Uses the MBH-MBH merger catalogs from Illustris, along with comprehensive semi-analytic models of the unresolved binary evolution process, to calculate the expected properties of individual continuous wave (CW) GW sources and PTA detection prospects.

* **Nelson et al. 2015**, [Nelson2015]_ - `The illustris simulation: Public data release <https://ui.adsabs.harvard.edu/abs/2015A&C....13...12N>`_
  * One of the standard references for the original Illustris simulations written by the Illustris team.
  * Summarizes the Illustris public data and API.

* **Phinney 2001**, [Phinney2001]_ - `A Practical Theorem on Gravitational Wave Backgrounds <https://ui.adsabs.harvard.edu/abs/2001astro.ph..8028P/abstract>`_
    * Pioneering analytic calculation of the GWB by integrating the GW emission of binaries over the history of the universe.

* **Rodriguez-Gomez et al. 2015**, [Rodriguez-Gomez2015]_ - `The merger rate of galaxies in the Illustris simulation: a comparison with observations and semi-empirical models <https://ui.adsabs.harvard.edu/abs/2015MNRAS.449...49R>`_
  * Methods and results for galaxy-galaxy merger rates from the Illustris simulations.
  * These rates are used to prescribe merger rates in the observational-populations `holodeck` catalogs.

* **Sesana et al. 2008** [Sesana2008]_ - `The stochastic gravitational-wave background from massive black hole binary systems: implications for observations with Pulsar Timing Arrays <https://ui.adsabs.harvard.edu/abs/2008MNRAS.390..192S/abstract>`_.
    * Thorough description of how to calculate the GWB, with a discussion on some of the nuances.
    * Particular attention is given to the difference between the analytic formalism of [Phinney2001]_ and numerical / semi-analytic approaches, i.e. the effects of discreteness of binary sources which produces a turnover in the GWB spectrum at high frequencies.

* **Sijacki et al. 2015**, [Sijacki2015]_ - `The Illustris simulation: the evolving population of black holes across cosmic time <https://ui.adsabs.harvard.edu/abs/2015MNRAS.452..575S>`_
  * One of the standard references for the original Illustris simulations written by the Illustris team.
  * Describes the MBH/AGN population derived from the simulations.

* **Springel 2010**, [Springel2010]_ - `E pur si muove: Galilean-invariant cosmological hydrodynamical simulations on a moving mesh <https://ui.adsabs.harvard.edu/abs/2010MNRAS.401..791S>`_
  * Methods paper for the arepo hydrodynamics code, used in the Illustris simulations.

* **Vogelsberger et al. 2014**, [Vogelsberger2014]_ - `Introducing the Illustris Project: simulating the coevolution of dark and visible matter in the Universe <https://ui.adsabs.harvard.edu/abs/2014MNRAS.444.1518V>`_
  * One of the standard references for the original Illustris simulations written by the Illustris team.
  * Gives a summary of the simulation methodology and results.



References
==========
These are provided here for easy copy-and-paste usage in other files.

.. [Behroozi2013] : Behroozi, Wechsler & Conroy 2013.  ApJ, 770, 1.
    The Average Star Formation Histories of Galaxies in Dark Matter Halos from z = 0-8
    https://ui.adsabs.harvard.edu/abs/2013ApJ...770...57B/abstract

.. [BBR1980] Begelman, Blandford & Rees 1980.  Nature, 287, 5780.
    Massive black hole binaries in active galactic nuclei.
    https://ui.adsabs.harvard.edu/abs/1980Natur.287..307B/abstract

.. [Chen2017] Chen, Sesana, & Del Pozzo 2017
    Efficient computation of the gravitational wave spectrum emitted by eccentric massive
    black hole binaries in stellar environments
    https://ui.adsabs.harvard.edu/abs/2017MNRAS.470.1738C/abstract

.. [Chen2019] Chen, Sesana, Conselice 2019.  MNRAS, 488, 1.
    Constraining astrophysical observables of galaxy and supermassive black hole binary mergers
    using pulsar timing arrays
    https://ui.adsabs.harvard.edu/abs/2019MNRAS.488..401C/abstract

.. [EN2007] Enoki & Nagashima 2007.  PTP, 117, 2.  astro-ph/0609377.
    The Effect of Orbital Eccentricity on Gravitational Wave Background Radiation from Supermassive Black Hole Binaries
    https://ui.adsabs.harvard.edu/abs/2007PThPh.117..241E/abstract

.. [Enoki2004] Enoki, Inoue, Nagashima, & Sugiyama 2004.  ApJ, 615, 1.  astro-ph/0404389.
    Gravitational Waves from Supermassive Black Hole Coalescence in a Hierarchical Galaxy Formation Model
    https://ui.adsabs.harvard.edu/abs/2004ApJ...615...19E/abstract

.. [Genel2014] : Genel et al. (2014), MNRAS, 445, 1.
   Introducing the Illustris project: the evolution of galaxy populations across cosmic time
   https://ui.adsabs.harvard.edu/abs/2014MNRAS.445..175G

.. [Guo2010] Guo, White, Li & Boylan-Kolchin 2010.  MNRAS, 404, 3.
    How do galaxies populate dark matter haloes?
    https://ui.adsabs.harvard.edu/abs/2010MNRAS.404.1111G/abstract

.. [WMAP9] Hinshaw, Larson, Komatsu et al. 2013. ApJS, 208, 2. (1212.5226).
    Nine-year Wilkinson Microwave Anisotropy Probe (WMAP) Observations: Cosmological Parameter Results.
    https://ui.adsabs.harvard.edu/abs/2013ApJS..208...19H/abstract

.. [Hogg1999] Hogg 1999.  arXiv. (astro-ph/9905116).
    Distance measures in cosmology.
    https://ui.adsabs.harvard.edu/abs/1999astro.ph..5116H

.. [Kelley2017a] Kelley, Blecha, and Hernquist (2017), MNRAS, 464, 3.
   Massive black hole binary mergers in dynamical galactic environments
   https://ui.adsabs.harvard.edu/abs/2017MNRAS.464.3131K

.. [Kelley2017b] Kelley et al. (2017), MNRAS, 471, 4.
   The gravitational wave background from massive black hole binaries in Illustris: spectral features and time to detection with pulsar timing arrays
   https://ui.adsabs.harvard.edu/abs/2017MNRAS.471.4508K

.. [Kelley2018] Kelley et al. (2018), MNRAS, 477, 1.
   Single sources in the low-frequency gravitational wave sky: properties and time to detection by pulsar timing arrays
   https://ui.adsabs.harvard.edu/abs/2018MNRAS.477..964K

.. [Klypin2016] : Klypin, Yepes, Gottlöber, et al. 2016.  MNRAS, 457, 4.
    MultiDark simulations: the story of dark matter halo concentrations and density profiles
    https://ui.adsabs.harvard.edu/abs/2016MNRAS.457.4340K/abstract

.. [KH2013] Kormendy & Ho 2013. ARAA, 51, 1.
    Coevolution (Or Not) of Supermassive Black Holes and Host Galaxies
    https://ui.adsabs.harvard.edu/abs/2013ARA%26A..51..511K/abstract

.. [MM2013] McConnell & Ma 2013.  ApJ, 764, 2.
    Revisiting the Scaling Relations of Black Hole Masses and Host Galaxy Properties
    https://ui.adsabs.harvard.edu/abs/2013ApJ...764..184M/abstract

.. [NFW1997] Navarro, Frenk & White 1997.  ApJ, 490, 2.
    A Universal Density Profile from Hierarchical Clustering
    https://ui.adsabs.harvard.edu/abs/1997ApJ...490..493N/abstract

.. [Nelson2015] Nelson et al. (2015), A&C, 13,.
   The illustris simulation: Public data release
   https://ui.adsabs.harvard.edu/abs/2015A&C....13...12N

.. [Peters1964] Peters 1964.  PR, 136, 4B.
    Gravitational Radiation and the Motion of Two Point Masses
    https://ui.adsabs.harvard.edu/abs/1964PhRv..136.1224P/abstract

.. [Phinney2001] Phinney 2001.  arXiv. (astro-ph/0108028).
    A Practical Theorem on Gravitational Wave Backgrounds.
    https://ui.adsabs.harvard.edu/abs/2001astro.ph..8028P/abstract

.. [Quinlan1996] Quinlan 1996
    The dynamical evolution of massive black hole binaries I. Hardening in a fixed stellar background
    https://ui.adsabs.harvard.edu/abs/1996NewA....1...35Q/abstract

.. [Rodriguez-Gomez2015] : Rodriguez-Gomez et al. (2015), MNRAS, 449, 1.
   The merger rate of galaxies in the Illustris simulation: a comparison with observations and semi-empirical models
   https://ui.adsabs.harvard.edu/abs/2015MNRAS.449...49R

.. [Sesana2004] Sesana, Haardt, Madau, & Volonteri 2004.  ApJ, 611, 2.  astro-ph/0401543.
    Low-Frequency Gravitational Radiation from Coalescing Massive Black Hole Binaries in Hierarchical Cosmologies
    http://adsabs.harvard.edu/abs/2004ApJ...611..623S

.. [Sesana2006] Sesana, Haardt & Madau et al. 2006
    Interaction of Massive Black Hole Binaries with Their Stellar Environment. I. Ejection of Hypervelocity Stars
    https://ui.adsabs.harvard.edu/abs/2006ApJ...651..392S/abstract

.. [Sesana2008] Sesana, Vecchio, Colacino 2008.  MNRAS, 390, 1. (0804.4476).
    The stochastic gravitational-wave background from massive black hole binary systems:
    implications for observations with Pulsar Timing Arrays.
    https://ui.adsabs.harvard.edu/abs/2008MNRAS.390..192S/abstract

.. [Sesana2010] Sesana 2010
    Self Consistent Model for the Evolution of Eccentric Massive Black Hole Binaries in Stellar Environments:
    Implications for Gravitational Wave Observations
    https://ui.adsabs.harvard.edu/abs/2010ApJ...719..851S/abstract

.. [Sijacki2015] Sijacki et al. (2015), MNRAS, 452, 1.
   The Illustris simulation: the evolving population of black holes across cosmic time
   https://ui.adsabs.harvard.edu/abs/2015MNRAS.452..575S

.. [Springel2010] Springel (2010), MNRAS, 401, 2.
   E pur si muove: Galilean-invariant cosmological hydrodynamical simulations on a moving mesh
   https://ui.adsabs.harvard.edu/abs/2010MNRAS.401..791S

.. [Vogelsberger2014] Vogelsberger et al. (2014), MNRAS, 444, 2.
   Introducing the Illustris Project: simulating the coevolution of dark and visible matter in the Universe
   https://ui.adsabs.harvard.edu/abs/2014MNRAS.444.1518V

