# Change Log

## Future (To-Do)
* General
    - [ ] Add fit/interpolated GW spectra from Chen19 into SAM modules.
* Structures for comprehensive binary evolution / hardening modules
    - [ ] a class-based structure allowing for modular hardening processes to be added/removed/modified easily.
    - [ ] structures allowing for storing host-galaxy information (e.g. based on semi-analytic models, Illustris, observations, etc) that can be used for hardening calculations
    - [ ] accretion rate models based on semi-analytic models, Illustris, observations, etc
* Explicit evolution/hardening models for
    - [ ] dynamical friction
    - [ ] stellar scattering using uniform/standard stellar distributions (e.g. isotropic & isothermal)
    - [ ] comprehensive stellar scattering using arbitrary stellar distributions
    - [ ] circumbinary accretion mediated hardening ( inward-migration)
    - [ ] circumbinary accretion mediated softening (outward-migration)
    - [ ] eccentric binary evolution
    - [ ] triple MBH interactions
* Comparisons with observations (particularly EM) to calibrate sythesized populations
    - [ ] accurate catalogs of 'direct' MBH mass measurements from the local universe
    - [ ] approximate catalogs of 'indirect' MBH mass measurements from populations of AGN/quasars
    - [ ] MBH--host-galaxy scaling relationships
    - [ ] AGN/Quasar luminosity functions
    - [ ] constraints on kpc--Mpc scale galaxy and AGN mergers
    - [ ] constraints on sub-kpc separation binary AGN based on EM candidates (and upper-limits)
* Gaussian Processes
    - [ ] How significant are deviations in predicted spectra from Gaussians?  What produces those deviations, are they single (or single-like) sources, or are they actual "population" trends?  (Former is okay to ignore, latter is not!)
    - [ ]
* Testing
  * Add sphinx docs build to github action for testing


----

## Current


----

## Past

### v0.2 - 2022/03/28

* Binary Evolution (`evolution.py`)
    - Now tracking hardening rates in evolution.
    - Simple implementation of some binary hardening models, both physical and phenomenological (i.e. power-law like).
    - Modules for Dynamical Friction, Stellar Scattering, and GW hardening.
* Logistical and Internals
    - Added submodule for logging (`log.py`)
    - Added submodule for plotting (`plot.py`)
    - Added submodule for observational data and relations (`observations.py`)
    - New, and also improvements to old, notebooks for testing and demonstration purposes.  Addition of more unit tests and test scripts.
    - Extensive additions to utility / mathematical / numerical functions (`utils.py`).
    - Improved README.md, and started adding basics to holodeck paper manuscript.
* Populations
    - Cleaned up of observationally-based populations (`pop_observational.py`)
    - Unified implementation of MBH-galaxy relationships (`relations.py`)
    - Significant cleanup and upgrades in Semi-Analytic Models based populations (`sam.py`)
        - Developed methodology for sampling discrete binaries from continuous distributions (in coordination with `kalepy` modules)

### v0.1 - 2021/08/15

* Basic GW spectra can be generated using simple versions of population synthesis based on:
    - A finite, discrete population of binaries from the Illustris simulations
    - Continuous distributions from semi-analytic modeling
    - Continuous distributions from semi-analytic modeling, with Illustris merger rates, calibrated to local galaxy observations.
* A class-based implementation is used in a way to facilitate subclassing (i.e. extensibility).
* Only the simplest models for binary evolution (i.e. fixed time-delays and GW emission) are currently included.
* Continuous population distributions can be easily interfaced with the `kalepy` package to facilitate discrete sampling.  Even without formal discrete sampling, proper GW (foreground and background) statistics can be approximated.
* Cosmology class (subclass of astropy.cosomology) providing convenience functions and more rapid calculations on arrays (via interpolation).
