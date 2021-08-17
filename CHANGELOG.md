# Change Log

## Future (To-Do)
* General
    - [ ] combine galaxy-scaling relationship implementations in `holodeck.observations` and `holodeck.sam`.
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

----

## Current


----

## Past

### v0.1 - 2021/08/15


* Basic GW spectra can be generated using simple versions of population synthesis based on:
    - A finite, discrete population of binaries from the Illustris simulations
    - Continuous distributions from semi-analytic modeling
    - Continuous distributions from semi-analytic modeling, with Illustris merger rates, calibrated to local galaxy observations.
* A class-based implementation is used in a way to facilitate subclassing (i.e. extensibility).
* Only the simplest models for binary evolution (i.e. fixed time-delays and GW emission) are currently included.
* Continuous population distributions can be easily interfaced with the `kalepy` package to facilitate discrete sampling.  Even without formal discrete sampling, proper GW (foreground and background) statistics can be approximated.
* Cosmology class (subclass of astropy.cosomology) providing convenience functions and more rapid calculations on arrays (via interpolation).
