# Change Log


## Future (To-Do)

* General
    * [ ] Add fit/interpolated GW spectra from Chen19 into SAM modules.
* Explicit evolution/hardening models for
    * [ ] comprehensive stellar scattering using arbitrary stellar distributions
    * [ ] eccentric binary evolution
    * [ ] triple MBH interactions
* Comparisons with observations (particularly EM) to calibrate synthesized populations
    * [ ] accurate catalogs of 'direct' MBH mass measurements from the local universe
    * [ ] approximate catalogs of 'indirect' MBH mass measurements from populations of AGN/quasars
    * [ ] AGN/Quasar luminosity functions
    * [ ] constraints on kpc--Mpc scale galaxy and AGN mergers
    * [ ] constraints on sub-kpc separation binary AGN based on EM candidates (and upper-limits)


## Current

## v1.6 - 2024/05/01

* DEPRECATIONS:
    * `holodeck.librarian.py` ==> `holodeck.librarian.lib_tools`
        * Rename submodule.  All components remain the same.  All `lib_tools` elements are now also imported into the `librarian` namespace.  i.e. elements like `holodeck.librarian.lib_tools._Param_Space` will now be accessible via `holodeck.librarian._Param_Space`.
    * Library filenames:
        * Standard library simulation files will now be saved to the 'library_sims' subdirectory, and filenames 'library__p######.npz'.  Combined library files will now be 'sam-library.hdf5'.
        * 'Domain' simulation files will be saved to the 'domain_sims' subdirectory, and filenames 'domain__p######.npz'.  Combined domain files will now be 'sam-domain.hdf5'.

* BUGS:
    * Semi-Analytic Models
        * `Semi_Analytic_Model._dynamic_binary_number_at_fobs_inconsistent` was not checking for systems that had already coalesced.  Only effected GW-only evolution using the python version of the calculation.

* DEFAULTS:
    * Semi-Analytic Models
        * `Semi_Analytic_Models` instances now use galaxy merger-rates (instead of galaxy pair-fractions and merger-times) by default.  To use GPF+GMT SAMs, the user must pass in at least a GPF instance manually.

* General Changes

    * Semi-Analytic Models (`holodeck.sams`)
        * Improve accuracy of dynamic binary number calculation for consistent evolution models.

    * `holodeck.librarian`
        * Added functionality to construct 'domain' sets of simulations, to explore each parameter in a parameter-space one at a time.
        * NOTE: Standard library files will now be called "sam-library.hdf5" instead of "sam_lib.hdf5"


## v1.5.2 - 2024/04/12

* DEPRECATIONS
    * `host_relations.py`: remove the `mamp` parameter and `MASS_AMP` attributes in the MMBulge relationships, and use `mamp_log10` and `MASS_AMP_LOG10` exclusively.


## v1.5 - 2024/03/29

* Deprecated `relations.py`.
    * **Material from this file has mostly been moved to `host_relations.py`**.  The components for galaxy/halo density/velocity profiles have been moved to `galaxy_profiles.py`.  Stellar-mass vs. halo-mass relations are still in `host_relations.py`.
    * All of the same material can temporarily still be accessed/imported from `relations.py`, and it will log/print a deprecation warning.
* **M-Mbulge relations now use separate bulge-fractions.**
    * All subclasses of `_MMBulge_Relation` now utilize separate bulge-fraction instances, implemented as subclasses of the new `holodeck.host_relations._Bulge_Frac` class.
    * The overall API remains unchanged (users can still perform conversions from total stellar-mass `mstar` to black-hole masses `mbh`), but internally conversions from total stellar-mass to stellar bulge-mass are performed by the bulge-fraction instances, and then stellar bulge-masses are converted to black-hole masses by the M-Mbulge instances.
    * The API for M-Mbulge relations has also been cleaned up and unified.
    * Two `_Bulge_Frac` subclasses have been implememted:
        * `BF_Constant` which is a single, fixed bulge-fraction value.  This maintains the behavior that was previously performed within `_MMBulge_Relation`.
        * `BF_Sigmoid` is a new implementation that transitions from one bulge-fraction value at asymptotically low stellar-masses, up to a second bulge-fraction at and above a fixed characteristic stellar mass.  The 'width' or 'steepness' of the transition can also be varied.

----


## Past

### v0.2 - 2022/03/28

* Binary Evolution (`evolution.py`)
    * Now tracking hardening rates in evolution.
    * Simple implementation of some binary hardening models, both physical and phenomenological (i.e. power-law like).
    * Modules for Dynamical Friction, Stellar Scattering, and GW hardening.
* Logistical and Internals
    * Added submodule for logging (`log.py`)
    * Added submodule for plotting (`plot.py`)
    * Added submodule for observational data and relations (`observations.py`)
    * New, and also improvements to old, notebooks for testing and demonstration purposes.  Addition of more unit tests and test scripts.
    * Extensive additions to utility / mathematical / numerical functions (`utils.py`).
    * Improved README.md, and started adding basics to holodeck paper manuscript.
* Populations
    * Cleaned up of observationally-based populations (`pop_observational.py`)
    * Unified implementation of MBH-galaxy relationships (`relations.py`)
    * Significant cleanup and upgrades in Semi-Analytic Models based populations (`sam.py`)
        * Developed methodology for sampling discrete binaries from continuous distributions (in coordination with `kalepy` modules)

### v0.1 - 2021/08/15

* Basic GW spectra can be generated using simple versions of population synthesis based on:
    * A finite, discrete population of binaries from the Illustris simulations
    * Continuous distributions from semi-analytic modeling
    * Continuous distributions from semi-analytic modeling, with Illustris merger rates, calibrated to local galaxy observations.
* A class-based implementation is used in a way to facilitate subclassing (i.e. extensibility).
* Only the simplest models for binary evolution (i.e. fixed time-delays and GW emission) are currently included.
* Continuous population distributions can be easily interfaced with the `kalepy` package to facilitate discrete sampling.  Even without formal discrete sampling, proper GW (foreground and background) statistics can be approximated.
* Cosmology class (subclass of astropy.cosomology) providing convenience functions and more rapid calculations on arrays (via interpolation).
