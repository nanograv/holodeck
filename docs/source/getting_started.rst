===============
Getting Started
===============

.. contents:: File Contents
   :local:
   :depth: 1


The Getting Started Guide
=========================

.. toctree::
   :maxdepth: 1

   Calculating Gravitational Waves <calc_gws>
   Definitions and Abbreviations <defs_abbrevs>
   Annotated Bibliography <biblio>


Gravitational Waves (GWs)
=========================

Calculating GW Signatures
-------------------------

For more details, see the document :doc:`Getting Started: Calculating Gravitational Waves <calc_gws>`.

The chirp-mass is defined as: :math:`\mathcal{M} \equiv \frac{\left(m_1 m_2\right)^{3/5}}{M^{1/5}} = M \frac{q^{3/5}}{\left(1 + q\right)^{6/5}}`, for a total mass :math:`M = m_1 + m_2`, and mass-ratio :math:`q \equiv m_2 / m_1 \leq 1`.

The 'hardening timescale' is defined as, :math:`\tau_f \equiv \frac{dt}{d\ln f_r} = \frac{f_r}{df_r/dt}`.

The hardening timescale due purely to GW emission, from a circular binary, is: :math:`\tau_\textrm{GW,circ}= \frac{5}{96}\left(\frac{G\mathcal{M}}{c^3}\right)^{-5/3} \left(2 \pi f_r\right)^{-8/3}`.

The GW strain produced by a circular binary is,

.. math::
   h_\textrm{s,circ}(f_r) = \frac{8}{10^{1/2}} \frac{\left(G\mathcal{M}\right)^{5/3}}{c^4 \, d_L}
           \left(2 \pi f_r\right)^{2/3},

Where :math:`d_L` is the luminosity distance to the source.

The characteristic strain (:math:`h_c`) of the GWB can be calculate based on the comoving-volumetric number-density of binaries in the universe (:math:`n_c \equiv dN/dV_c`) as:

.. math::
   h_c^2(f) = \int_0^\infty \!\! dz \; \frac{dn_c}{dz} \, h_s^2 \, 4\pi c \, d_c^2 \cdot \left(1+z\right) \, \tau_f,

where :math:`d_c` is the comoving distance to a source, related to luminosity distance as: :math:`d_L = d_c \, (1+z)`.  For a finite volume, with a finite number of binaries, this can be discretized to:

.. math::
   h_c^2(f) = \sum_\textrm{redshift} \; \sum_\textrm{binaries} \; h_s^2 \; \frac{4\pi \, c \, d_c^2 \cdot \left(1 + z\right)}{V_\textrm{sim}} \; \tau_f.

To account for cosmic variance and the discreteness of binary sources, we can instead treat this as an expectation value and draw from a Poisson distribution (:math:`\mathcal{P}(x)`):

.. math::
    h_c^2(f) = & \sum_\textrm{redshift} \; \sum_\textrm{binaries} \; h_s^2 \cdot \mathcal{P}(\Lambda), \\
           \Lambda \equiv & \frac{4\pi \, c \, d_c^2 \cdot \left(1 + z\right)}{V_\textrm{sim}} \; \tau_f.

The observed GW frequencies :math:`f` are arbitrarily chosen.  Typically, for pulsar timing arrays, these are chosen based on Nyquist sampling for a given observational duration :math:`T \sim 15 \, \textrm{yr}` and cadence :math:`\Delta t \sim 2 \, \textrm{week}`, such that :math:`f = \left[1/T, 2/T, 3/T, \, \ldots \, , 1/\left(2 \Delta t\right)\right]`.


GW Detection with Pulsar Timing Arrays (PTAs)
---------------------------------------------
Pulsars are rapidly spinning neutron stars which produce beams of radio emission that periodically intersect the observer's line of sight.  Millisecond pulsars, in particular, often maintain incredibly precise periodicities with precision down to the order of 1s to 100s of nanoseconds.  The presence of GWs induce variations to the times of arrival (TOAs) measured by observatories on Earth.

Many noise sources (e.g. due to the neutron star, the source of radio emission, or the radio observatory) can also produce TOA variations.  Unlike noise sources, GWs are believed to produce a very unique spatial correlation pattern in the TOA variations across the sky.  In particular, the angular correlation function of TOA variations between pairs of pulsars produces a pseudo-quadrupolar pattern as a function of angular separation which is believed to be a 'smoking gun' signatures of GWs (i.e. a signature that is not produced by any noise source).  The identification of this angular correlation function can thus be used to detect gravitational waves.

The frequency range that PTAs are sensitive to is determined primarily by the ability to reconstruct periodic signals from the time-series data of pulsar TOAs.  The lowest sensitive frequency is determined by the Nyquist frequency, i.e. the inverse of the total observing duration.  High-precision TOA measurements have been performed for about two decades, meaning that PTAs are starting to probe frequencies as low as nanohertz (nHz).  The highest sensitive frequency is determined by the typically interval between observations, which tends to be on the order of a few weeks, or hundreds of nanohertz.

While there have been a large number of proposed sources of GWs in the nanohertz regime, the best studied sources are binaries of supermassive black holes, with total masses :math:`M \gtrsim 10^8 \, M_\odot`.


Methods for Simulating MBH Binary Populations
=============================================
Simulations of MBH Binary (MBHB) populations require three components:
1) Origins: the events which produce encounters between pairs (or more) of MBHs, which are typically galaxy-galaxy mergers (which occur on large scales of :math:`\mathrm{kpc} - \mathrm{Mpc}`);
2) Populations: the properties of the MBHs (and often their host galaxies) involved in the encounter events (at least a redshift and pair of masses); and
3) Evolution: the process by which the two (or more) MBHs are able to reach the small binary separations at which detectable GWs are produced.

(1) and (2) can be produced in a variety of ways, and in numerous combinations of components from cosmological simulations or electromagnetic observations.  **Cosmological hydrodynamic simulations** strive to model the fundamental processes underlying galaxy and star formation to produce populations of galaxies, including galaxy mergers, and often massive black holes.  **Semi-analytic models (SAMs)** use relatively simple analytic prescriptions for propterties of galaxy and MBHs, typically calibrated to observations, to produce initial populations of binary MBHs.  **Semi-empirical models (SEMs)** are similar to SAMs, but rely more strongly on observational relationships, and are typically much more complex/comprehensive, often including models for not only galaxy populations but even the internal structure of galaxies and their co-evolution with MBHs.  In this way, SEMs are somewhere in-between SAMs and hydrodynamic simulations, but rely heavily on observations of populations of galaxies and MBHs/AGNs.  Observations can also be used to construct binary population more directly, e.g. by starting from **observational catalogs** of AGN or quasars, and somehow prescribing the occurrence rate of binaries.  These observational catalogs will typically use components of SAMs, SEMs to hydrodynamic simulations to complement the directly observed properties.

Semi-Analytic Models
--------------------

Theory
^^^^^^

Implementation
^^^^^^^^^^^^^^


Binary Evolution/Hardening Models
=================================
Prescriptions for modeling binary evolution on less-than about kpc scales, with interactions between binaries are their galactic environments.


Observational Constraints on MBH and MBH-Binary Populations
===========================================================