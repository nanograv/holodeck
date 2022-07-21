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


Calculating Gravitational Waves
===============================

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


GW Detection with Pulsar Timing Arrays
======================================


Methods for Simulating MBH Binary Populations
=============================================
Hydrodynamic simulations, semi-analytic / semi-empirical models, observational catalogs.  Similarities and differences.


Binary Evolution/Hardening Models
=================================
Prescriptions for modeling binary evolution on less-than about kpc scales, with interactions between binaries are their galactic environments.


Observational Constraints on MBH and MBH-Binary Populations
===========================================================