===============
Getting Started
===============

Overview
========

The `holodeck` package aims to simulate populations of MBH binaries, and calculate their GW signals.  In general the calculation proceeds in three stages.

(1) **Population**: Construct an initial population of MBH 'binaries'.  This is typically done for pairs of MBHs when their galaxies merge (i.e. long before the two MBHs are actually a gravitationally-bound binary).  Constructing the initial binary population may occur in a single step: e.g. gathering MBH-MBH encounters from cosmological hydrodynamic simulations; or it may occur over two steps: (i) gathering galaxy-galaxy encounters, and (ii) prescribing MBH properties for each galaxy.
(2) **Evolution**: Evolve the binary population from their initial conditions (i.e. large separations) until coalescence (i.e. small separations).  The complexity of this evolutionary stage can range tremendously in complexity.  In the simplest models, binaries are assumed to coalesce instantaneously (in that the age of the universe is the same at formation and coalescence), and are assumed to evolve purely due to GW emission (in that the time spent in any range of orbital frequencies can be calculated from the GW hardening timescale).  Note that these two assumptions are contradictory.
(3) **Gravitational Waves**: Calculate the resulting GW signals based on the binaries and their evolution.  Note that GWs can only be calculated based on some sort of model for binary evolution.  The model may be extremely simple, in which case it is sometimes glanced over.

The contents of this file are as follows.

.. contents:: File Contents
   :local:
   :depth: 2


Files in the Getting Started Guide
==================================

.. toctree::
   :maxdepth: 1

   Getting Started Overview/Introduction <getting_started>
   Calculating Gravitational Waves <calc_gws>
   Definitions and Abbreviations <defs_abbrevs>
   Annotated Bibliography <biblio>


Gravitational Waves (GWs)
=========================

Calculating GW Signatures
-------------------------

For more details, see the document :doc:`Getting Started: Calculating Gravitational Waves <calc_gws>`.
For the implementation of much of the gravitational wave calculations, see the :mod:`GravWaves submodule <holodeck.gravwaves>`.

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

   (1) **Origins**: the events which produce encounters between pairs (or more) of MBHs, which are typically galaxy-galaxy mergers (which occur on large scales of :math:`\mathrm{kpc} - \mathrm{Mpc}`);
   (2) **Populations**: the properties of the MBHs (and often their host galaxies) involved in the encounter events (at least a redshift and pair of masses); and
   (3) **Evolution**: the process by which the two (or more) MBHs are able to reach the small binary separations at which detectable GWs are produced.

Components (1) and (2) can be produced in a variety of ways described below and, indeed, in numerous combinations of these methods.

   (A) **Cosmological hydrodynamic simulations** strive to model the fundamental processes underlying galaxy and star formation to produce populations of galaxies, including galaxy mergers, and often massive black holes.
   (B) **Semi-analytic models (SAMs)** use relatively simple analytic prescriptions for properties of galaxy and MBHs, typically calibrated to observations, to produce initial populations of binary MBHs.
   (C) **Semi-empirical models (SEMs)** are similar to SAMs, but rely more strongly on observational relationships, and are typically much more complex/comprehensive, often including models for not only galaxy populations but even the internal structure of galaxies and their co-evolution with MBHs.  In this way, SEMs are somewhere in-between SAMs and hydrodynamic simulations, but rely heavily on observations of populations of galaxies and MBHs/AGNs.
   (D) **Observational catalogs** can also be used to construct binary population more directly, e.g. by starting from  observed AGN or quasars, and prescribing the occurrence rates of binaries.  These observational catalogs will typically use components of SAMs, SEMs, or hydrodynamic simulations to complement the directly observed properties of systems.

Component (3) always requires the same type of implementation: some level of semi-analytic modeling of the physical interactions between MBHBs and the environments of their surrounding galactic nuclei.  The physical processes which mediate *"binary hardening"*, the process of extracting energy and angular momentum to allow the binary orbit to shrink, cannot be resolved directly in either cosmological hydrodynamic simulations or observations which are currently both limited to physical scales :math:`r \gtrsim 100 \, \mathrm{pc}` where the most massive MBHBs are only just starting to gravitationally interact with each other.

Semi-Analytic Models
--------------------

Theory
^^^^^^

Define the distribution function of sources as :math:`F(M,q,a,z) = d^2 n(M,q,a,z) / dM dq`.  Here :math:`M` is the total mass of each systems, the mass-ratio is :math:`q\equiv m_2/m_1 \leq 1`, :math:`a` is the binary separation, and :math:`z` is the redshift.  We can write the conservation equation for binaries as of function of redshift as,

.. math::
   \frac{\partial F}{\partial z} +
      \frac{\partial }{\partial M} \left({F \frac{\partial M}{\partial z}}\right) +
      \frac{\partial }{\partial q} \left({F \frac{\partial q}{\partial z}}\right) +
      \frac{\partial }{\partial a} \left({F \frac{\partial a}{\partial z}}\right) = S_{\!F}(M, q, a, z).
   :name: eq:conservation

Here :math:`S_{\!F}` is a source/sink function that can account for the creation or destruction of binaries.

We consider the standard semi-analytic model (SAM) formalism of MBH binary populations [Sesana2008]_, [Chen2019]_.  In this style of calculation, :math:`F` is determined in a region of parameter space that can be observed/estimated, and this is evolved to find the distribution in a different region of parameter space that is of interest.  In practice, the observed parameter space is galaxies and galaxy mergers, and the parameter space of interest is closely separated MBH binaries that could be GW detectable.  Thus we assume that all binary 'formation' is encapsulated from binaries moving from one part of parameter space (i.e. large separations and redshifts) to other parts of parameter space (i.e. smaller separations and redshifts), and we set :math:`S_{\!F} = 0`.

We will express the distribution function as a product of a mass function, and a pair fraction:

.. math::
   F(M,q,z) = \frac{\Phi(M, z)}{M \ln\!10} \cdot P(M,q,z),
   :name: eq:dist_func

where the mass function, :math:`\Phi(M, q, z) \equiv \frac{\partial n_g}{\partial \log_{10}M}`, is calculated based on the number density of galaxies (:math:`n_g`).  We assume that there is a one-to-one mapping from galaxy mass to MBH mass, such that the galaxy mass-function can still be used to uniquely define the mass distribution of MBHs.  Typically the MBH--galaxy relation is given in terms of an :math:`M_\mathrm{BH}-M_\mathrm{bulge}` relation [KH2013]_, which is an observationally derived relation between the mass of MBHs and the mass of the stellar bulge component of their host galaxy.
In [Chen2019]_, the pair fraction is measured over some range of separations, and the separation-dependence is suppressed, i.e. :math:`P = \int_{a_0}^{a_1} P_a \, da`.

From :math:numref:`eq:conservation`, we use the chain rule to mix time and redshift evolution, and assume that the mass-change of binaries is negligible, i.e. :math:`\frac{\partial m}{\partial t} = 0` and :math:`\frac{\partial q}{\partial t} = 0`, giving:

.. math::
   \frac{\partial F}{\partial z} = - \frac{\partial t}{\partial z} \frac{\partial}{\partial a} \left(F \frac{\partial a}{\partial t}\right).

The binary population is assumed to be changing only in separation and redshift, which are related by :math:`\partial a / \partial z = (\partial a / \partial t) (\partial t / \partial z)`.  Because the overall number-density is conserved, we can take a finite step in separation and time/redshift, :math:`a\rightarrow a'` and :math:`z\rightarrow z'`.  Here the time it takes for a binary to go from :math:`a \rightarrow a'` is :math:`T(M,q,a,z|a')`, which leads to a redshift at the later time of :math:`z' = z'(t + T)`.  So far, we have left the binary separation :math:`a` as implicit in the expression for F.  To obtain the standard expression [Chen2019]_ (Eq.5), we make the approximation that,

.. math::
   \frac{\partial}{\partial a} \left( F(M,q,z) \frac{\partial a}{\partial t} \right) \approx \frac{F}{T(M,q,a,z|a')}.

Thus giving,

.. math::
   \frac{\partial F(M,q,a',z')}{\partial z'} = \frac{\partial n}{\partial M \partial q \partial z'} = - \frac{\partial t}{\partial z} \frac{\Phi(M,z) \, P(M,q,z)}{T(M,q,a,z|a')}.
   :name: eq:cont_eq_result

Combining :math:numref:`eq:cont_eq_result` with :math:numref:`num_num_dens`, we can finally write,

.. math::
   \frac{\partial N}{\partial M \partial q \partial z \partial \ln\!f_r} = \frac{\Phi(M,z) \, P(M,q,z)}{T(M,q,a,z|a')} \, \tau_f \, \frac{\partial V_c}{\partial z}.
   :name: eq:sam_final


Implementation
^^^^^^^^^^^^^^
Full code documentation: :mod:`SAM submodule <holodeck.sam>` submodule.

The core element of the SAM module is the :class:`Semi_Analytic_Model <holodeck.sam.Semi_Analytic_Model>` class.  This class requires four
components as arguments:

(1) Galaxy Stellar Mass Function (GSMF): gives the comoving number-density of galaxies as a function
    of stellar mass.  This is implemented as subclasses of the :class:`_Galaxy_Stellar_Mass_Function <holodeck.sam._Galaxy_Stellar_Mass_Function>`
    base class.
(2) Galaxy Pair Fraction (GPF): gives the fraction of galaxies that are in a 'pair' with a given
    mass ratio (and typically a function of redshift and primary-galaxy mass).  Implemented as
    subclasses of the :class:`_Galaxy_Pair_Fraction <holodeck.sam._Galaxy_Pair_Fraction>` subclass.
(3) Galaxy Merger Time (GMT): gives the characteristic time duration for galaxy 'mergers' to occur.
    Implemented as subclasses of the :class:`_Galaxy_Merger_Time <holodeck.sam._Galaxy_Merger_Time>` subclass.
(4) M_bh - M_bulge Relation (mmbulge): gives MBH properties for a given galaxy stellar-bulge mass.
    Implemented as subcalsses of the :class:`holodeck.relations._MMBulge_Relation` subclass.

The :class:`Semi_Analytic_Model <holodeck.sam.Semi_Analytic_Model>` class defines a grid in parameter space of total MBH mass ($M=M_1 + M_2$),
MBH mass ratio ($q \\equiv M_1/M_2$), redshift ($z$), and at times binary separation
(semi-major axis $a$) or binary rest-frame frequency ($f_r$).  Over this grid, the distribution of
comoving number-density of MBH binaries in the Universe is calculated.  Methods are also provided
that interface with the `kalepy` package to draw 'samples' (discretized binaries) from the
distribution, and to calculate GW signatures.

The step of going from a number-density of binaries in $(M, q, z)$ space, to also the distribution
in $a$ or $f$ is subtle, as it requires modeling the binary evolution (i.e. hardening rate).



Finite Volume Cosmological (Hydrodynamic) Simulations
-----------------------------------------------------

Cosmological hydrodynamic simulations model the universe by co-evolving gas along with particles that
represent dark matter (DM), stars, and often BHs.  These simulations strive to model physical
processes at the most fundamental level allowed by resolution constraints / computational
limitations.  For example, BH accretion will typically be calculated by measuring the local density
(and thermal properties) of gas, which may also be subjected to 'feedback' processes from the
accreting BH itself, thereby producing a 'self-consistent' model.  However, no cosmological
simulations are able to fully resolve either the accretion or the feedback process, such that
'sub-grid models' (simplified prescriptions) must be adopted to model the physics at sub-resolution
scales.

In `holodeck`, MBH binary populations are derived from processed data files produced from cosmo-hydro
simulations.  To get to MBHBs, data must be provided either on the encounter ('merger')
rate of MBHs from the cosmological simulations directly, or based on the galaxy-galaxy encounters
and then prescribing MBH-MBH pairs onto those.  The initial binary populations must specify the binary
masses, their initial binary separation, and the redshift at which they formed (or are otherwise identified).

Note that the evolution of binaries, i.e. hardening from large separations to small separations and eventually coalescence, is treated separately (See :ref:`Binary Evolution/Hardening Models` below).

Implementation
^^^^^^^^^^^^^^
Full code documentation: :mod:`discrete populations <holodeck.population>` submodule.

This submodule provides a generalized base-class, :class:`_Population_Discrete <holodeck.population._Population_Discrete>`, that is subclassed
to implement populations from particular cosmological simulations.  At the time of this writing,
an Illustris-based implementation is included, :class:`Pop_Illustris <holodeck.population.Pop_Illustris>`.  Additionally, a set of
classes are also provided that can make 'modifications' to these populations based on subclasses of
the :class:`_Population_Modifier <holodeck.population._Population_Modifier>` base class.  Examples of currently implemented modifiers are:
adding eccentricity to otherwise circular binaries (:class:`PM_Eccentricity <holodeck.population.PM_Eccentricity>`), or changing the MBH
masses to match prescribed scaling relations (:class:`PM_Mass_Reset <holodeck.population.PM_Mass_Reset>`).

The fundamental, required attributes for all population classes are:
* `sepa` the initial binary separation in [cm].  This should be shaped as (N,) for N binaries.
* `mass` the mass of each component in the binary in [gram].  This should be shaped as (N, 2) for
  N binaries, and the two components of the binary.  The 0th index should refer to the more massive
  primary, while the 1th component refers to the less massive secondary.
* `scafa` the scale-factor defining the age of the universe for formation of this binary.  This
  should be shaped as (N,).

The implementation for binary evolution (e.g. environmental hardening processes), as a function of
separation or frequency, are included in the :mod:`holodeck.evolution` module, see also below.


Observational (AGN/Quasar) Catalogs
-----------------------------------

Implementation
^^^^^^^^^^^^^^
Full code documentation: :mod:`observational populations <holodeck.pop_observational>` submodule.


Binary Evolution/Hardening Models
=================================
Prescriptions for modeling binary evolution on less-than about kpc scales, with interactions between binaries are their galactic environments.


Observational Constraints on MBH and MBH-Binary Populations
===========================================================




References
==========

* [Sesana2008]_ Sesana, Veccio, & Colacino 2008.
* [Chen2019]_ Chen, Sesana, Conselice 2019.
