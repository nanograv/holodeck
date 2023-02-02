===============
Getting Started
===============

.. contents:: File Contents
   :local:

Overview
========

The `holodeck` package aims to simulate populations of MBH binaries, and calculate their GW signals.  In general the calculation proceeds in three stages:

(1) :ref:`Population <populations>`: Construct an initial population of MBH 'binaries'.  This is typically done for pairs of MBHs when their galaxies merge (i.e. long before the two MBHs are actually a gravitationally-bound binary).  Constructing the initial binary population may occur in a single step: e.g. gathering MBH-MBH encounters from cosmological hydrodynamic simulations; or it may occur over two steps: (i) gathering galaxy-galaxy encounters, and (ii) prescribing MBH properties for each galaxy.
(2) :ref:`Evolution <binary-evolution>`: Evolve the binary population from their initial conditions (i.e. large separations) until coalescence (i.e. small separations).  The complexity of this evolutionary stage can range tremendously in different implementations/models.  In the simplest models, binaries are assumed to coalesce instantaneously (in that the age of the universe is the same at formation and coalescence), and are assumed to evolve purely due to GW emission (in that the time spent in any range of orbital frequencies can be calculated from the GW hardening timescale).  Note that these two assumptions are contradictory.
(3) :ref:`Gravitational Waves <grav-waves>`: Calculate the resulting GW signals based on the binaries and their evolution.  Note that GWs can only be calculated based on some sort of model for binary evolution.  The model may be extremely simple, in which case it is sometimes glanced over.


Files in the Getting Started Guide
==================================

.. toctree::
   :maxdepth: 1

   Calculating Gravitational Waves <calc_gws>
   Definitions and Abbreviations <defs_abbrevs>
   Annotated Bibliography <biblio>


.. _grav-waves:

Gravitational Waves (GWs)
=========================

Calculating GW Signatures
-------------------------

For more details, see the document :doc:`Getting Started: Calculating Gravitational Waves <calc_gws>`.
For the implementation of much of the gravitational wave calculations, see the :mod:`GravWaves submodule <holodeck.gravwaves>`.

The chirp-mass is defined as: :math:`\mathcal{M} \equiv \frac{\left(m_1 m_2\right)^{3/5}}{M^{1/5}} = M \frac{q^{3/5}}{\left(1 + q\right)^{6/5}}`, for a total mass :math:`M = m_1 + m_2`, and mass-ratio :math:`q \equiv m_2 / m_1 \leq 1`.  NOTE: we define chirp-mass as an intrinsic property of the binary (i.e. it always refers to the rest-frame chirp-mass).  Chirp-mass can also be defined as the mass inferred based on a *measured* chirp of GW frequency, but we do not use this definition.

The 'hardening timescale' is defined as, :math:`\tau_f \equiv \frac{dt}{d\ln f_r} = \frac{f_r}{df_r/dt}`.

The hardening timescale due purely to GW emission, from a circular binary, is: :math:`\tau_\textrm{GW,circ}= \frac{5}{96}\left(\frac{G\mathcal{M}}{c^3}\right)^{-5/3} \left(2 \pi f_r\right)^{-8/3}`.

The GW strain amplitude produced by a circular binary is,

.. math::
   h_\textrm{s,circ}(f_r) = \frac{8}{10^{1/2}} \frac{\left(G\mathcal{M}\right)^{5/3}}{c^4 \, d_c}
           \left(2 \pi f_r\right)^{2/3},

Where :math:`d_c` is the comoving distance to the source.  NOTE: we always use 'chirp mass' to refer to the rest-frame chirp mass; if, instead, one uses the redshifted (i.e. measured) chirp mass, then the distance in the denominator should instead be the luminosity distance.

The characteristic strain (:math:`h_c`) of the GWB can be calculate from the distribution of number of binaries in the universe ($N$) per log-frequency interval as [Sesana2008]_:

.. math::
   h_c^2(f) = \int_0^\infty \!\! h_s^2(f_r) \, \frac{d^2N}{dz \, d\ln f_r} \bigg|_{f_r = f \cdot (1 + z)} \; dz,
   :name: eq:gwb_numtotal

This can also be written equivalently in terms of the comoving volumetric number-density of binaries in the universe (:math:`n_c \equiv dN/dV_c`) as [Phinney2001]_:

.. math::
   h_c^2(f) = \int_0^\infty h_s^2 \, \frac{dn_c}{dz} \, 4\pi c \, d_c^2 \cdot \left(1+z\right) \, \tau_f \; dz,
   :name: eq:gwb_numdens

where :math:`d_c` is the comoving distance to a source, related to luminosity distance as: :math:`d_L = d_c \, (1+z)`.  Comparing :math:numref:`eq:gwb_numdens` to :math:numref:`eq:gwb_numtotal`, we can identify the term,

.. math::
   \frac{d^2N}{dz \, d\ln f_r} = \frac{dn_c}{dz} \, 4\pi c \, d_c^2 \cdot \left(1+z\right) \, \tau_f.

For a finite volume, with a finite number of binaries, this can be discretized to:

.. math::
   h_c^2(f) = \sum_\textrm{redshift} \; \sum_\textrm{binaries} \; h_s^2 \; \frac{4\pi \, c \, d_c^2 \cdot \left(1 + z\right)}{V_\textrm{sim}} \; \tau_f.

To account for cosmic variance and the discreteness of binary sources, we can instead treat this as an expectation value and draw from a Poisson distribution (:math:`\mathcal{P}(x)`):

.. math::
    h_c^2(f) = & \sum_\textrm{redshift} \; \sum_\textrm{binaries} \; h_s^2 \cdot \frac{ \mathcal{P}(\Lambda \cdot \Delta \ln f_r) }{ \Delta \ln f_r }, \\
           \Lambda \equiv & \frac{4\pi \, c \, d_c^2 \cdot \left(1 + z\right)}{V_\textrm{sim}} \; \tau_f.

The interpretation for $\Lambda$, is the number of binaries in the observers past light-cone, per unit logarithmic frequency interval, that is expected for a given simulated binary, it's redshift, and it's rate of frequency evolution.

The observed GW frequencies :math:`f` are arbitrarily chosen.  Typically, for pulsar timing arrays, these are chosen based on Nyquist sampling for a given observational duration :math:`T \sim 15 \, \textrm{yr}` and cadence :math:`\Delta t \sim 2 \, \textrm{week}`, such that :math:`f = \left[1/T, 2/T, 3/T, \, \ldots \, , 1/\left(2 \Delta t\right)\right]`.


GW Detection with Pulsar Timing Arrays (PTAs)
---------------------------------------------
Pulsars are rapidly spinning neutron stars which produce beams of radio emission that periodically intersect the observer's line of sight.  Millisecond pulsars, in particular, often maintain incredibly precise periodicities with precision down to the order of 1s to 100s of nanoseconds.  The presence of GWs induce variations to the times of arrival (TOAs) measured by observatories on Earth.

Many noise sources (e.g. due to the neutron star, the source of radio emission, or the radio observatory) can also produce TOA variations.  Unlike noise sources, GWs are believed to produce a very unique spatial correlation pattern in the TOA variations across the sky.  In particular, the angular correlation function of TOA variations between pairs of pulsars produces a pseudo-quadrupolar pattern as a function of angular separation which is believed to be a 'smoking gun' signatures of GWs (i.e. a signature that is not produced by any noise source).  The identification of this angular correlation function can thus be used to detect gravitational waves.

The frequency range that PTAs are sensitive to is determined primarily by the ability to reconstruct periodic signals from the time-series data of pulsar TOAs.  The lowest sensitive frequency is determined by the Nyquist frequency, i.e. the inverse of the total observing duration.  High-precision TOA measurements have been performed for about two decades, meaning that PTAs are starting to probe frequencies as low as nanohertz (nHz).  The highest sensitive frequency is determined by the typically interval between observations, which tends to be on the order of a few weeks, or hundreds of nanohertz.

While there have been a large number of proposed sources of GWs in the nanohertz regime, the best studied sources are binaries of supermassive black holes, with total masses :math:`M \gtrsim 10^8 \, M_\odot`.





.. _populations:

Methods for Simulating MBH Binary Populations
=============================================
Simulations of MBH Binary (MBHB) populations require three components:

   (1) **Origins**: the events which produce encounters between pairs (or more) of MBHs, which are typically galaxy-galaxy mergers (which occur on large scales of :math:`\mathrm{kpc} - \mathrm{Mpc}`);
   (2) **Populations**: the properties of the MBHs (and often their host galaxies) involved in the encounter events (at least a redshift and pair of masses); and
   (3) **Evolution**: the process by which the two (or more) MBHs are able to reach the small binary separations at which detectable GWs are produced.

Components (1) and (2) can be produced in a variety of ways described below and, indeed, in numerous combinations of these methods.

   (A) :ref:`Cosmological hydrodynamic simulations <pop-discrete>` strive to model the fundamental processes underlying galaxy and star formation to produce populations of galaxies, including galaxy mergers, and often massive black holes.
   (B) :ref:`Semi-analytic models (SAMs) <pop-sams>` use relatively simple analytic prescriptions for properties of galaxy and MBHs, typically calibrated to observations, to produce initial populations of binary MBHs.
   (C) `Semi-empirical models (SEMs)` [*Not currently implemented in Holodeck*] are similar to SAMs, but rely more strongly on observational relationships, and are typically much more complex/comprehensive, often including models for not only galaxy populations but even the internal structure of galaxies and their co-evolution with MBHs.  In this way, SEMs are somewhere in-between SAMs and hydrodynamic simulations, but rely heavily on observations of populations of galaxies and MBHs/AGNs.
   (D) :ref:`Observational catalogs <pop-obs>` can also be used to construct binary population more directly, e.g. by starting from  observed AGN or quasars, and prescribing the occurrence rates of binaries.  These observational catalogs will typically use components of SAMs, SEMs, or hydrodynamic simulations to complement the directly observed properties of systems.

Component (3) always requires the same type of implementation: some level of semi-analytic modeling of the physical interactions between MBHBs and the environments of their surrounding galactic nuclei.  The physical processes which mediate *"binary hardening"*, the process of extracting energy and angular momentum to allow the binary orbit to shrink, cannot be resolved directly in either cosmological hydrodynamic simulations or observations which are currently both limited to physical scales :math:`r \gtrsim 100 \, \mathrm{pc}` where the most massive MBHBs are only just starting to gravitationally interact with each other.





.. _pop-sams:

Semi-Analytic Models
--------------------

.. _pop-sams-theory:

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


.. _pop-sams-implement:

SAM Implementation
^^^^^^^^^^^^^^^^^^
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
    Implemented as subclasses of the :class:`holodeck.relations._MMBulge_Relation` subclass.

The :class:`Semi_Analytic_Model <holodeck.sam.Semi_Analytic_Model>` class defines a grid in parameter space of total MBH mass ($M=M_1 + M_2$),
MBH mass ratio ($q \\equiv M_1/M_2$), redshift ($z$), and at times binary separation
(semi-major axis $a$) or binary rest-frame frequency ($f_r$).  Over this grid, the distribution of
comoving number-density of MBH binaries in the Universe is calculated.  Methods are also provided
that interface with the `kalepy` package to draw 'samples' (discretized binaries) from the
distribution, and to calculate GW signatures.

To calculate the distribution of binaries, `Semi_Analytic_Model` first calculates a 'static volume-density' of binaries, as a differential distribution over the intrinsic parameters $M$, $q$, $z$.  This is the quantity,

.. math::
   \frac{d^3 n}{d\log_{10}M \, dq \, dz} = \frac{dt}{dz} \frac{\Phi(M,z) \, P(M,q,z)}{T(M,q,z)} \, J_M.

The $J_M$ term includes conversions between different masses and mass ratios, namely,

.. math::
   J_M = \frac{d \log_{10} m_{\star,1}}{dM_{\star}} \, \frac{dq_\star}{dq} \, \frac{dM_\star}{dM} M \log(10).

This is returned by the :meth:`Semi_Analytic_Model.static_binary_density <holodeck.sam.Semi_Analytic_Model.static_binary_density>` method, and has units of $[\\mathrm{Mpc}^{-3}]$.

The next step converts from a static volume-density of binaries in $(M, q, z)$ space, to also the distribution
in $a$ or $f$.  This step is subtle, as it requires modeling the binary evolution (i.e. hardening rate; see :ref:`Binary Evolution <binary-evolution>`).  This is performed in the :meth:`Semi_Analytic_Model.dynamic_binary_number <holodeck.sam.Semi_Analytic_Model.dynamic_binary_number>` method, which also converted from volume-density to number of binaries (i.e. $dn \\rightarrow dN$).  The result is the quantity,

.. math::
   \frac{d^4 N}{d\log_{10}M \, dq \, dz \, d\ln X} = \frac{d^3 n}{d\log_{10}M \, dq \, dz} \frac{dz}{dt} \frac{dV_c}{dz} t_X,
   :name: eq:N_from_n

where $X$ can either be the binary separation $a$ or the binary GW frequency (?!GW OR ORBITAL?!) $f_r$.  The timescale is correspondingly defined in terms of either of these variables, i.e. $t_a = dt / d \ln f_r$ or $t_f = df_r / d \ln f_r$.  The returned value is dimensionless (i.e. a number).  Note that the universe light-cone term can be expressed as,

.. math::
   \frac{dV_c}{dt} = \frac{dV_c}{dz} \frac{dz}{dt} = 4 \pi \, c \, (1 + z) \, d_c^2,

where $d_c$ is the comoving distance to the corresponding age (or redshift or distance) of the Universe.

Gravitational Wave signals can then be calculated in two ways.  The smooth grid of differential number of binaries ($d^4 N / d\log_{10}M \, dq \, dz \, d\ln f_r$) can be used, which is done in the :meth:`Semi_Analytic_Model.gwb <holodeck.sam.Semi_Analytic_Model.gwb>` method.  Or the smooth distribution of binaries can be discretely sampled into a finite population of binaries in the universe, from which the total GW strain can be added up, which is done within the :meth:`sam.sampled_gws_from_sam <holodeck.sam.sampled_gws_from_sam>` method.

DESCRIBE GW CALCULATION MORE HERE.





.. _pop-discrete:

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

Note that the evolution of binaries, i.e. hardening from large separations to small separations and eventually coalescence, is treated separately (See `Binary Evolution`_ below).


Finite Volume / Discrete Implementation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
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
separation or frequency, are included in the :mod:`holodeck.evolution` module, see also below.  The result is an 'evolutionary track' or 'evolutionary history' for each binary in the finite volume of the cosmological simulation.

At times, instead of the finite volume, we wish to calculate the distribution of binaries we would expect in a full universe.  To either sample such a population of binaries in a full universe, or to calculate the GW signals produced by such a population, we must extrapolate to the larger volume, taking into account both the size of the cosmological simulation and the rate of evolution of each binary.  The rate of evolution of the binaries is needed because distance/volume is connected to time by the observer's past light-cone.  i.e. something observed a given distance away, is also observed the corresponding light-travel time in the past.  This is treated as follows.

First we choose an independent variable of interest to define where we sample our population.  Because GW signals are our primary interest, it is typically natural to pre-define a grid of observer-frame GW frequencies.  Equivalently binary separations can be defined, or redshift intervals, etc, but we typically use observer-frame GW frequencies.  Each binary evolution track is interpolated to find the point in time when the binary reaches each frequency of interest.  For each frequency, and each binary, we then have the binary intrinsic properties (e.g. $M, q, z$) in additional to its hardening rate ($da/dt$ or $df/dt$).  If the corresponding redshift (or scale-factor or age of the Universe) takes place after the current epoch (i.e. $z=0$), then the binary can be discarded at that frequency.

Each binary, at each frequency, is then interpreted as contributing a volume-density of $n_{ij} = 1/V_\mathrm{sim}$ to the total distribution of binaries.  We can then convert from volume-density to number of binaries in the same way as for the semi-analytic-model calculation :math:numref:`eq:N_from_n`,

.. math::
   \frac{d N}{d\ln X} = n \frac{dV_c}{dz} \frac{dz}{dt} \frac{d t}{d\ln X},

where $X$ can either be the binary separation $a$ or the binary GW frequency (?!GW OR ORBITAL?!) $f_r$.  The timescale is correspondingly defined in terms of either of these variables, i.e. $t_a = dt / d \ln f_r$ or $t_f = df_r / d \ln f_r$.  The returned value is dimensionless (i.e. a number).  Note that the universe light-cone term can be expressed as,

.. math::
   \frac{dV_c}{dt} = \frac{dV_c}{dz} \frac{dz}{dt} = 4 \pi \, c \, (1 + z) \, d_c^2,

where $d_c$ is the comoving distance to the corresponding age (or redshift or distance) of the Universe.




.. _pop-obs:

Observational (AGN/Quasar) Catalogs
-----------------------------------

Observational Implementation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Full code documentation: :mod:`observational populations <holodeck.pop_observational>` submodule.



.. _binary-evolution:

Binary Evolution
================

In `holodeck`, initial binary populations are typically defined near the time of galaxy-galaxy
merger, when two MBHs come together at roughly kiloparsec scales.  Environmental 'hardening'
mechanisms are required to dissipate orbital energy and angular momentum, allowing the binary
separation to shrink ('harden'). Typically *dynamical friction (DF)* is most important at large scales
($\sim \mathrm{kpc}$).  Near where the pair of MBHs become a gravitationally-bound binary, the
DF approximations break down, and individual *stellar scattering* events (from stars in the 'loss
cone' of parameter space) need to be considered.  In the presence of significant gas (i.e. from
accretion), a circumbinary accretion disk (CBD) may form, and gravitational *circumbinary disk torques*
from the gas distribution (typically spiral waves) may become important.  Finally, at the smaller separations,
*GW emission* takes over.  The classic work describing the overall process of binary evolution in
its different stages is [BBR1980]_.  [Kelley2017a]_ goes into significant detail on implementations
and properties of each component of binary evolution also.  Triple MBH interactions, and perturbations
from other massive objects (e.g. molecular clouds) can also be important.

The :mod:`holodeck.evolution` submodule provides tools for modeling the binary evolution from the
time of binary 'formation' (i.e. galaxy merger) until coalescence.  Models for binary evolutionary
can range tremendously in complexity.  In the simplest models, binaries are assumed to coalesce
instantaneously (in that the age of the universe is the same at formation and coalescence), and are
also assumed to evolve purely due to GW emission (in that the time spent in any range of orbital
frequencies can be calculated from the GW hardening timescale).  Note that these two assumptions
are contradictory, but commonly employed in the literature.  The ideal, self-consistent approach,
is to model binary evolution using self-consistently derived environments (e.g. gas and stellar
mass distributions), and applying the same time-evolution prescription to both the redshift
evolution of each binary, and also the GW calculation.  Note that GWs can only be calculated based
on some sort of model for binary evolution.  The model may be extremely simple, in which case it is
sometimes glanced over.


Evolution Implementation
------------------------
Full code documentation: :mod:`evolution submodule <holodeck.evolution>` submodule.

The core component of the evolution module is the :class:`Evolution <holodeck.evolution.Evolution>` class.  This class combines a population of initial binary parameters (i.e. from the :class:`_Population_Discrete <holodeck.population._Population_Discrete>` class), along with a specific binary hardening model (:class:`_Hardening` subclass), and performs the numerical integration of each binary over time - from large separations to coalescence.
The
:class:`Evolution` class also acts to store the binary evolution histories ('trajectories' or
'tracks'), which are then used to calculate GW signals (e.g. the GWB).  To facilitate GW and
similar calculations, :class:`Evolution` also provides interpolation functionality along binary
trajectories.

Integration of the binary evolution takes place starting from a regular range/grid of binary separations that are chosen for each binary being evolved.  **The separation (`sepa`) thus acts as the independent variable**.  The integration calculates the time it takes for each step to complete.  This is unlike most types of dynamical integration in which there is a prescribed time-step, and the amount of distance (etc) traveled over that time is then calculated.

The :class:`Evolution <holodeck.evolution.Evolution>` class is (currently) primary built to be used with a
:class:`Pop_Illustris <holodeck.population.Pop_Illustris>` instance.
More generally, the `Evolution` class is instantiated with a :class:`_Population_Discrete <holodeck.population._Population_Discrete>`
instance, and a particular binary hardening model (one or more subclass of :class:`_Hardening <holodeck.evolution._Hardening>`).  It
then numerically integrates each binary from their
initial separation to coalescence, determining how much time that process takes, and thus the
rate of redshift/time evolution.

**Initialization**: the evolution constructor sets all attributes to empty arrays of the appropriate sizes.
NOTE: the 0th step is *not* initialized at this time, instead it happens in :meth:`Evolution.evolve()` method.

**Integration**: binary evolution is performed by running the
:meth:`Evolution.evolve <holodeck.evolution.Evolution.evolve>` method.  The process is as follows:

(1) Step zero is initialized using the
    :meth:`Evolution._init_step_zero <holodeck.evolution.Evolution._init_step_zero>` method.  This transfers
    attributes from the stored :class:`_Population_Discrete <holodeck.population._Population_Discrete>` instance to the 0th index of the
    evolution arrays.  The attributes are [`sepa`, `scafa`, `mass`, and optionally `eccen`].  The hardening model
    is also used to calculate the 0th hardening rates [`dadt`, `dedt`].
(2) Integration is performed over all subsequent steps, by progressively calling :meth:`Evolution._take_next_step <holodeck.evolution.Evolution._take_next_step>`.  For an integration step `s`, we move from index `s-1` to index
    `s`.  These correspond to the 'left' and 'right' edges of the step.  The evolutionary trajectory values have
    already been calculated on the left edges (during either the previous time step, or the initial time step).
    The numerical integration proceeds as:

        (a) The hardening rate is calculated at the right edge of the step.
        (b) The time it takes to move from the left to right edge is calculated using a trapezoid
            rule in log-log space.
        (c) The right edge evolution values are stored and updated.

(3) Integration is completed once all steps have been taken, and the :meth:`Evolution._finalize <holodeck.evolution.Evolution._finalize>` method is called.  Any stored modifiers (:class:`holodeck.utils._Modifier` subclasses) are applied using the :meth:`Evolution._modify <holodeck.evolution.Evolution._modify>` method, and any diagnostic checks are run with the :meth:`Evolution._check <holodeck.evolution.Evolution._check>` method.


Binary Hardening Models
-----------------------
A large number of physical processes serve to extract energy and angular momentum from MBH binaries, making their binary separation (i.e. semi-major axis) decrease.  This is called 'hardening'.  In `holodeck` hardening is typically parametrized in terms of a rate of change of the binary semi-major axis $a$, i.e. $da/dt$. It can just as easily be defined with respect to frequency, $df/dt$.  During hardening processes, the eccentricity ($e$) of binaries can also change, parametrized with $de/dt$.  From a theoretical perspective, the eccentricity evolution is even more uncertain than the semi-major axis evolution, and even fewer prescriptions exist for its modeling.  The definitive, early reference for MBH binary hardening (evolution more generally), is [BBR1980]_.

Presentation of example hardening models and their usage can be found in the :mod:`discrete hardening models notebook <holodeck.notebooks.discrete_hardening_models>`.


Hardening Implementation
^^^^^^^^^^^^^^^^^^^^^^^^
Full code documentation: :mod:`evolution submodule <holodeck.evolution>` submodule, and :mod:`evolution submodule <holodeck.hardening>` submodule.

Hardening models, i.e. methods of calculating $da/dt$ and $de/dt$, are implemented as subclasses of :class:`holodeck.hardening._Hardening`.  Each one, or a combination of them, are passed to the constructor of an :class:`Evolution <holodeck.evolution.Evolution>` instance, and they are used when calling :meth:`Evolution.evolve <holodeck.evolution.Evolution.evolve>`.  **Currently, hardening rates are combined by adding the $da/dt$ and $de/dt$ values directly.  NOTE that this is an approximation, and is not necessarily the correct way to calculate the cumulative hardening rate.**  A more accurate approach would be to calculate (or estimate) the rate of change of the binaries' energy and angular momentum, sum these rates, and derive a corresponding rate of change of semi-major axis and eccentricity.

The only API method that individual hardening models (`_Hardening` subclasses) must implement are the :meth:`_Hardening.dadt_dedt(self, evo, step, *args, **kwargs) <holodeck.hardening._Hardening.dadt_dedt>` method.  The first two arguments are the `Evolution` instance (this is passed in as `self` indirectly during calls to  `Evolution.evolve`), and the `step` number - which of the finite number of evolutionary 'steps' is currently being evaluated.  Each evolution 'step' corresponds to moving from a 'before' state to an 'after' state, such that `step` is the destination: the 'after' state.  Binary parameters are loaded from the 'before' state (e.g. universe scale-factor as `evo.scafa[:, step-1]`), and the values at the 'after' state are calculated and stored (e.g. at `evo.scafa[:, step]`).  `_Hardening.dadt_dedt` then returns the semi-major axis and eccentricity evolution rates, ``dadt, dedt``.  By convention, when eccentricity is not being evolved (i.e. ``evo.eccen is None``) then ``dedt = None`` should be returned.  When eccentricity is being evolved, if the given hardening model does not evolve eccentricity, then `dedt` should be an `np.ndarray` of all zeros, matching the shape of `dadt`.  It is preferable, although not required by the API, for there to be an internal method `_dadt_dedt` which accepts the necessary physical parameters at a given point in time, and returns the derived ``dadt, dedt`` values (e.g. `_dadt_dedt(self, mtot, mrat, sepa)`).  This is useful for testing, plotting, and other diagnostics; and may be required in the future.

Recall that upon initialization of the `Evolution` instance, a fixed range/grid of binary separations is chosen for each binary (in log-space between the initial binary separation, and the coalescence radius).  From :meth:`Evolution.evolve <holodeck.evolution.Evolution.evolve>`, once the cumulate evolution rates ($da/dt$, $de/dt$) are calculated from all stored `_Hardening` instances, then the time it takes to move one step of binary separation is calculated.  This time is then converted to a change in scale-factor of the universe, and the `Evolution.scafa` attribute is updated for the next step.

Some of the implemented hardening models are described below:

*   :class:`Hard_GW <holodeck.hardening.Hard_GW>` : gravitational-wave driven evolution.
*   :class:`Dynamical_Friction_NFW <holodeck.hardening.Dynamical_Friction_NFW>` : dynamical friction based on a simple semi-analytic model assuming an NFW dark matter halo, and Dehnen stellar density profile.
*   :class:`Sesana_Scattering <holodeck.hardening.Sesana_Scattering>` : stellar 'loss-cone' scattering model.  This is implemented assuming a Dehnen stellar density profile, and the prescription of [Quinlan1996]_, with the numerically calculated coefficients and fit-functions from [Sesana2006]_.
*   :class:`Fixed_Time <holodeck.hardening.Fixed_Time>` : a phenomenological hardening model with 4 parameters: two power-laws (defined in hardening time, $a / (da/dt)$), $\gamma_1$ and $\gamma_2$; a characteristic transition radius between the two, $r_c$; and an overall hardening rate normalization $A$.  The `Fixed_Time` model includes GW hardening.



Observational Constraints on MBH and MBH-Binary Populations
===========================================================






References
==========

* [BBR1980]_ Begelman, Blandford & Rees 1980.
* [Chen2019]_ Chen, Sesana, Conselice 2019.
* [Kelley2017a]_ Kelley, Blecha, and Hernquist (2017)
* [Sesana2008]_ Sesana, Veccio, & Colacino 2008.
