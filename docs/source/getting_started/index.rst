===============
Getting Started
===============

.. raw:: latex


.. contents:: File Contents
   :local:
   :depth: 1

**Quick-links:**

* :doc:`Definitions & Abbreviations <../defs_abbrevs>`

* :doc:`Bibliography <../biblio>`


**Overview:**

The ``holodeck`` framework simulates populations of MBH binaries, and calculates their GW signals.  In general, the calculation proceeds in three stages:

(1) :ref:`Populations`: Construct an initial population of MBH 'binaries'.  This is typically done for pairs of MBHs when their galaxies merge (i.e. long before the two MBHs are actually a gravitationally-bound binary).  The initial populations must specify, for each binary:

    (a) both MBH masses (typically as total-mass :math:`M` and mass-ratio :math:`q`),

    (b) the redshift (:math:`z`) at which the pair of MBHs form,

    (c) the initial separation (:math:`a_{init}`) of the MBHs at their formation time.

   Additional information can be very useful.  In particular, information about the host galaxy of the MBH pair can be used in the binary evolution calculation.

(2) :ref:`Binary Evolution`: Evolve the binary population from their initial conditions (i.e. large separations) until they reach the regime of interest (i.e. small separations).  In the simplest models, binaries are assumed to coalesce instantaneously, and are assumed to evolve purely due to GW emission.  Note that these two assumptions are contradictory.  More complex, self-consistent evolution models are recommended.  These models typically involve interactions between MBH binaries and their host galaxies ('environmental' interactions).

(3) :ref:`Gravitational Waves`: From the population of MBH binaries at the separations (or frequencies) of interst, calculate the resulting GW signals.


Populations
===========

'Continuous' Semi-Analytic-Model (SAM) Populations
--------------------------------------------------

The SAMs use simple, analytic components to calculate populations of binaries.  Holodeck calculates the number-density of MBH binaries, by calculating a number-density of galaxy-galaxy mergers, and then converting from galaxy properties to MBH properties by using an MBH-host relationship.

Galaxy Mergers
^^^^^^^^^^^^^^

``holodeck`` SAMs always start with a Galaxy Stellar-Mass Function (GSMF) that determines how many galaxies there are as a function of stellar mass, :math:`\psi(m_\star) \equiv \partial n_\star / \partial \log_{10} \! m_\star`, where :math:`n_\star` is the comoving number density of galaxies.  We then have to add a galaxy merger rate (GMR), :math:`R_\star(M_\star, q_\star) \equiv (1/n_\star) \partial^2 n_{\star\star} / \partial q_\star \, \partial t`, to find the number density of galaxy-pairs:

.. math::

   \frac{\partial^3 n_{\star\star}(M_\star, q_\star, z)}{\partial \log_{10} \! M_\star \, \partial q_\star \, \partial z}
   = \psi(m_{1,\star}) \, R_\star(M_\star, q_\star).

Here, :math:`M_\star = m_{1,\star} + m_{2,\star}` is the total stellar mass of both galaxies, and :math:`q_\star = m_{2,\star} / m_{1,\star} \leq 1` is the stellar mass ratio. Often in the literature, the GMR is estimated as a galaxy pair fraction (GPF; :math:`P_\star`) divided by a galaxy merger timescale (GMT; :math:`T_\star`), i.e. :math:`R_\star \approx P_\star / T_\star`.  The GPF is typically an observationally-derived component, defined roughly as, :math:`P_\star(m_{1,\star}, q_\star) \equiv N_{\star\star}(m_{1,\star}, q_\star) / N_\star(m_{1,\star})`, i.e. the number of galaxy pairs in a given survey divided by the number of all galaxies in the parent sample.  Note that there are significant selection effects in determing the number of galaxy pairs, including cuts on galaxy brightness/mass, and especially on the separations :math:`a_0` and :math:`a_1` between which pairs can be identified robustly.  The GMT is typically derived from numerical simulations, and defined roughly as, :math:`T_\star(M_\star, q_\star) \equiv \int_{a_0}^{a_1} \left[da/dt\right]^{-1}_{\star\star} da`, i.e. the total time that the galaxy pair spends at separations between :math:`a_0` and :math:`a_1`.



'Discrete' Illustris Populations
--------------------------------


Binary Evolution
================


Gravitational Waves
===================


.. References
.. ==========

.. * [BBR1980]_ Begelman, Blandford & Rees 1980.
.. * [Chen2019]_ Chen, Sesana, Conselice 2019.
.. * [Kelley2017a]_ Kelley, Blecha, and Hernquist (2017)
.. * [Sesana2008]_ Sesana, Veccio, & Colacino 2008.
