===============
Getting Started
===============

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

Semi-Analytic Models (SAMs)
---------------------------

``holodeck`` SAMs are handled in the :doc:`holodeck.sams <../apidoc_modules/holodeck.sams>` module.  The core of the module is the :class:`~holodeck.sams.sam.Semi_Analytic_Model` class, in the: :doc:`holodeck.sams.sam <../apidoc_modules/holodeck.sams.sam>` submodule.

The SAMs use simple, analytic components to calculate populations of binaries.  Holodeck calculates the number-density of MBH binaries, by calculating a number-density of galaxy-galaxy mergers, and then converting from galaxy properties to MBH properties by using an MBH-host relationship.

The SAMs are initialized over a 3-dimensional parameter space of total MBH mass (:math:`M = m_1 + m_2`), MBH mass ratio (:math:`q = m_2 / m_1 \leq 1`), and redshift (:math:`z`).  The ``holodeck`` code typically refers to the number of bins in each of these dimensions as ``M``, ``Q``, and ``Z``; for example, the shape of the number-density of galaxy mergers will be ``(M, Q, Z)``.  Most calculations retrieve the number of binaries in the Universe at a given set of frequencies (or sometimes binary separations), so the returned values will be 4-dimensional with an additional axis with ``F`` frequency bins added.  For example, the number of binaries at a given set of frequencies will typically be arrays of shape ``(M, Q, Z, F)``.

Galaxy Mergers
^^^^^^^^^^^^^^

``holodeck`` SAMs always start with a Galaxy Stellar-Mass Function (GSMF) that determines how many galaxies there are as a function of stellar mass, :math:`\psi(m_\star) \equiv \partial n_\star / \partial \log_{10} \! m_\star`, where :math:`n_\star` is the comoving number density of galaxies.  We then have to add a galaxy merger rate (GMR), :math:`R_\star(M_\star, q_\star) \equiv (1/n_\star) \partial^2 n_{\star\star} / \partial q_\star \, \partial t`, to find the number density of galaxy-pairs:

.. math::

   \frac{\partial^3 n_{\star\star}(M_\star, q_\star, z)}{\partial \log_{10} \! M_\star \, \partial q_\star \, \partial z}
   = \psi(m_{1,\star}) \, R_\star(M_\star, q_\star).

Here, :math:`M_\star = m_{1,\star} + m_{2,\star}` is the total stellar mass of both galaxies, and :math:`q_\star = m_{2,\star} / m_{1,\star} \leq 1` is the stellar mass ratio. Often in the literature, the GMR is estimated as a galaxy pair fraction (GPF; :math:`P_\star`) divided by a galaxy merger timescale (GMT; :math:`T_\star`), i.e. :math:`R_\star \approx P_\star / T_\star`.  The GPF is typically an observationally-derived component, defined roughly as, :math:`P_\star(m_{1,\star}, q_\star) \equiv N_{\star\star}(m_{1,\star}, q_\star) / N_\star(m_{1,\star})`, i.e. the number of galaxy pairs in a given survey divided by the number of all galaxies in the parent sample.  Note that there are significant selection effects in determining the number of galaxy pairs, including cuts on galaxy brightness/mass, and especially on the separations :math:`a_0` and :math:`a_1` between which pairs can be identified robustly.  The GMT is typically derived from numerical simulations, and defined roughly as, :math:`T_\star(M_\star, q_\star) \equiv \int_{a_0}^{a_1} \left[da/dt\right]^{-1}_{\star\star} da`, i.e. the total time that the galaxy pair spends at separations between :math:`a_0` and :math:`a_1`.  So we can also write:

.. math::

   \frac{\partial^3 n_{\star\star}(M_\star, q_\star, z)}{\partial \log_{10} \! M_\star \, \partial q_\star \, \partial z}
   = \psi(m_{1,\star}) \, \frac{P_\star(m_{1,\star}, q_\star)}{T_\star(M_\star, q_\star)}.


MBH Populations and MBH-Host Relations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We now have a galaxy-galaxy merger rate, and we need to populate these galaxies with MBHs.  To do this, we need an MBH-host relationship, typically in the form of M-MBulge (:math:`m_\textrm{BH} = M_\mu(m_\textrm{bulge}, z)`; mass of the MBH, relative to the stellar-bulge mass of the host galaxy), and possibly a relationship between bulge mass and overall stellar-mass (i.e. :math:`m_\textrm{bulge} = m_\textrm{bulge}(m_\star)`).  Given this relationship, we can convert to MBH mergers as,

.. math::

   \frac{\partial^3 n(M, q, z)}{\partial \log_{10} \! M \, \partial q \, \partial z}
   = \frac{\partial^3 n_{\star\star}(M_\star, q_\star, z)}{\partial \log_{10} \! M_\star \, \partial q_\star \, \partial z}
      \left[\frac{\partial M_\star}{\partial M}\right] \left[\frac{\partial q_\star}{\partial q} \right],

where the masses must be evaluated at the appropriate locations: :math:`m_1 = M_\mu(m_{1,\star}) \, \& \, m_2 = M_\mu(m_{2,\star})`.


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
