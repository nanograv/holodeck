===============
Getting Started
===============

.. contents:: File Contents
   :local:

Overview
========

The `holodeck` package aims to simulate populations of MBH binaries, and calculate their GW signals.  In general, the calculation proceeds in three stages:

(1) :ref:`Population <populations>`: Construct an initial population of MBH 'binaries'.  This is typically done for pairs of MBHs when their galaxies merge (i.e. long before the two MBHs are actually a gravitationally-bound binary).  Constructing the initial binary population may occur in a single step: e.g. gathering MBH-MBH encounters from cosmological hydrodynamic simulations; or it may occur over two steps: (i) gathering galaxy-galaxy encounters, and (ii) prescribing MBH properties for each galaxy.
(2) :ref:`Evolution <binary-evolution>`: Evolve the binary population from their initial conditions (i.e. large separations) until coalescence (i.e. small separations).  The complexity of this evolutionary stage can range tremendously in different implementations/models.  In the simplest models, binaries are assumed to coalesce instantaneously (in that the age of the universe is the same at formation and coalescence), and are assumed to evolve purely due to GW emission (in that the time spent in any range of orbital frequencies can be calculated from the GW hardening timescale).  Note that these two assumptions are contradictory.
(3) :ref:`Gravitational Waves <grav-waves>`: Calculate the resulting GW signals based on the binaries and their evolution.  Note that GWs can only be calculated based on some sort of model for binary evolution.  The model may be extremely simple, in which case it is sometimes glanced over.


'Continuous' Semi-Analytic-Model (SAM) Populations
==================================================


'Discrete' Illustris Populations
================================


References
==========

.. * [BBR1980]_ Begelman, Blandford & Rees 1980.
.. * [Chen2019]_ Chen, Sesana, Conselice 2019.
.. * [Kelley2017a]_ Kelley, Blecha, and Hernquist (2017)
.. * [Sesana2008]_ Sesana, Veccio, & Colacino 2008.
