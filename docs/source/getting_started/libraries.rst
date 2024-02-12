=======================================
Generating and Using Holodeck Libraries
=======================================

.. contents:: File Contents
   :local:
   :depth: 1

**Quick-links:**

* :doc:`Definitions & Abbreviations <../defs_abbrevs>`

* :doc:`Bibliography <../biblio>`

Holodeck 'libraries' are collections of simulations run from the same parameter-space and using the same hyper parameters.  Libraries are constructed using the :mod:`librarian` holodeck module, with a 'parameter space' class that organizes the different simulations.  The base-class is called :class:`_Param_Space`, and all parameter space classes inherit from this, and should typically be prefixed by `PS_` to denote that they are parameter spaces.  For example, the fiducial library and parameter space for :doc:`the 15yr astrophysics analysis <nanograv_15yr>` was the 'phenomenological uniform' library, implemented as :class:`PS_Classic_Phenom_Uniform` (at the time, it was internally called `PS_Uniform_09B`).  This library spanned a 6D parameter space using a 'phenomenological' binary evolution model, and assuming a uniform distribution in sampling from the parameter priors.  Two parameters from the galaxy stellar-mass function were varied, along with two parameters from the M-MBulge relationship, and two parameters from the hardening model.
