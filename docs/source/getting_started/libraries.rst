=======================================
Generating and Using Holodeck Libraries
=======================================

.. include:: ../header.rst

.. contents:: File Contents
   :local:
   :depth: 1


Overview
========

|holodeck| 'libraries' are collections of simulations run from the same parameter-space and using the same hyper parameters.  Libraries are constructed using the :py:mod:`~holodeck.librarian` module, with a 'parameter space' class that organizes the different simulations.  The base-class is called :py:class:`~holodeck.librarian.params._Param_Space` (defined in the :py:mod:`holodeck.librarian.params` file), and all parameter space classes inherit from this, and should typically be prefixed by ``PS_`` to denote that they are parameter spaces.  The parameter-space subclasses implement a number of parameters that are varied.  Each parameter is implemented as a subclass of :py:class:`~holodeck.librarian.params._Param_Dist`, for example the :py:class:`~holodeck.librarian.params.PD_Uniform` class that implements a uniform distribution.

As an example, the fiducial library and parameter space for :doc:`the 15yr astrophysics analysis <nanograv_15yr>` was the 'phenomenological uniform' library, implemented as :py:class:`~holodeck.librarian.param_spaces_classic.PS_Classic_Phenom_Uniform` (at the time, it was internally called ``PS_Uniform_09B``).  This library spanned a 6D parameter space using a 'phenomenological' binary evolution model, and assuming a uniform distribution in sampling from the parameter priors.  Two parameters from the galaxy stellar-mass function were varied, along with two parameters from the M-MBulge relationship, and two parameters from the hardening model.


Parameter Spaces
================
**NOTE: currently parameter-spaces are only designed for use with SAMs.**

Parameter spaces must subclass :py:class:`~holodeck.librarian.params._Param_Space`, and provide 4 elements:

(1) A class attribute called ``DEFAULTS`` which is a ``dict`` of default parameter values for all of the parameters needed by the initialization methods.

(2) An ``_init_sam()`` function that is a ``classmethod``, which takes the input parameters, and then constructs and returns a |sam_class| instance.

(2) An ``_init_hard()`` function that is a ``classmethod``, which takes the input parameters, and then constructs and returns a |hard_class| instance.

(4) An ``__init__()`` method that passes all required parameter distributions (:py:class:`~holodeck.librarian.params._Param_Dist` subclasses) to the super-class ``__init__()`` method.