=======================================
Generating and Using Holodeck Libraries
=======================================

.. include:: ../header.rst

.. contents:: File Contents
   :local:
   :depth: 1


Overview
========

|holodeck| 'libraries' are collections of simulations in which a certain set of parameters are varied, producing different populations and/or GW signatures at each sampled parameter value.  Libraries are run from the same parameter-space and using the same hyper parameters (for example, the functional form that is assumed for the galaxy stellar-mass function).  Libraries are constructed using the :py:mod:`~holodeck.librarian` module, with a 'parameter space' class that organizes the different simulations.  The base-class is called :py:class:`~holodeck.librarian.libraries._Param_Space` (defined in the :py:mod:`holodeck.librarian.libraries` file), and all parameter space classes inherit from this, and should typically be prefixed by ``PS_`` to denote that they are parameter spaces.  The parameter-space subclasses implement a number of parameters that are varied.  Each parameter is implemented as a subclass of :py:class:`~holodeck.librarian.libraries._Param_Dist`, for example the :py:class:`~holodeck.librarian.libraries.PD_Uniform` class that implements a uniform distribution.

As an example, the fiducial library and parameter space for :doc:`the 15yr astrophysics analysis <nanograv_15yr>` was the 'phenomenological uniform' library, implemented as :py:class:`~holodeck.librarian.param_spaces_classic.PS_Classic_Phenom_Uniform` (at the time, it was internally called ``PS_Uniform_09B``).  This library spanned a 6D parameter space using a 'phenomenological' binary evolution model, and assuming a uniform distribution in sampling from the parameter priors.  Two parameters from the galaxy stellar-mass function were varied, along with two parameters from the M-MBulge relationship, and two parameters from the hardening model.


Parameter Spaces
================
**NOTE: currently parameter-spaces are only designed for use with SAMs.**

Parameter spaces are implemented as subclasses of the |pspace_class| class.  The class generates a certain number of samples using a latin hypercube to efficiently sample the parameter space.  Each parameter being varied in the parameter space corresponds to parameter distribution, implemented as a :class:`~holodeck.librarian.libraries._Param_Dist` subclass.  These parameter distributions convert from uniform random variables (samples in the latin hypercube) to the desired distributions.  For example, the :class:`~holodeck.librarian.libraries.PD_Normal(mean, stdev)` class draws from a normal (Gaussian) distribution, and the :class:`~holodeck.librarian.libraries.PD_Normal(min, max)` class draws from a uniform distribution.

Parameter spaces must subclass |pspace_class|, and provide 4 elements:

(0) OPTIONAL/Recommended: A class attribute called ``DEFAULTS`` which is a ``dict`` of default parameter values for all of the parameters needed by the initialization methods.  **This is strongly recommended to ensure that parameters are set consistently, by setting them explicitly.**

(2) An ``_init_sam()`` a function that takes the input parameters, and then constructs and returns a |sam_class| instance.

(2) An ``_init_hard()`` a function that takes the input parameters, and then constructs and returns a |hard_class| instance.

(4) An ``__init__()`` method that passes all required parameter distributions (:py:class:`~holodeck.librarian.libraries._Param_Dist` subclasses) to the super-class ``__init__()`` method.

Public parameter spaces should also be 'registered' to the :data:`holodeck.librarian.param_spaces_dict` dictionary.  See :mod:`holodeck.librarian`.


Generating Libraries
====================