=======================================
Generating and Using Holodeck Libraries
=======================================

.. include:: ../header.rst

.. contents:: File Contents
   :local:
   :depth: 1


Libraries Overview
==================

|holodeck| 'libraries' are collections of simulations in which a certain set of parameters are varied, producing different populations and/or GW signatures at each sampled parameter value.  Libraries are run from the same parameter-space and using the same hyper parameters (for example, the functional form that is assumed for the galaxy stellar-mass function).  Libraries are constructed using the :py:mod:`~holodeck.librarian` module, with a 'parameter space' class that organizes the different simulations.  The base-class is called :py:class:`~holodeck.librarian.libraries._Param_Space` (defined in the :py:mod:`holodeck.librarian.libraries` file), and all parameter space classes inherit from this, and should typically be prefixed by ``PS_`` to denote that they are parameter spaces.  The parameter-space subclasses implement a number of parameters that are varied.  Each parameter is implemented as a subclass of :py:class:`~holodeck.librarian.libraries._Param_Dist`, for example the :py:class:`~holodeck.librarian.libraries.PD_Uniform` class that implements a uniform distribution.

As an example, the fiducial library and parameter space for :doc:`the 15yr astrophysics analysis <nanograv_15yr>` was the 'phenomenological uniform' library, implemented as :py:class:`~holodeck.librarian.param_spaces_classic.PS_Classic_Phenom_Uniform` (at the time, it was internally called ``PS_Uniform_09B``).  This library spanned a 6D parameter space using a 'phenomenological' binary evolution model, and assuming a uniform distribution in sampling from the parameter priors.  Two parameters from the galaxy stellar-mass function were varied, along with two parameters from the M-MBulge relationship, and two parameters from the hardening model.


Parameter Spaces and Distributions
==================================
**NOTE: currently parameter-spaces are only designed for use with SAMs.**

Parameter spaces are implemented as subclasses of the |pspace_class| class, and are generally named with a ``PS_`` prefix.  Each class generates a certain number of samples using a latin hypercube to efficiently sample the parameter space.  Each parameter being varied in the parameter space corresponds to parameter distribution, implemented as a :py:class:`~holodeck.librarian.libraries._Param_Dist` subclass.  Each subclass is generally named with a ``PD_`` prefix.  These parameter distributions convert from uniform random variables (uniform samples in $[0.0, 1.0]$ in the latin hypercube) to the desired distributions.  For example, the :py:class:`~holodeck.librarian.libraries.PD_Normal(mean, stdev)` class draws from a normal (Gaussian) distribution, and the :py:class:`~holodeck.librarian.libraries.PD_Normal(min, max)` class draws from a uniform distribution.

Parameter Distributions
-----------------------

New parameter distributions should subclass :py:class:`~holodeck.librarian.libraries._Param_Dist`, and must provide a method with signature: ``_dist_func(self, xx)`` which accepts a float value ``xx`` in $[0.0, 1.0]$ and maps it to a value in the desired distribution, and returns a float value.  Typically an ``__init__`` function will also be provided to set any required parameters.  See the :py:class:`~holodeck.librarian.libraries.PD_Uniform` class for a simple example that maps from $[0.0, 1.0]$ to another uniform variable with a different minimum (``lo``) and maximum (``hi``) value.

How the parameter distributions are used in parameter spaces is described below, but in summary, each |pspace_class| subclass will build a list of |pdist_class| subclass instances which are used to specify the domain of the parameter space.  The construct for each |pdist_class| subclass must accept first the variable name, and then any additional required arguments, for example: ``PD_Normal("gsmf_phi0", -2.56, 0.4)``.  The name of the variable **must match the name used in the |pspace_class|**, i.e. for the previous example, the |pspace_class| will be expecting a variable named ``gsmf_phi0``.  All |pdist_class| subclasses optionally accept a ``default=`` keyword-argument, for example, ``PD_Uniform("hard_time", 0.1, 11.0, default=3.0)``.  The 'default' values are provided so that |pspace_class|'s can construct a model using default parameters (see: :py:meth:`holodeck.librarian.libraries._Param_Space.default_params`), typically as a fiducial model.  In the preceding example, the default 'hard_time' parameter would be 3.0.  If a ``default`` is not specified in the instance constructor, then the value produced by an input of ``0.5`` is used.  In the preceding example, if no ``default`` was specified, then the middle value of $(11.0 + 0.1) / 2 = 5.55$ would be used.

Parameter Spaces
----------------

As an example, consider the |ps_test_class| class which is included as a test-case and usage example.

Parameter spaces must subclass |pspace_class|, and provide 4 elements:

(0) OPTIONAL/Recommended: A class attribute called ``DEFAULTS`` which is a ``dict`` of default parameter values for all of the parameters needed by the initialization methods.  **This is strongly recommended to ensure that parameters are set consistently, by setting them explicitly.**

   * *|ps_test_class| Example:* while this example construct a 3-dimensional parameter space (over "hard_time", "hard_gamma_inner", "mmb_mamp"), there are ``DEFAULTS`` specified for all of the parameters used to construct the GSMF, GMR, M-MBulge, and hardening models.

(1) An ``__init__()`` method that passes all required parameter distributions (:py:class:`~holodeck.librarian.libraries._Param_Dist` subclasses) to the super-class ``__init__()`` method.  The list of |pdist_class| instances is where the actual parameter-space being explored is defined.  Adding or removing a new element to this list of instances is all that it takes to increase or decrease the parameter space.

   * *|ps_test_class| Example:* in this case, a 3-dimensional parameter space is constructed, using uniform distributions (:py:class:`~holodeck.librarian.libraries.PD_Uniform`) for "hard_time" and "hard_gamma_inner", and a normal (i.e. Gaussian, :py:class:`~holodeck.librarian.libraries.PD_Normal`) distribution for "hard_time".

(2) An ``_init_sam()`` function that takes the input parameters, and then constructs and returns a |sam_class| instance.

(3) An ``_init_hard()`` function that takes the input parameters, and then constructs and returns a |hard_class| instance.

Public parameter spaces should also be 'registered' to the :py:data:`holodeck.librarian.param_spaces_dict` dictionary.  See :py:mod:`holodeck.librarian`.


Generating Libraries
====================

|holodeck| libraries are generated by running a number of simulations (i.e. SAM models) at different points in a parameter space (i.e. |pspace_class| subclass).  The module :py:mod:`holodeck.librarian.gen_lib` is designed to do this.  It provides a command-line interface to run many simulations in parallel (using MPI), or its functions can be used as API methods.  For detailed usage information, see: :py:mod:`holodeck.librarian.gen_lib`.

Once a |pspace_class| is defined, and is registered in the :py:data:`holodeck.librarian.param_spaces_dict` dictionary, then it can be accessed using the :py:mod:`holodeck.librarian.gen_lib` script.  A typical usage example is::

    mpirun -np 16  python -m holodeck.librarian.gen_lib PS_Classic_Phenom_Uniform ./output/ps-classic-phenom -n 512 -f 40

This command starts an mpi job with 16 processors, and runs the :py:mod:`holodeck.librarian.gen_lib` module.  There are two required positional arguments: first, the name of the parameter-space class (this much match exactly the class definition), and the desired output directory for simulation files.  A number of optional arguments can also be specified, in the above example for instance, the number of sample points in this library (i.e. in the latin hypercube) is set to 512, and the number of frequency bins is set to 40.  The command-line usage can be seen by running::

   python -m holodeck.librarian.gen_lib --help

The holodeck installer also makes available the command line alias ``holodeck_lib_gen`` which is equivalent to running ``python -m holodeck.librarian.gen_lib``.

In the preceding example, a :py:class:`~holodeck.librarian.param_spaces_classic.PS_Classic_Phenom_Uniform` parameter space instance is constructed which defines a 5-dimensional parameter space.  512 points are sampled using a latin hypercube, and $512/16 = 32$ sets of parameters are given to each processor.  Each set of parameters is then run as an individual simulation, and the resulting files are saved to a ``sims/`` subdirectory in the specified output path.  Once all of the jobs are completed, the results from all 512 simulations are combined into a single 'library' file called ``sam_lib.hdf5`` in the output path.  Once the ``sam_lib.hdf5`` file is created, typically the individual simulation output files in the ``sims/`` subdirectory can safely be deleted, but **note that this is not done automatically**, and thus almost twice as much space is used up as is required.

A copy of the parameter-space instance itself (in this case an instance of :py:class:`~holodeck.librarian.param_spaces_classic.PS_Classic_Phenom_Uniform`) is saved as a numpy npz file to output directory also.  This allows for the library generation to be resumed if it is halted (or fails) part way through, and also ensures that the specifications for the parameter space are easily accessible.

A more complicated job execution script which works for the SLURM job scheduler, is included in the file: `scripts/run_holodeck_lib_gen.sh <https://github.com/nanograv/holodeck/blob/dev/scripts/run_holodeck_lib_gen.sh>`.


Analytic GWB Fits to Libraries
==============================

It is often useful to fit analytic functions to the GWB spectra in a library.  This can be done using the :py:mod:`holodeck.librarian.fit_spectra` script/submodule, which is also parallelized using MPI.  This script can be run as, for example::

   mpirun -np 16  python -m holodeck.librarian.fit_spectra ./output/ps-classic-phenom

This will find the library file (``sam_lib.hdf5``) in the given directory, and fit all of the included spectra.  For more information, see: :py:mod:`holodeck.librarian.fit_spectra`.  For command-line usage information, run::

   python -m holodeck.librarian.fit_spectra --help


Using holodeck libraries
========================

Loading a saved parameter-space instance
----------------------------------------

TLDR: Use the :py:func:`~holodeck.librarian.libraries.load_pspace_from_path` function, passing in the path to the directory containing the save file (a ``.pspace.npz`` file).

Typically all that is needed for using/analyzing a holodeck library is the combined library output file ``sam_lib.hdf5``.  A saved instance of the parameter-space class which generated the library is also saved to the output directory (as a ``.pspace.npz`` file), and can be useful for some use cases, for example if new simulations/realizations are desired from the same parameter space.  The |pspace_class| provides a method to load saved instances, see the :py:meth:`~holodeck.librarian.libraries._Param_Space.from_save` method.  Typically, the best way to load a saved parameter-space instance is to use the :py:func:`~holodeck.librarian.libraries.load_pspace_from_path` function.

The combined holodeck library file ``sam_lib.hdf5``
---------------------------------------------------

In general, it is recommended to use the `h5py <https://github.com/h5py/h5py>`_ python package for handling `HDF5 files <https://en.wikipedia.org/wiki/Hierarchical_Data_Format>`_.  The HDF5 standard provides a hierarchical dictionary-like structure to store data sets (including very large ones).  It also provides structures for storing simple header/metadata (with strings, primitive types, etc), including metadata to document all datasets.  In general, every level of an hdf5 file can contains either data or 'groups' (a deeper, dictionary-like level), which are accessed as if they were dictionary elements.  Each level also has a set of metadata called `attrs` which are also accessed if they were dictionary elements.

Example:

.. code-block:: python

   # Open the HDF5 file using the `h5py` package
   library = h5py.File("./sam_lib.hdf5", "r")
   # Load the parameter-space parameter names from the file metadata
   param_names = library.attrs["param_names"]
   # Load the parameter values for each simulation in the library
   sample_params = library["sample_params"]

Metadata
^^^^^^^^

Metadata is included in the top-level ``attrs`` of the ``sam_lib.hdf5`` file.

* ``holodeck_git_hash``: string.  The git hash code for the current repository commit.  *NEW in librarian v1.1.*

* ``holodeck_librarian_version``: string.  The version number of holodeck.librarian module specifically.  *NEW in librarian v1.1.*

* ``holodeck_version``: string.  The version number of the holodeck package as a whole.  *NEW in librarian v1.1.*

* ``parameter_space_class_name``: string.  The name of the parameter-space class used to generate the library.  *NEW in librarian v1.1.*

* ``param_names``: an array of bytes (strings).  The names (and ordering) of the parameter-space parameters.  To convert to strings, use ``library.attrs['param_names'].astype(str)``.

Data
^^^^

* ``fobs_edges``: ndarray (F+1,).  Units of [Hz].  Frequency bin edges at which GW properties are evaluated.  There are one more bin-edges than bins. 

* ``fobs_cents``: ndarray (F,).  Units of [Hz].  Frequency bin centers at which GW properties are evaluated.  We use `F` to designate the number of frequency bins.

* ``sample_params``: ndarray (S, P).  Units vary.  The parameter values used to construct each simulation.  The number of simulations (sample points) is `S` and the number of parameters (dimensions of the parameter space) is `P`.  Each of the `P` parameters corresponds to the parameter names in the `param_names` metadata attribute.

If the ``gwb`` flag was used when generating the library:

* ``gwb``: ndarray (S, F, R).  Unitless (characteristic strain).  The GW background characteristic strain amplitude for each simulation `S`, frequency bin `F`, and realization `R`.

If the ``ss`` flag was used when generating the library:

* ``hc_bg``: ndarray (S, F, R).  The characteristic strain of the GW background **minus** the `L` loudest individual sources in each frequency bin (given in `hc_ss`).  This is evaluated at each simulation `S`, frequency bin `F`, and realization `R`.

* ``hc_ss``: ndarray (S, F, R, L).  The characteristic strain of the `L` loudest individual binaries in each frequency bin.  This is evaluated at each simulation `S`, frequency bin `F`, and realization `R`.

If the ``params`` flag was used when generating the library:

* ``bgpar``: ndarray (S, X, F, R).  The GWB-amplitude weighted parameters of simulated binaries for `S` simulations, `F` frequency bins, and `R` realizations.  The `X` values are the different weighted parameters.  Currently these are: 

            * [0]: total mass, in units of [gram]

            * [1]: mass ratio

            * [2]: formation redshift

            * [3]: redshift to binary at emission

            * [4]: comoving distance to binary at emission, in units of [cm]

            * [5]: binary physical separation at emission, in units of [cm]

            * [6]: binary angular separation at emission, in units of [radian]

* ``sspar``: ndarray ().  (S, X, F, R, L).  The parameters of the `L` loudest binaries for `S` simulations, `F` frequency bins, and `R` realizations.  The `Y` values are the different parameters.  Currently these are: 

            * [0]: total mass, in units of [gram]

            * [1]: mass ratio

            * [2]: formation redshift

            * [3]: redshift to binary at emission