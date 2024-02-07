========
holodeck
========

**Massive Black-Hole Binary Population Synthesis for Gravitational Wave Calculations ≋●≋●≋**

`holodeck on github <https://github.com/nanograv/holodeck>`_

This package is aimed at providing a comprehensive framework for MBH binary population synthesis.  The framework includes modules to perform population synthesis using a variety of methodologies from semi-analytic models, to cosmological hydrodynamic simulations, and even observationally-derived galaxy merger catalogs.

**This File:**

.. contents:: :local:


Getting Started
===============

| (1) Read/skim the :doc:`Getting Started <getting_started/index>` guide.
| (2) Install ``holodeck`` following the `installation`_ instructions below, or in the `package README.md file <https://github.com/nanograv/holodeck/tree/main/README.md>`_.
| (3) Explore the `package demonstration notebooks <https://github.com/nanograv/holodeck/tree/main/notebooks>`_.



Installation
============

The ``holodeck`` framework is currently under substantial, active development.  Recent versions will not generally be available with ``pip`` or ``conda`` install.  Currently ``holodeck`` requires ``python >= 3.9`` (tests are run on versions ``3.9``, ``3.10``, ``3.11``).  The recommended installation is:

0) OPTIONAL but recommended: create and activate a new anaconda environment to isolate your build::

      conda create --name holo311 python=3.11; conda activate holo311

1) Clone the ``holodeck`` repository, and move into the repo directory::

      git clone https://github.com/nanograv/holodeck.git; cd holodeck

2) Install the required external packages specified in the requirements file::

      pip install -r requirements.txt

   OPTIONAL: install development requirements::

      pip install -r requirements-dev.txt

3) Build the required c libraries from ``holodeck`` ``cython`` code::

      cd holodeck; python setup.py build_ext -i

4) Perform a development/editable local installation::

      python setup.py develop

The 'editable' installation allows the code base to be modified, and have those changes take effect when using the ``holodeck`` module without having to rebuild/reinstall it.  Note that any chances to the cython library files do still require a rebuild by running steps (3) and (4) above.

MPI
---

For some scripts (particularly for generating libraries), an MPI implementation is required (e.g. ``openmpi``), along with the `mpi4py package <https://github.com/mpi4py/mpi4py>`_.  This is not included as a requirement in the ``requirements.txt`` file as it significantly increases the installation complexity, and is not needed for many ``holodeck`` use cases.  If you already have an MPI implementation installed on your system, you should be able to install ``mpi4py`` with anaconda: ``conda install mpi4py``.  To see if you have ``mpi4py`` installed, run ``python -c 'import mpi4py; print(mpi4py.__version__)'`` from a terminal.

**macos users**: if you are using homebrew on macos, you should be able to simply run: ``brew install mpi4py`` which will `include the required openmpi implementation <https://mpi4py.readthedocs.io/en/latest/install.html#macos>`_.


Development & Contributions
===========================

This project is being led by the `NANOGrav <http://nanograv.org/>`_ Astrophysics Working Group.  Details on contributions and the mandatory code of conduct can be found in `CONTRIBUTING.md <https://raw.githubusercontent.com/nanograv/holodeck/docs/CONTRIBUTING.md>`_.

Contributions are welcome and encouraged, anywhere from new modules/customizations, to bug-fixes, to improved documentation and usage examples.  The git workflow is based around a ``main`` branch which is intended to be (relatively) stable and operational, and an actively developed ``dev`` branch.  New development should be performed in "feature" branches (made off of the ``dev`` branch), and then incorporated via pull-request (back into the ``dev`` branch).

For active developers, please install the additional development package requirements::

   pip install -r requirements-dev.txt


Formatting
----------

New code should generally abide by `PEP8 formatting <https://peps.python.org/pep-0008/>`_, with `numpy-style docstrings <https://numpydoc.readthedocs.io/en/latest/format.html#>`_.  Exceptions are:

* lines may be broken at either 100 or 120 columns

Notebooks
---------

Please strip all notebook outputs before commiting notebook changes.  The `nbstripout <https://github.com/kynan/nbstripout>`_ package is an excellent option to automatically strip all notebook output only in git commits (i.e. it doesn't change your notebooks in-place).  You can also use ``nbconvert`` to strip output in place::

   jupyter nbconvert --clear-output --inplace <NOTEBOOK-NAME>.ipynb

To install ``nbstripout`` for the ``holodeck`` git repository, make sure you're in the ``holodeck`` root directory and run:

.. code-block:: bash

  pip install --upgrade nbstripout    # install nbstripout
  nbstripout --install                # install git hook in current repo only


Test Suite
----------

**Before submitting a pull request, please run the test suite on your local machine.**

Tests can be run by using ``$ pytest`` in the root holodeck directory.  Tests can also be run against all supported python versions and system configurations by using ``$ tox``.  ``tox`` creates anaconda environments for each supported python version, sets up the package and test suite, and then runs ``pytest`` to execute tests.

Two types of unit-tests are generally used in ``holodeck``.

(1) Simple functions and behaviors are included as normal unit-tests, e.g. in "holodeck/tests" and similar directories.  These are automatically run by ``pytest`` and ``tox``.

(2) More complex functionality should be tested in notebooks (in "notebooks/") where they can also be used as demonstrations/tutorials for that behavior.  Certain notebooks are also converted into unit-test modules to be automatically run by ``pytest`` and ``tox``.  The python script `scripts/convert_notebook_tests.py <https://github.com/nanograv/holodeck/blob/main/scripts/convert_notebook_tests.py>`_ converts target notebooks into python scripts in the ``holodeck/tests/converted_notebooks`` directory, which are then run by ``pytest``.  The script `scripts/tester.sh <https://github.com/nanograv/holodeck/blob/main/scripts/tester.sh>`_ will run the conversion script and then run ``pytest``.  For help and usage information, run ``$ scripts/tester.sh -h``.


Attribution & Referencing
=========================

Copyright (c) 2024, NANOGrav.

The ``holodeck`` package uses an `MIT license <https://raw.githubusercontent.com/nanograv/holodeck/docs/LICENSE>`_.

A dedicated paper on ``holodeck`` is currently in preparation, but the package is also described in the recent `astrophysics analysis from the NANOGrav 15yr dataset <https://ui.adsabs.harvard.edu/abs/2023ApJ...952L..37A/abstract>`_.

.. code-block:: tex

   @ARTICLE{2023ApJ...952L..37A,
         author = {{Agazie}, Gabriella and {et al} and {Nanograv Collaboration}},
         title = "{The NANOGrav 15 yr Data Set: Constraints on Supermassive Black Hole Binaries from the Gravitational-wave Background}",
         journal = {\apjl},
            year = 2023,
         month = aug,
         volume = {952},
         number = {2},
            eid = {L37},
         pages = {L37},
            doi = {10.3847/2041-8213/ace18b},
   archivePrefix = {arXiv},
         eprint = {2306.16220},
   primaryClass = {astro-ph.HE},
         adsurl = {https://ui.adsabs.harvard.edu/abs/2023ApJ...952L..37A},
   }


Full package documentation
==========================

.. toctree::
   :maxdepth: 2

   getting_started/index

.. toctree::
   :maxdepth: 1

   Bibliography <biblio>

.. toctree::
   :maxdepth: 2

   Full package documentation <apidoc_modules/holodeck>


Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
* :ref:`modindex`
