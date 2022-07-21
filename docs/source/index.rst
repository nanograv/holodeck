========
holodeck
========

|rtd| |actions|

.. |rtd| image:: https://readthedocs.org/projects/kalepy/badge/?version=latest
.. |actions| image:: https://github.com/nanograv/holodeck/actions?query=workflow%3ACI

*Massive Black-Hole Binary Population Synthesis.*

`holodeck on github <https://github.com/nanograv/holodeck>`_

This package, which is actively under development, is aimed at providing a comprehensive framework for MBH binary population synthesis.  The framework includes modules to perform pop synth using a variety of methodologies to get a handle on both statistical and systematic uncertainties.  Currently, binary populations can be synthesis based on: cosmological hydrodynamic simulations (Illustris), semi-analytic/semi-empirical models (SAMs), and observational catalogs of local galaxies and/or quasars.

.. contents:: :local:


Contents of Documentation
=========================

| A guide can be found in :doc:`Getting Started <getting_started>`.
| `The README file on github also includes installation and quick-start examples. <https://github.com/nanograv/holodeck/blob/master/README.md>`_.
| There are a large number of demonstration/testing notebooks included in `the package notebooks <https://github.com/nanograv/holodeck/tree/master/notebooks>`_.

.. toctree::
   :maxdepth: 1

   Getting Started Guide <getting_started>

   Calculating Gravitational Waves <calc_gws>

   Definitions and Abbreviations <defs_abbrevs>

   Annotated Bibliography <biblio>

   Full Package Documentation <apidoc_modules/holodeck>

   Modules list <apidoc_modules/modules>




Installation
============

The `holodeck` framework is currently under substantial, active development.  It will not available on `pypi` (`pip`) or via `conda` install until it has stabilized.  Currently `holodeck` requires `python >= 3.8` (See `Python Versions`_ below), and tests are run on versions `3.8`, `3.9`, `3.10`.  To build directly from source:

.. code-block:: bash

    git clone https://github.com/nanograv/holodeck.git
    pip install -e holodeck -r holodeck/requirements-dev.txt

The `pip install -e` command builds the package in 'editable' mode, so that changes to the source code are reflected in the package when it's imported from external python code.  The `-r holodeck/requirements-dev.txt` install not only the standard package requirements, but also the 'development' requirements.

Currently tests are run on python versions 3.8, 3.9, 3.10.  Ensure that you are using version >= 3.8

To uninstall the package, you can run `pip uninstall holodeck`.

Python Versions
---------------

If you do not currently use one of these python versions, it is strongly recommended to use `anaconda <https://www.anaconda.com/products/distribution>`_ which makes managing/using multiple python versions very easy.  `anaconda` is available through most OS package managers, for example `homebrew` on macos.  With `anaconda` installed, you can create a new conda environment:

.. code-block:: bash

   conda create -n py39_holodeck python=3.9     # create a new environment named `py39_holodeck` with python version 3.9
   conda activate py39_holodeck                 # activate the new conda environment
   pip install -e . -r requirements-dev.txt     # install holodeck from within the holodeck top-level directory

All conda environments currently active on your system can be listed with `conda info -e`.  To deactivate an environment, use the command `conda deactivate`.


Development & Contributions
===========================

This project is being led by the `NANOGrav <http://nanograv.org/>`_ Astrophysics Working Group.

Details on contributions and the mandatory code of conduct can be found in `CONTRIBUTING.md <https://raw.githubusercontent.com/nanograv/holodeck/docs/CONTRIBUTING.md>`_.

To-do items and changes to the API should be included in the `CHANGELOG.md <https://raw.githubusercontent.com/nanograv/holodeck/docs/CHANGELOG.md>`_.

Contributions are not only welcome but encouraged, anywhere from new modules/customizations to bug-fixes to improved documentation and usage examples.  The git workflow is based around a `main` branch which is intended to be (relatively) stable and operational, and an actively developed `dev` branch.  New development should be performed in "feature" branches (made off of the `dev` branch), and then incorporated via pull-request (back into the `dev` branch).

Formatting
----------

New code should generally abide by PEP8 formatting, with `numpy` style docstrings.  Exceptions are:

* lines may be broken at either 100 or 120 columns

Notebooks
---------

Please strip all notebook outputs before commiting notebook changes.  The `[nbstripout](https://github.com/kynan/nbstripout)` package is an excellent option to automatically strip all notebook output only in git commits (i.e. it doesn't change your notebooks in-place).  You can also use `nbconvert` to strip output in place: `jupyter nbconvert --clear-output --inplace <NOTEBOOK-NAME>.ipynb`.

To install `nbstripout` for the `holodeck` git package, make sure you're in the `holodeck` root directory and run:

.. code-block:: bash

  pip install --upgrade nbstripout    # install nbstripout
  nbstripout --install                # install git hook in current repo only


Test Suite
----------

(Unit)tests should be developed in two ways: for simple functions/behaviors, standard unit-tests can be placed in the `holodeck/tests/` directory.  More complex functionality should be tested in notebooks (in `notebooks/`) where they can also be used as demonstrations/tutorials for that behavior.  The python script `scripts/convert_notebook_tests.py` converts target notebooks into python scripts in the `holodeck/tests/` directory, which can then be run by `pytest`.  The script `scripts/tester.sh` will run the conversion script and then run `pytest`.

The full test suite can be run on all supported python versions using `tox`.  In the base package directory, simply run `$ tox` and it will use `conda` to create evironments and run tests on all supported python versions.

**Before submitting a pull request, run `scripts/tester.sh -bv` to run the builtin tests.**
For more comprehensive testing (e.g. against numerous python versions, and building from the sdist package), you can use the python `tox` package: simply run `tox` in the root directory and it will use the configuration specified in `tox.ini`.  The scripy `scripts/run_tox.sh` is also provided to setup and execute tox tests and required environments.


Attribution & Referencing
=========================

Copyright (c) 2022, NANOGrav.

The `holodeck` package uses an `MIT license <https://raw.githubusercontent.com/nanograv/holodeck/docs/LICENSE>`_.

A NANOGrav paper on `holodeck` is currently in preparation.

.. .. code-block:: tex

..     @article{Kelley2021,
..       doi = {10.21105/joss.02784},
..       url = {https://doi.org/10.21105/joss.02784},
..       year = {2021},
..       publisher = {The Open Journal},
..       volume = {6},
..       number = {57},
..       pages = {2784},
..       author = {Luke Zoltan Kelley},
..       title = {kalepy: a Python package for kernel density estimation, sampling and plotting},
..       journal = {Journal of Open Source Software}
..     }

Indices and tables
==================
..    :template: custom-module-template.rst

.. autosummary::
   :toctree: _autosummary
   :recursive:

..    holodeck

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`