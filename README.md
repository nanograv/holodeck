# holodeck

[//]: # (Badges)
[![GitHub version](https://badge.fury.io/gh/nanograv%2Fholodeck.svg)](https://badge.fury.io/gh/nanograv%2Fholodeck)
[![build](https://github.com/nanograv/holodeck/actions/workflows/unit-tests-ci.yaml/badge.svg)](https://github.com/nanograv/holodeck/actions/workflows/unit-tests-ci.yaml)
[![codecov](https://codecov.io/gh/nanograv/holodeck/branch/main/graph/badge.svg?token=K63WQH3ED9)](https://codecov.io/gh/nanograv/holodeck)
[![Documentation Status](https://readthedocs.org/projects/holodeck-gw/badge/?version=main)](https://holodeck-gw.readthedocs.io/en/main/?badge=main)

*Massive Black-Hole Binary Population Synthesis for Gravitational Wave Calculations ≋●≋●≋*

This package, which is actively under development, is aimed at providing a comprehensive framework for MBH binary population synthesis.  The framework includes modules to perform pop synth using a variety of methodologies to get a handle on both statistical and systematic uncertainties.  Currently, binary populations can be synthesis based on: cosmological hydrodynamic simulations (Illustris), semi-analytic/semi-empirical models (SAMs), and observational catalogs of local galaxies and/or quasars.

## Installation

The `holodeck` framework is currently under substantial, active development.  It will not be available on `pypi` (`pip`) or via `conda` install until it has stabilized.  Currently `holodeck` requires `python >= 3.8`, and tests are run on versions `3.8`, `3.9`, `3.10`.

The recommended installation for active development is to:

0) OPTIONAL: create and activate a new anaconda environment to isolate your build: `conda create --name holo310 python=3.10; conda activate holo310`
1) Clone the holodeck repository: `git clone https://github.com/nanograv/holodeck.git`
2) Install the required external packages specified in the requirements file: `pip install -r requirements.txt`.  OPTIONAL: install development requirements: `pip install -r requirements-dev.txt`
3) Build the required c libraries from holodeck cython code: `cd holodeck; python setup.py build_ext -i`
4) Perform an development/editable local installation: `python setup.py develop`

The 'editable' installation allows the code base to be modified, and have those changes take effect when using the `holodeck` module without having to rebuild/reinstall it.


## Quickstart

The best way to get started is using the demonstration/testing notebooks included in the `notebooks/` directory.

## Documentation

The primary sources of documentation for `holodeck` are this `README.md` file, the notebooks included in the `notebooks/` directory, and docstrings included in the source code directly.  [`readthedocs` documentation](https://readthedocs.org/projects/holodeck-gw) are being written and improved, and a methods paper is in preparation.

## Contributing

This project is being led by the [NANOGrav](http://nanograv.org/) Astrophysics Working Group.

Details on contributions and the mandatory code of conduct can be found in the [CONTRIBUTING.md](./CONTRIBUTING.md) file.

To-do items and changes to the API should be included in the [CHANGELOG.md](./CHANGELOG.md) file.

Contributions are not only welcome but encouraged, anywhere from new modules/customizations to bug-fixes to improved documentation and usage examples.  The git workflow is based around a `main` branch which is intended to be (relatively) stable and operational, and an actively developed `dev` branch.  New development should be performed in "feature" branches (made off of the `dev` branch), and then incorporated via pull-request (back into the `dev` branch).

### Testing:

(Unit)tests should be developed in two ways: for basic functions/behaviors, standard unit-tests can be placed in the `holodeck/tests/` directory.  More complex functionality should be tested in notebooks (in `notebooks/`) where they can also be used as demonstrations/tutorials for that behavior.  The python script `scripts/convert_notebook_tests.py` converts target notebooks into python scripts in the `holodeck/tests/` directory, which can then be run by `pytest`.  The script `scripts/tester.sh` will run the conversion script and then run `pytest`.


### Formatting:

New code should generally abide by PEP8 formatting, with [`numpy` style docstrings](https://numpydoc.readthedocs.io/en/latest/format.html).  Exceptions are:

   * lines may be broken at either 100 or 120 columns

### Notebooks:

Please strip all notebook outputs before commiting notebook changes.  The [`nbstripout`](https://github.com/kynan/nbstripout) package is an excellent option to automatically strip all notebook output only in git commits (i.e. it doesn't change your notebooks in-place).  You can also use `nbconvert` to strip output in place: `jupyter nbconvert --clear-output --inplace <NOTEBOOK-NAME>.ipynb`.

To install this package for just this repository:

   * (1) install:  `pip install --upgrade nbstripout`  or  `conda install -c conda-forge nbstripout`
   * (2) enable: `nbstripout --install --attributes .gitattributes`   (run this in the top-level `holodeck/` directory)

## Copyright

Copyright (c) 2022, NANOGrav

The `holodeck` package uses an [MIT license](./LICENSE).


## Attribution

A NANOGrav paper on `holodeck` is currently in preparation.


## Acknowledgements
