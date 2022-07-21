"""holodeck

Massive Black-Hole Binary Population Synthesis for Gravitational Wave Calculations ≋●≋●≋

This package is aimed at providing a comprehensive framework for MBH binary population synthesis.
The framework includes modules to perform pop synth using a variety of methodologies to get a handle
on both statistical and systematic uncertainties.  Currently, binary populations can be synthesis
based on: cosmological hydrodynamic simulations (Illustris), semi-analytic/semi-empirical models,
and observational catalogs of local galaxies and/or quasars.

"""

__author__ = "NANOGrav"
__copyright__ = "Copyright (c) 2022 NANOGrav"
__license__ = "MIT"

import os
import logging


# ---- Setup root package variables

_PATH_PACKAGE = os.path.dirname(os.path.abspath(__file__))
_PATH_ROOT = os.path.join(_PATH_PACKAGE, os.path.pardir)
_PATH_NOTEBOOKS = os.path.join(_PATH_ROOT, "notebooks", "")
_PATH_DATA = os.path.join(_PATH_PACKAGE, "data", "")
_PATH_OUTPUT = os.path.join(_PATH_ROOT, "output", "")

# NOTE: can only search for paths within the package _*NOT the root directory*_
_check_paths = [_PATH_PACKAGE, _PATH_ROOT, _PATH_DATA]
for cp in _check_paths:
    cp = os.path.abspath(cp)
    if not os.path.isdir(cp):
        err = "ERROR: could not find directory '{}'!".format(cp)
        raise FileNotFoundError(err)


# ---- Load logger

from . import logger   # noqa
log = logger.get_logger(__name__, logging.DEBUG)       #: global root logger from `holodeck.logger`


# ---- Import submodules

# NOTE: Must load and initialize cosmology before importing other submodules!
from . import cosmology   # noqa
cosmo = cosmology.Cosmology()              #: global cosmology instance for cosmolical calculations

from . import constants   # noqa
from . import evolution   # noqa
from . import relations   # noqa
from . import population  # noqa
from . import utils       # noqa
from . import sam         # noqa

# ---- Handle version

fname_version = os.path.join(_PATH_PACKAGE, 'version.txt')
with open(fname_version) as inn:
    version = inn.read().strip()

__version__ = version

# cleanup module namespace
del os, logging, _check_paths
