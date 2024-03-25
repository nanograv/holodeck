"""holodeck: Massive Black-Hole Binary Population Synthesis & Gravitational Wave Calculations ≋●≋●≋

This package is aimed at providing a comprehensive framework for MBH binary population synthesis.
The framework includes modules to perform pop synth using a variety of methodologies to get a handle
on both statistical and systematic uncertainties.  Currently, binary populations can be synthesis
based on: cosmological hydrodynamic simulations (Illustris), semi-analytic/semi-empirical models,
and observational catalogs of local galaxies and/or quasars.

See the `README.md` file for more information.
The github repository is: `<https://github.com/nanograv/holodeck>`_.
Additional documentation can be found at: `<holodeck-gw.readthedocs.io/en/docs/index.html>`_.
Note that the readthedocs documentation can also be built locally from the `holodeck/docs` folder.
A methods paper for `holodeck` is currently in preparation.

In general, `holodeck` calculations proceed in three stages:

(1) **Population**: Construct an initial population of MBH 'binaries'.  This is typically done for
    pairs of MBHs when their galaxies merge (i.e. long before the two MBHs are actually a
    gravitationally-bound binary).  Constructing the initial binary population may occur in a
    single step: e.g. gathering MBH-MBH encounters from cosmological hydrodynamic simulations; or
    it may occur over two steps: (i) gathering galaxy-galaxy encounters, and (ii) prescribing MBH
    properties for each galaxy.
(2) **Evolution**: Evolve the binary population from their initial conditions (i.e. large
    separations) until coalescence (i.e. small separations).  The complexity of this evolutionary
    stage can range tremendously in complexity.  In the simplest models, binaries are assumed to
    coalesce instantaneously (in that the age of the universe is the same at formation and
    coalescence), and are assumed to evolve purely due to GW emission (in that the time spent in
    any range of orbital frequencies can be calculated from the GW hardening timescale).  Note
    that these two assumptions are contradictory.
(3) **Gravitational Waves**: Calculate the resulting GW signals based on the binaries and their
    evolution.  Note that GWs can only be calculated based on some sort of model for binary
    evolution.  The model may be extremely simple, in which case it is sometimes glanced over.

"""

__author__ = "NANOGrav"
__copyright__ = "Copyright (c) 2024 NANOGrav"
__license__ = "MIT"

import os
import logging

__all__ = ["log", "cosmo"]

# ---- Define Global Parameters


class Parameters:
    """These are WMAP9 parameters, see: [WMAP9]_ Table 3, WMAP+BAO+H0
    """
    Omega0 = 0.2880                #: Matter density parameter "Om0"
    OmegaBaryon = 0.0472           #: Baryon density parameter "Ob0"
    HubbleParam = 0.6933           #: Hubble Parameter as H0/[100 km/s/Mpc], i.e. 0.69 instead of 69


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
    if not os.path.isdir(cp):   # nocov
        err = "ERROR: could not find directory '{}'!".format(cp)
        raise FileNotFoundError(err)


# ---- Load logger

from . import logger   # noqa
log = logger.get_logger(__name__, logging.WARNING)       #: global root logger from `holodeck.logger`

# ---- Load cosmology instance

# NOTE: Must load and initialize cosmology before importing other submodules!
import cosmopy   # noqa
cosmo = cosmopy.Cosmology(h=Parameters.HubbleParam, Om0=Parameters.Omega0, Ob0=Parameters.OmegaBaryon)
del cosmopy

# ---- Import submodules

from . import constants       # noqa
# from . import evolution       # noqa
# from . import gps             # noqa
# from . import gravwaves       # noqa
# from . import hardening       # noqa
# from . import librarian       # noqa
# from . import param_spaces    # noqa
# from . import plot            # noqa
# from . import population      # noqa
# from . import host_relations  # noqa
# from . import sams            # noqa
# from . import librarian       # noqa
from . import utils           # noqa

# ---- Handle version

fname_version = os.path.join(_PATH_PACKAGE, 'version.txt')
with open(fname_version) as inn:
    version = inn.read().strip()

__version__ = version

# cleanup module namespace
del os, logging, _check_paths
