"""
holodeck
Supermassive binary black hole simulator for pulsar timing array signals and galaxy population statistics.
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

# NOTE: can only search for paths within the package _*NOT the root directory*_
_check_paths = [_PATH_PACKAGE, _PATH_ROOT, _PATH_DATA]
for cp in _check_paths:
    cp = os.path.abspath(cp)
    if not os.path.isdir(cp):
        err = "ERROR: could not find directory '{}'!".format(cp)
        raise FileNotFoundError(err)


# ---- Load logger

from . import log   # noqa
log = log.get_logger(__name__, logging.DEBUG)


# ---- Import submodules

# Must load and initialize cosmology FIRST!
from . import cosmology   # noqa
cosmo = cosmology.Cosmology()

from . import constants   # noqa
from . import evolution   # noqa
from . import relations   # noqa
from . import population  # noqa
from . import utils       # noqa
from . import sam         # noqa

# from . import constants  # noqa
# from .constants import *  # noqa
# from . import evolution  # noqa
# from .evolution import *  # noqa
# from . import gravwaves  # noqa
# from .gravwaves import *  # noqa
# from . import observations # noqa
# from .observations import *  # noqa
# from . import population  # noqa
# from .population import *  # noqa
# from . import utils     # noqa
# from .utils import *  # noqa


# ---- Handle version

fname_version = os.path.join(_PATH_PACKAGE, 'version.txt')
with open(fname_version) as inn:
    version = inn.read().strip()

__version__ = version

# Full cleanup
del os, _check_paths, logging
