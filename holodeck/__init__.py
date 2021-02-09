"""
holodeck
Supermassive binary black hole simulator for pulsar timing array signals and galaxy population  statistics.
"""

# Must load and initialize cosmology FIRST!
from . import cosmology
cosmo = cosmology.Cosmology()

from . import constants  # noqa
# from .constants import *  # noqa
from . import evolution  # noqa
from .evolution import *  # noqa
from . import gravwaves  # noqa
from .gravwaves import *  # noqa
from . import observations # noqa
from .observations import *  # noqa
from . import population  # noqa
from .population import *  # noqa
from . import utils     # noqa
from .utils import *  # noqa

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
