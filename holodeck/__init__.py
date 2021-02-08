"""
holodeck
Supermassive binary black hole simulator for pulsar timing array signals and galaxy population  statistics.
"""

# Must load and initialize cosmology FIRST!
from . import cosmology
cosmo = cosmology.Cosmology()

from . import utils     # noqa
from . import holodeck  # noqa
from . import observations # noqa


# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
