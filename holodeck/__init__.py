"""
holodeck
Supermassive binary black hole simulator for pulsar timing array signals and galaxy population  statistics.
"""

from . import utils     # noqa
from . import holodeck  # noqa

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
