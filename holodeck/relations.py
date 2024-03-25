"""DEPRECATED: use ``host_relations.py``.

This submodule will **temporarily** allow access to elements that have now been moved to
``holodeck.host_relations`` and ``holodeck.galaxy_profiles``.  [2024-03-25]

"""

import warnings
from holodeck import log
from holodeck.host_relations import *    # noqa
from holodeck.galaxy_profiles import *   # noqa

msg = (
    "WARNING: ``holodeck.relations`` has been deprecated!  "
    "Use ``host_relations`` and/or ``galaxy_profiles``!\n"
    "This file will be removed in the near future."
)
log.warning(msg)
warnings.warn(msg)
