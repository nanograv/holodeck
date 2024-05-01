"""
"""

from holodeck import log

try:
    import schwimmbad
    import emcee
    import george
except ImportError as err:
    log.error("Failed to import packages used in `gps` submodule!")
    log.exception(err)
    log.error(
        "Some required packages for the `gps` submodule have been temporarily disabled in the "
        "global 'requirements.txt' file, so they are not installed by default!  Please install "
        "the required packages manually for now, and feel free to raise a github issue."
    )
    raise

# ---- Import submodules

# from . import gp_utils        # noqa
# from . import plotting_utils  # noqa
# from . import sam_utils       # noqa
