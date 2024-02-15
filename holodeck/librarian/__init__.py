"""Module to generate and manage holodeck libraries.

"""

__version__ = "1.0"

DEF_NUM_REALS = 100     #: Default number of realizations to construct in libraries.
DEF_NUM_FBINS = 40      #: Default number of frequency bins at which to calculate GW signals.
DEF_NUM_LOUDEST = 5     #: Default number of loudest binaries to calculate in each frequency bin.
DEF_PTA_DUR = 16.03     #: Default PTA duration which determines Nyquist frequency bins [yrs].

FITS_NBINS_PLAW = [3, 4, 5, 10, 15]
FITS_NBINS_TURN = [5, 10, 15]

FNAME_SIM_FILE = "sam-lib__p{pnum:06d}.npz"
PSPACE_FILE_SUFFIX = ".pspace.npz"

from . import params        # noqa
from . params import (      # noqa
    _Param_Space, _Param_Dist,
    PD_Uniform, PD_Normal,
)

param_spaces_dict = {}    #: Registry of standard parameter-spaces
from . import param_spaces_classic as psc_temp  # noqa
param_spaces_dict.update(psc_temp._param_spaces_dict)
from . import param_spaces as ps_temp  # noqa
param_spaces_dict.update(ps_temp._param_spaces_dict)
del psc_temp, ps_temp
