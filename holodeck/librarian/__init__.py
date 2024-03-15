"""Module to generate and manage holodeck libraries.

Holodeck 'libraries' are collections of simulations in which a certain set of parameters are varied,
producing different populations and/or GW signatures at each sampled parameter value.  Libraries are
run from the same parameter-space and using the same hyper parameters.  Libraries are constructed
using a 'parameter space' class that organizes the different simulations.  The base-class is
:class:`~holodeck.librarian.libraries._Param_Space` (defined in the :mod:`holodeck.librarian.libraries`
file).  The parameter-space subclasses are given a number of different parameters to be varied.
Each parameter is implemented as a subclass of :py:class:`~holodeck.librarian.libraries._Param_Dist`,
for example the :py:class:`~holodeck.librarian.libraries.PD_Uniform` class that implements a uniform
distribution.

For more information, see the :doc:`'libraries' page in the getting-started guide
<../getting_started/libraries>`.

Notes
-----
The ``librarian`` module is composed of the following elements:

* The core components of the holodeck libraries are defined in
  :py:mod:`~holodeck.librarian.libraries`.  Constructing simulations from parameter spaces can be
  performed using the relevant parameter spaces themselves (subclasses of
  :py:class:`~holodeck.librarian.libraries._Param_Space`).

* Parameter spaces are defined in the 'param_spaces' files, particularly:

    * 'Classic' parameter spaces from the 15yr NANOGrav astrophysics analysis are in
      :py:mod:`~holodeck.librarian.param_spaces_classic`.

    * More recent parameter are in :py:mod:`~holodeck.librarian.param_spaces`.

* Library generation functionality is in the :py:mod:`~holodeck.librarian.gen_lib` module.

* Analytic fits to libraries can be performed using :py:mod:`~holodeck.librarian.fit_spectra`.

* Drawing samples (sample populations) from libraries fit to particular constraints (for example,
  the NANOGrav 15yr observations) is performed using the module
  :py:mod:`~holodeck.librarian.posterior_populations`.

"""

__version__ = "1.1"

DEF_NUM_REALS = 100     #: Default number of realizations to construct in libraries.
DEF_NUM_FBINS = 40      #: Default number of frequency bins at which to calculate GW signals.
DEF_NUM_LOUDEST = 5     #: Default number of loudest binaries to calculate in each frequency bin.
DEF_PTA_DUR = 16.03     #: Default PTA duration which determines Nyquist frequency bins [yrs].

FITS_NBINS_PLAW = [3, 4, 5, 10, 15]
FITS_NBINS_TURN = [5, 10, 15]

FNAME_SIM_FILE = "sam-lib__p{pnum:06d}.npz"
PSPACE_FILE_SUFFIX = ".pspace.npz"

from . libraries import (      # noqa
    _Param_Space, _Param_Dist,
    PD_Uniform, PD_Normal,
    run_model, load_pspace_from_path,
)

param_spaces_dict = {}    #: Registry of standard parameter-spaces
from . import param_spaces_classic as psc_temp  # noqa
param_spaces_dict.update(psc_temp._param_spaces_dict)
from . import param_spaces as ps_temp  # noqa
param_spaces_dict.update(ps_temp._param_spaces_dict)
del psc_temp, ps_temp
