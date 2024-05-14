"""holodeck - Semi-Analytic Models (SAMs)

The `holodeck` SAMs construct MBH binary populations using simple semi-analytic and/or
semi-empirical relationships.  For more information, see the :doc:`SAMs getting-started guide
<../getting_started/index>`.

The core element of holodeck SAMs is the :class:`~holodeck.sams.sam.Semi_Analytic_Model` class.
Instances of this class piece together the different components of the model (defined in
:mod:`holodeck.sams.components`) to construct a population.  The population is evolved with a binary
evolution model, implemented as a subclass of the :class:`~holodeck.hardening._Hardening`, typically
in the :mod:`holodeck.hardening` module.

"""

from . import sam                              # noqa
from . import components                            # noqa
from .sam import Semi_Analytic_Model           # noqa
from .components import (                           # noqa
    GSMF_Schechter, GSMF_Double_Schechter,
    GPF_Power_Law,
    GMT_Power_Law,
    GMR_Illustris,
)
