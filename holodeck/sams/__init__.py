"""holodeck - Semi-Analytic Models (SAMs)

The `holodeck` SAMS construct MBH binary populations using simple semi-analytic and/or
semi-empirical relationships.  For more information, see the :doc:`SAMs getting-started guide
<../getting_started/index>`.

"""

from . import sam
from . import comps
from .sam import Semi_Analytic_Model
from .comps import (
    GSMF_Schechter, GSMF_Double_Schechter,
    GPF_Power_Law,
    GMT_Power_Law,
    GMR_Illustris,
)