"""Holodeck - Electromagnetic Signals (EMS)
"""

from . import basics   # noqa
from . import drw   # noqa
from . runnoe2012 import Runnoe2012   # noqa


bands_sdss = basics.Bands_SDSS()
bands_lsst = basics.Bands_LSST()
MacLeod2010 = drw.MacLeod2010
runnoe2012 = Runnoe2012()

