"""Holodeck - Electromagnetic Signals (EMS)
"""

from . import basics   # noqa
from . import drw   # noqa
from . runnoe2012 import Runnoe2012   # noqa


bands_sdss = basics.SDSS_Bands()
MacLeod2010 = drw.MacLeod2010
runnoe2012 = Runnoe2012()

