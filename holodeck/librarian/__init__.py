"""Holodeck's Librarian submodule: handle library generation and management.

"""


__version__ = "1.0"

DEF_NUM_REALS = 100
DEF_NUM_FBINS = 40
DEF_NUM_LOUDEST = 10
DEF_PTA_DUR = 16.03     # [yrs]

FITS_NBINS_PLAW = [3, 4, 5, 10, 15]
FITS_NBINS_TURN = [5, 10, 15]

FNAME_SIM_FILE = "sam-lib__p{pnum:06d}.npz"
PSPACE_FILE_SUFFIX = ".pspace.npz"
