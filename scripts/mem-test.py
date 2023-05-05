"""Memory test script.

Can be analyzed using the `memray` package.
https://bloomberg.github.io/memray/


Run with:
    $ memray run mem-test.py

    This will produce an output file like "memray-mem-test.py.61610.bin"


Generate output with (e.g.):
    $ memray tree memray-mem-test.py.61610.bin


"""

import holodeck as holo
from holodeck.constants import GYR

SHAPE = 60
TIME = 3 * GYR
NREALS = 100

print("0")
holo.librarian._log_mem_usage(None)

gsmf = holo.sam.GSMF_Schechter()               # Galaxy Stellar-Mass Function (GSMF)
gpf = holo.sam.GPF_Power_Law()                 # Galaxy Pair Fraction         (GPF)
gmt = holo.sam.GMT_Power_Law()                 # Galaxy Merger Time           (GMT)
mmbulge = holo.relations.MMBulge_Standard()    # M-MBulge Relation            (MMB)

print("1")
holo.librarian._log_mem_usage(None)

sam = holo.sam.Semi_Analytic_Model(gsmf=gsmf, gpf=gpf, gmt=gmt, mmbulge=mmbulge, shape=SHAPE)
hard = holo.hardening.Fixed_Time.from_sam(sam, TIME)

print("2")
holo.librarian._log_mem_usage(None)

fobs_edges = holo.utils.nyquist_freqs_edges()
fobs_cents = holo.utils.midpoints(fobs_edges, log=False)
# gwb = sam.gwb(fobs_edges, hard=hard, realize=NREALS, use_redz_after_hard=False)

vals = sam.dynamic_binary_number(hard, fobs_orb=fobs_cents/2.0)

print("3")
holo.librarian._log_mem_usage(None)
