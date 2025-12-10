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

fobs_edges = holo.utils.nyquist_freqs_edges()
fobs_cents = holo.utils.midpoints(fobs_edges, log=False)

print("0 begin")
holo.librarian._log_mem_usage(None)

# ---- initialization

gsmf = holo.sam.GSMF_Schechter()               # Galaxy Stellar-Mass Function (GSMF)
gpf = holo.sam.GPF_Power_Law()                 # Galaxy Pair Fraction         (GPF)
gmt = holo.sam.GMT_Power_Law()                 # Galaxy Merger Time           (GMT)
mmbulge = holo.host_relations.MMBulge_Standard()    # M-MBulge Relation            (MMB)

holo.librarian._log_mem_usage(None)

sam = holo.sam.Semi_Analytic_Model(gsmf=gsmf, gpf=gpf, gmt=gmt, mmbulge=mmbulge, shape=SHAPE)
hard = holo.hardening.Fixed_Time.from_sam(sam, TIME)

print("1 init")
holo.librarian._log_mem_usage(None)

# ---- static_binary_density

dens = sam.static_binary_density

print("2 static_binary_density")
holo.librarian._log_mem_usage(None)

# ---- dynamic_binary_number

vals = sam.dynamic_binary_number(hard, fobs_orb=fobs_cents/2.0)

print("3 dynamic_binary_number")
holo.librarian._log_mem_usage(None)

# ---- GWB

gwb = sam.gwb(fobs_edges, hard=hard, realize=NREALS)

print("4 GWB")
holo.librarian._log_mem_usage(None)
