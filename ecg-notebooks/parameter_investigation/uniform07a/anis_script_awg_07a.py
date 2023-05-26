#!/usr/bin/env python3

import holodeck as holo
from holodeck import detstats, anisotropy
from holodeck.constants import YR
import numpy as np
import healpy as hp
import h5py
import os

LIB_PATH = '/Users/emigardiner/GWs/holodeck/output/2023-05-09-mbp-ss15_n100_r30_f100_d15_l5_p0'
HC_REF15_10YR = 11.2*10**-15 

NSIDE = anisotropy.NSIDE
LMAX = 6
NBEST = 100
NREALS = 50

def main():

    anisotropy.lib_anisotropy(LIB_PATH, hc_ref_10yr=HC_REF15_10YR, nbest=NBEST,
                              nreals=NREALS, lmax=LMAX, nside=NSIDE)

if __name__ == "__main__":
    main()