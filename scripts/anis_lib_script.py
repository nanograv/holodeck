#!/usr/bin/env python3

import holodeck as holo
from holodeck import detstats, anisotropy
from holodeck.constants import YR
import numpy as np
import healpy as hp
import argparse



HC_REF_10YR = anisotropy.HC_REF15_10YR

DEF_NSIDE = anisotropy.NSIDE
DEF_LMAX = anisotropy.LMAX
DEF_NBEST = 100
DEF_NREALS = 50
DEF_SPLIT = 1

def _setup_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('lib_path', action='store', type=str,
                        help="library path")
    parser.add_argument('-s', '--nside', action='store', dest='nside', type=int, default=DEF_NSIDE,
                        help='nside for healpix maps')
    parser.add_argument('-l', '--lmax', action='store', dest='lmax', type=int, default=DEF_LMAX,
                        help='max l for spherical harmonics')
    parser.add_argument('-b', '--nbest', action='store', dest='nbest', type=int, default=DEF_NBEST,
                        help='number of best ranked samples to calculate spherical harmonics for')
    parser.add_argument('-r', '--nreals', action='store', dest='nreals', type=int, default=DEF_NREALS,
                        help='number of realizations to use ranking and harmonics, must be less than or equal to nreals of library')
    parser.add_argument('--split', action='store', dest='split', type=int, default=DEF_SPLIT,
                        help='number of sections to split nbest calculations into')
    
    args = parser.parse_args()
    return args

def main():
    args = _setup_argparse()
    anisotropy.lib_anisotropy_split(args.lib_path, hc_ref_10yr=HC_REF_10YR, nbest=args.nbest,
                              nreals=args.nreals, lmax=args.lmax, nside=args.nside, split=args.split)

if __name__ == "__main__":
    main()