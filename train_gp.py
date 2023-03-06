#!/usr/bin/env python3
"""Train GPs on a spectral library."""

import os
from pathlib import Path

import argparse
import gp_utils as gu

# Emcee doesn't like multithreading
os.environ["OMP_NUM_THREADS"] = "1"


def setup_argparse():
    """Set up argparse for GP training."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "library_path",
        type=Path,
        help="The path to the library HDF5 file.",
    )

    parser.add_argument(
        '-f',
        '--nfreqs',
        action='store',
        dest='nfreqs',
        type=int,
        default=30,
        help=
        'Number of frequencies to train on, starting at first frequency in library.'
    )

    parser.add_argument(
        '-w',
        '--nwalkers',
        action='store',
        dest='nwalkers',
        type=int,
        help=
        'Number of emcee walkers to use. Ideally, choose 2 * available cores.',
        default=36)

    parser.add_argument(
        '-s',
        '--nsamples',
        action='store',
        dest='nsamples',
        type=int,
        help=
        'Number of emcee samples. This will be the total number of samples returned after burn-in.',
        default=1500)

    parser.add_argument(
        '-b',
        '--burn_frac',
        action='store',
        dest='burn_frac',
        type=float,
        help=
        'Fraction of samples to discard for burn-in. Starts at beginning of dataset.',
        default=0.25)

    parser.add_argument(
        '-t',
        '--test_frac',
        action='store',
        dest='test_frac',
        type=float,
        help=
        'Fraction of parameter combinations to reserve at beginning for testing.',
        default=0.0)

    parser.add_argument(
        '-c',
        '--center_measure',
        action='store',
        dest='center_measure',
        type=str,
        help=
        'The measure of center to use when returning zero-center data. Can be either "mean" or "median".',
        default="median")

    parser.add_argument(
        '-k',
        '--kernel',
        action='store',
        dest='kernel',
        type=str,
        help=
        'The kernel to use for the GP.',
        default="ExpSquaredKernel")

    args = parser.parse_args()

    return args


def main(args):
    """Train GPs on user supplied library."""
    gu.train_gp(spectra_file=args.library_path,
                nfreqs=args.nfreqs,
                nwalkers=args.nwalkers,
                nsamples=args.nsamples,
                burn_frac=args.burn_frac,
                test_frac=args.test_frac,
                center_measure=args.center_measure,
                kernel=args.kernel)


if __name__ == "__main__":
    args = setup_argparse()

    main(args)
