#!/usr/bin/env python3
"""Train GPs on a spectral library."""

import argparse
import configparser
import os
import pickle
import shutil
from datetime import datetime
from pathlib import Path

import holodeck.gps.gp_utils as gu

# Emcee doesn't like multithreading
os.environ["OMP_NUM_THREADS"] = "1"


def setup_argparse():
    """Set up argparse for GP training."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "config_path",
        type=Path,
        help="The path to the ini configuration file.",
    )

    args = parser.parse_args()

    return args


def main(config):
    """Train GPs on user-supplied library

    Parameters
    ----------
    config : configparser.ConfigParser
        The ConfigParser object that contains the configuration options for
        training

    Raises
    ------
    FileNotFoundError
        Raise if path to training library does not exist

    Examples
    --------
    FIXME: Add docs.

    """
            # Split up config into sections
    train_opts = config['Training Options']
    kern = dict(config['Kernel'])
    # Make sure the library exists
    spectra_file = Path(train_opts.get('spectra_file'))
    if not spectra_file.exists():
        raise FileNotFoundError(
            f"The library at {spectra_file} does not exist!")

    trained_gps = gu.train_gp(spectra_file=spectra_file,
                              nfreqs=train_opts.getint('nfreqs', None),
                              nwalkers=train_opts.getint('nwalkers', 36),
                              nsamples=train_opts.getint('nsamples', 1500),
                              burn_frac=train_opts.getfloat('burn_frac', 0.25),
                              test_frac=train_opts.getfloat('test_frac', 0.0),
                              center_measure=train_opts.get(
                                  'center_measure', 'median'),
                              mpi=train_opts.getboolean('mpi', True),
                              kernel=kern)

    # Add datestring to ensure unique name
    datestr = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save the trained GP as a pickle to be used with PTA data!
    gp_file = Path("trained_gp_" + spectra_file.parent.name + "_" + datestr +
                   ".pkl")
    loc_gp_file = spectra_file.parent / gp_file

    with open(loc_gp_file, "wb") as gpf:
        pickle.dump(trained_gps, gpf)
    print(f"GPs are saved at {spectra_file.parent / gp_file}")

    # Copy config file to same directory, use same datestring so that we can
    # match GPs with the config files later
    conf_copy_dest = spectra_file.parent / (args.config_path.stem + datestr +
                                            args.config_path.suffix)
    shutil.copy(args.config_path, conf_copy_dest)
    print(f"Config file copied {args.config_path} -> {conf_copy_dest}")


if __name__ == "__main__":
    args = setup_argparse()
    config = configparser.ConfigParser()
    config.read(args.config_path)

    main(config)
