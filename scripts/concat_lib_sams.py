"""Copied over from `gen_lib_sams.py` on 2022-12-04.  This should be a temporary fix!
"""

import argparse
import os
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import h5py
import tqdm

import holodeck as holo
from holodeck import log

log.setLevel(log.INFO)

DEBUG = False


def get_argparse():
    # ---- Setup ArgParse

    parser = argparse.ArgumentParser()
    parser.add_argument('output', metavar='output', type=str,
                        help='output path [created if doesnt exist]')

    args = parser.parse_args()
    return args


def main(args):
    PATH_OUTPUT = Path(args.output).resolve()
    if not PATH_OUTPUT.is_absolute:
        PATH_OUTPUT = Path('.').resolve() / PATH_OUTPUT
        PATH_OUTPUT = PATH_OUTPUT.resolve()

    log.info(f"{PATH_OUTPUT=}")

    holo.librarian.sam_lib_combine(PATH_OUTPUT, log)
    return


if __name__ == "__main__":
    args = get_argparse()
    main(args)
    print("Done.")
    sys.exit(0)
