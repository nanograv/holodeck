#!/usr/bin/env python3
"""
Module to combine indvidual GPs.

Usage: python combine_gps.py -i <path_to_pkl> <maybe_another> <as_many_as_you_have> -o <save_merged_here>

The saved PKLs are of the same format as the input: List[GaussProc].
"""

import argparse
import operator
import pickle
from functools import reduce
from pathlib import Path


def parse_cli_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        dest="input_gps",
        type=Path,
        help="Paths to PKLs of GPs. These need to be in order of increasing frequency!",
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="merged_gps",
        type=Path,
        help="Path to output merged PKL",
        default=Path(Path.cwd() / "merged_gps.pkl"),
    )
    return parser.parse_args()


def main():
    # Get CLI args
    options = parse_cli_args()

    # Make sure the args make sense
    if len(options.input_gps) < 2:
        raise ValueError("Need at least two input files!")

    unpkld_gps = []

    # Iterate over the supplied PKLs
    for pkl in options.input_gps:
        with open(pkl, "rb") as f:
            data = pickle.load(f)
            unpkld_gps.append(data)

    # Reduce to a single list from the list-of-lists
    # We could have just taken the zeroeth element above when loading, but this
    # method accounts for cases where we "chunk" the frequencies, which we might
    # implement in the near future.
    unpkld_gps = reduce(operator.add, unpkld_gps)

    # Write out to disk
    with open(options.merged_gps.with_suffix(".pkl"), "wb") as f:
        pickle.dump(unpkld_gps, f)


if __name__ == "__main__":
    main()
