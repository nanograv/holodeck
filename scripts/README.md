# `holodeck` Scripts

This folder contains scripts for the holodeck package, primarily for generating populations of binaries over ranges of parameters, libraries of GWB spectra, and executing unit-tests.

## Contents

* `convert_notebook_tests.py`
  * Converts target jupyter notebooks into python files which can then be executed as unit-tests.
* `gen_lib_sams.py`
  * Generate a (small) library of GWB spectra using SAMs.  This script is mostly for testing purposes.
  * `run_gen_lib_sams.sh` provides a SLURM script for submitting parallel jobs to run this file.
* `pop_ill.py`
  * Generate a full universe population of binaries using Illustris based populations.
* `pop_sam.py`
  * Generate a full universe population of binaries using the SAM based populations.
* `_tester.sh`
  * Deprecated script for running unit tests.  Standard practice is to use `pytest` via `tox`, but some of the included methodology in this script could be useful.