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
* `tester.sh`
  * Convenience script for running unit tests.  This script will convert holodeck notebooks into test modules (in `tests/converted_notebooks/`), and then run `pytest` on both the standard unit tests and also the converted notebooks.