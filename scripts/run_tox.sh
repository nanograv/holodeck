#!/bin/bash
#
# `pyenv` can be installed with homebrew, e.g. `brew install pyenv`
# NOTE: the versions of python (in `VERSIONS`) need to match those in the `tox.ini` file's `envlist` variable
#

set -e
VERSIONS=("3.7:latest" "3.8:latest" "3.9:latest" "3.10:latest")

# Iterate the string array using for loop
for val in ${VERSIONS[@]}; do
   echo $val
   pyenv install -s $val
done

# pyenv local ${VERSIONS[@]}
tox