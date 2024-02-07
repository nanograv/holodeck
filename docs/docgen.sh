#!/bin/zsh
# ------------------------------------------------------------------------------
# Script for building read-the-docs style sphinx documentation for holodeck.
#
# This builds a local copy, and does not need to be run for actually setting up the online
# readthedocs documentation.  That happens automatically through readthedocs, using the settings in
# the `.readthedocs.yml` file in the home directory.
#
# ------------------------------------------------------------------------------

# Filename for symlink to sphinx output html files
# the path is relative to this script's working-directory (changed to, below)
SYMLINK_FILE="readthedocs.html"

# move to this file's directory (`holodeck/docs/`)
cd "$(dirname "$0")"

# Remove previously created symlink file, if it exists
path_symlink_file=$PWD/$SYMLINK_FILE
if [ -f "$path_symlink_file" ] ; then
    rm "$path_symlink_file"
fi

# Install requirements (yes, this is also needed for building the docs)
pip install -r requirements.txt
# Cleanup previous material
make clean
rm -r ./source/apidoc_modules
# Generate apidoc automatically created documentation
# sphinx-apidoc: 'https://www.sphinx-doc.org/en/master/man/sphinx-apidoc.html'
sphinx-apidoc -e -P -T -f -o ./source/apidoc_modules ../holodeck ../holodeck/detstats.py ../holodeck/anisotropy.py ../holodeck/librarian/_librarian.py
# Run the normal sphinx build
make html
# create convenience symlink
ln -s $PWD/build/html/index.html $path_symlink_file
# move back to previous directory
cd -

# Done!