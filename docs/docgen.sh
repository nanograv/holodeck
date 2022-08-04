#!/bin/zsh
cd "$(dirname "$0")"    # move to this file's directory (`holodeck/docs/`)
make clean
sphinx-apidoc -f -o ./source/apidoc_modules ../holodeck
make html
ln -s $PWD/build/html/index.html $PWD/build/readthedocs.html   # create convenience symlink
cd -                    # move back to previous directory
