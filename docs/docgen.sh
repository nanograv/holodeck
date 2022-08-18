#!/bin/zsh
SYMLINK_FILE="readthedocs.html"    # relative to script's WD
cd "$(dirname "$0")"    # move to this file's directory (`holodeck/docs/`)

path_symlink_file=$PWD/$SYMLINK_FILE
if [ -f "$path_symlink_file" ] ; then
    rm "$path_symlink_file"
fi

pip install -r requirements.txt
make clean
sphinx-apidoc -e -P -T -f -o ./source/apidoc_modules ../holodeck
make html
# create convenience symlink
ln -s $PWD/build/html/index.html $path_symlink_file
# move back to previous directory
cd -
