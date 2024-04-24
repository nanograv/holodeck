"""Copy holodeck SAM library files from a set of output directories into a single output folder.

For usage information, run:

    python copy_sam_libs.py -h

Example:

    python holodeck/scripts/copy_sam_libs.py ~/scratch/astro-strong ~/scratch/ -p "astro-strong-*"


Notes
-----
* The search for folders and library files are NOT recursive.
* The original copies are not deleted or modified.
* The copied files are automatically renamed.
* Configuration and pspace files are also copied and renamed.

"""

from pathlib import Path
import re
import shutil

import argparse

from holodeck.librarian import (
    ARGS_CONFIG_FNAME, PSPACE_FILE_SUFFIX
)

ALLOW_MISSING_CONF = True
ALLOW_MISSING_PSPACE = False


def main():

    # load command-line arguments
    args = _setup_argparse()

    # find files
    if args.debug:
        print("-"*10 + " finding matching files " + "-"*10)
    lib_files, pspace_files, conf_files = find_files(args, "sam_lib.hdf5")

    # copy files to output directory
    if args.debug:
        print("-"*10 + " copying files to new directory " + "-"*10)
    copy_files(args, conf_files)
    copy_files(args, pspace_files)
    copy_files(args, lib_files)
    
    return


def find_files(args, file_pattern):
    """

    Find folders within the 'start_path' that match 'folder_pattern' (not recursive), then return
    files matching 'file_pattern' within those folders (not recursive).
    """
    lib_files = []
    pspace_files = []
    conf_files = []
    start_path = _to_abs_path(args.search_path)
    # Compile the regex patterns
    file_regex = re.compile(re.escape(file_pattern))
    folder_pattern = args.pattern
    # folder_regex = re.compile(re.escape("*" + folder_pattern))
    if args.debug:
        print(f"{start_path=}")
        print(f"{file_pattern=} ==> {file_regex=}")
        print(f"{folder_pattern=}")
    
    for path in start_path.glob(folder_pattern):
        if not path.is_dir():
            continue

        # for path in start_path.rglob('*')
        # if not folder_regex.match(str(path)):
        #     continue

        if args.debug:
            print(f"Found {path=} ...")
        
        for file in path.glob('*'):
            # Check if the file matches the file pattern
            if not file.is_file() or not file_regex.search(str(file)):
                continue

            # store library file
            lib_files.append(file)
            if args.debug:
                print(f"===> found {file=}")

            # get parameter-space save file
            pspace_file = list(path.glob('*' + PSPACE_FILE_SUFFIX))
            if len(pspace_file) == 1:
                pspace_files.append(pspace_file[0])
            else:
                err = f"Could not find unambiguous parameter-space file!  matches={pspace_file}"
                if ALLOW_MISSING_PSPACE:
                    print(err)
                    pspace_files.append(None)
                else:
                    raise FileNotFoundError(err)

            # get configuration file
            conf_file = path.joinpath(ARGS_CONFIG_FNAME)
            if conf_file.is_file():
                conf_files.append(conf_file)
            else:
                err = f"Could not find configuration file!  '{conf_file}'"
                if ALLOW_MISSING_CONF:
                    print(err)
                    conf_files.append(None)
                else:
                    raise FileNotFoundError(err)
                
    if args.debug:
        print(f"Found {len(lib_files)} files.")

    return lib_files, pspace_files, conf_files


def copy_files(args, files):
    """Copy all of the given files to the output (``args.output``) directory.
    """
    
    for src in files:
        if src is None:
            continue
        src = Path(src)
        assert src.is_file()

        folder = src.parts[-2]

        if args.rename:
            new_name = folder + "_" + src.name
            dst = args.output.joinpath(new_name)
        else:
            new_name = src.name
            dst = args.output.joinpath(folder, new_name)

        if args.debug:
            print(f"{src} ==>\n\t==> {dst}")

        if dst.exists() and not args.overwrite:
            print(f"destination already exists, skipping!  '{dst}'\n")
            print("Use `--overwrite` to overwrite the file.")
            
        if not args.dry_run:
            shutil.copy(src, dst)
            assert dst.is_file()
            if not args.debug:
                print(dst)
            
    return


def _setup_argparse(*args, **kwargs):

    parser = argparse.ArgumentParser()
    parser.add_argument('output', metavar='output', type=str,
                        help='output path [created if doesnt exist].')

    parser.add_argument('search_path', type=str,
                        help="where  to start the search for matching folders.")

    parser.add_argument('-p', '--pattern', action='store', type=str, default='*',
                        help="regex for folders to match (NOTE: put this in quotations!).")

    parser.add_argument('-o', '--overwrite', action='store_true', default=False,
                        help="overwrite existing files [otherwise raise error].")

    # If this is disabled, it will cause problems with config and pspace files...
    # so don't leave it as an option for now.  See hard-coded value below.
    #parser.add_argument('--rename', type=str_to_bool, nargs='?', default=True,
    #                    help='rename the sam_lib files based on their folder name.')
    
    parser.add_argument('--debug', action='store_true', default=False,
                        help='debug.')
    
    parser.add_argument('--dry-run', action='store_true', default=False, dest='dry_run',
                        help='dry-run.')
    
    namespace = argparse.Namespace(**kwargs)
    args = parser.parse_args(*args, namespace=namespace)

    # See note above, hard-code rename as true
    args.rename = True
    
    # ---- check / sanitize arguments

    if args.dry_run:
        print("`dry-run` is enabled.  Settings `debug=True` automatically.")
        args.debug = True

    args.output = _to_abs_path(args.output)
    if args.debug:
        print(f"absolute path: {args.output=}")
    if args.output.is_file():
        raise RuntimeError(f"The output path is already a file!  {output}")
    args.output.mkdir(parents=True, exist_ok=True)        

    return args


def str_to_bool(v):
    """Convert string to boolean value."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def _to_abs_path(rpath):
    apath = Path(rpath).resolve()
    if not apath.is_absolute:
        apath = Path('.').resolve() / apath
        apath = apath.resolve()
    return apath


if __name__ == "__main__":
    main()


