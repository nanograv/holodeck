"""
"""

import os
from pathlib import Path
import sys

import numpy as np

import holodeck as holo
import holodeck.librarian


def load_pspace_from_path(log, path, space_class=None):
    """Load a _Param_Space instance from the saved file in the given directory.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to directory containing save file.
        A single file matching "*.pspace.npz" is required in that directory.
        NOTE: the specific glob pattern is specified by `holodeck.librarian.PSPACE_FILE_SUFFIX` e.g. '.pspace.npz'
    space_class : _Param_Space subclass
        Class with which to call the `from_save()` method to load a new _Param_Space instance.

    Returns
    -------
    log : `logging.Logger`
    space : `_Param_Space` subclass instance
        An instance of the `space_class` class.
    space_fname : pathlib.Path
        File that `space` was loaded from.

    """
    path = Path(path)
    if not path.exists():
        raise RuntimeError(f"path {path} does not exist!")

    # If this is a directory, look for a pspace save file
    if path.is_dir():
        pattern = "*" + holo.librarian.PSPACE_FILE_SUFFIX
        space_fname = list(path.glob(pattern))
        if len(space_fname) != 1:
            raise FileNotFoundError(f"found {len(space_fname)} matches to {pattern} in output {path}!")

        space_fname = space_fname[0]

    # if this is a file, assume that it's already the pspace save file
    elif path.is_file():
        space_fname = path

    else:
        raise

    # Based on the `space_fname`, try to find a matching PS (parameter-space) in `holodeck.param_spaces`
    if space_class is None:
        space_class = _get_space_class_from_space_fname(space_fname)

    space = space_class.from_save(space_fname, log)
    return space, space_fname


def _get_space_class_from_space_fname(space_fname):
    # Based on the `space_fname`, try to find a matching PS (parameter-space) in `holodeck.param_spaces`
    space_name = space_fname.name.split(".")[0]
    space_class = holo.librarian.param_spaces[space_name]
    '''
    # get the filename without path, this should contain the name of the PS class
    space_name = space_fname.name
    # get a list of all parameter-space classes (assuming they all start with 'PS')
    space_list = [sl for sl in dir(holo.param_spaces) if sl.startswith('PS')]
    # iterate over space classes to try to find a match
    for space in space_list:
        # exist for-loop if the names match
        # NOTE: previously the save files converted class names to lower-case; that should no
        #       longer be the case, but use `lower()` for backwards compatibility at the moment
        #       LZK 2023-05-10
        if space.lower() in space_name.lower():
            break
    else:
        raise ValueError(f"Unable to find a PS class matching {space_name}!")

    space_class = getattr(holo.param_spaces, space)
    '''
    return space_class


def _get_sim_fname(path, pnum):
    temp = holo.librarian.FNAME_SIM_FILE.format(pnum=pnum)
    temp = path.joinpath(temp)
    return temp


def get_sam_lib_fname(path, gwb_only):
    fname = 'sam_lib'
    if gwb_only:
        fname += "_gwb-only"
    lib_path = path.joinpath(fname).with_suffix(".hdf5")
    return lib_path


def get_fits_path(library_path):
    """Get the name of the spectral fits file, given a library file path.
    """
    fits_path = library_path.with_stem(library_path.stem + "_fits")
    fits_path = fits_path.with_suffix('.npz')
    return fits_path


def log_mem_usage(log):

    try:
        import resource
        # results.ru_maxrss is KB on Linux, B on macos
        mem_max = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macos systems
        if sys.platform.lower().startswith('darwin'):
            mem_max = (mem_max / 1024 ** 3)
        # linux systems
        else:
            mem_max = (mem_max / 1024 ** 2)
    except Exception:
        mem_max = np.nan

    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem_rss = process.memory_info().rss / 1024**3
        mem_vms = process.memory_info().vms / 1024**3
        msg = f"Current memory usage: max={mem_max:.2f} GB, RSS={mem_rss:.2f} GB, VMS={mem_vms:.2f} GB"
    except Exception:
        msg = "Unable to load either `resource` or `psutil`.  Try installing at least `psutil`."

    if log is None:
        print(msg, flush=True)
    else:
        log.info(msg)

    return


