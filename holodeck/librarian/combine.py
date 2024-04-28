"""Combine output files from individual simulation runs into a single library hdf5 file.

This file can be executed as a script (see the :func:`main` function), and also provides an API
method (:func:`sam_lib_combine`) for programatically combining libraries.

"""

import argparse
from pathlib import Path

import numpy as np
import h5py
import tqdm

import holodeck as holo
import holodeck.librarian
from holodeck.librarian import libraries


def main():
    """Command-line interface executable method for combining holodeck simulation files.
    """

    log = holo.log

    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
    except ModuleNotFoundError:
        comm = None

    if (comm is not None) and (comm.rank > 0):
        err = f"Cannot run `{__file__}::main()` with multiple processors!"
        log.exception(err)
        raise RuntimeError(err)

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'path', default=None,
        help='library directory to run combination on; must contain the `sims` subdirectory'
    )
    parser.add_argument(
        '--recreate', '-r', action='store_true', default=False,
        help='recreate/replace existing combined library file with a new merge.'
    )
    parser.add_argument(
        '--gwb', action='store_true', default=False,
        help='only merge the key GWB data (no single source, or binary parameter data).'
    )

    args = parser.parse_args()
    log.debug(f"{args=}")
    path = Path(args.path)

    sam_lib_combine(path, log, recreate=args.recreate, gwb_only=args.gwb)

    return


def sam_lib_combine(path_output, log, path_pspace=None, recreate=False, gwb_only=False):
    """Combine individual simulation files into a single library (hdf5) file.

    Arguments
    ---------
    path_output : str or Path,
        Path to output directory where combined library will be saved.
    log : `logging.Logger`
        Logging instance.
    path_sims : str or None,
        Path to output directory containing simulation files.
        If `None` this is set to be the same as `path_output`.
    path_pspace : str or None,
        Path to file containing _Param_Space subclass instance.
        If `None` then `path_output` is searched for a `_Param_Space` save file.

    Returns
    -------
    lib_path : Path,
        Path to library output filename (typically ending with 'sam_lib.hdf5').

    """

    # ---- setup paths

    path_output = Path(path_output)
    log.info(f"Path output = {path_output}")
    path_sims = path_output.joinpath('sims')

    # ---- see if a combined library already exists

    lib_path = libraries.get_sam_lib_fname(path_output, gwb_only)
    if lib_path.exists():
        lvl = log.INFO if recreate else log.WARNING
        log.log(lvl, f"combined library already exists: {lib_path}, run with `-r` to recreate.")
        if not recreate:
            return

        log.log(lvl, "re-combining data into new file")

    # ---- load parameter space from save file

    '''
    if path_pspace is None:
        # look for parameter-space save files
        regex = "*" + holo.librarian.PSPACE_FILE_SUFFIX   # "*.pspace.npz"
        files = sorted(path_output.glob(regex))
        num_files = len(files)
        msg = f"found {num_files} pspace.npz files in {path_output}"
        log.info(msg)
        if num_files != 1:
            log.exception(msg)
            log.exception(f"{files=}")
            log.exception(f"{regex=}")
            raise RuntimeError(f"{msg}")
        path_pspace = files[0]

    pspace = _Param_Space.from_save(path_pspace, log)
    '''
    if path_pspace is None:
        path_pspace = path_output
    pspace, pspace_fname = libraries.load_pspace_from_path(path_pspace, log=log)

    log.info(f"loaded param space: {pspace} from '{pspace_fname}'")
    param_names = pspace.param_names
    param_samples = pspace.param_samples
    nsamp, ndim = param_samples.shape
    log.debug(f"{nsamp=}, {ndim=}, {param_names=}")

    # ---- make sure all files exist; get shape information from files

    log.info(f"checking that all {nsamp} files exist")
    fobs_cents, fobs_edges, nreals, nloudest, has_gwb, has_ss, has_params = _check_files_and_load_shapes(
        log, path_sims, nsamp
    )
    nfreqs = fobs_cents.size
    log.debug(f"{nfreqs=}, {nreals=}, {nloudest=}")
    log.debug(f"{has_gwb=}, {has_ss=}, {has_params=}")

    if not has_gwb and gwb_only:
        err = f"Combining with {gwb_only=}, but received {has_gwb=} from `_check_files_and_load_shapes`!"
        log.exception(err)
        raise RuntimeError(err)

    if (fobs_cents is None) or (nreals is None):
        err = f"After checking files, {fobs_cents=} and {nreals=}!"
        log.exception(err)
        raise ValueError(err)

    # ---- load results from all files

    gwb = np.zeros((nsamp, nfreqs, nreals)) if has_gwb else None

    if (not gwb_only) and has_ss:
        hc_ss = np.zeros((nsamp, nfreqs, nreals, nloudest))
        hc_bg = np.zeros((nsamp, nfreqs, nreals))
    else:
        hc_ss = None
        hc_bg = None

    if (not gwb_only) and has_params:
        sspar = np.zeros((nsamp, 4, nfreqs, nreals, nloudest))
        bgpar = np.zeros((nsamp, 7, nfreqs, nreals))
    else:
        sspar = None
        bgpar = None

    gwb, hc_ss, hc_bg, sspar, bgpar, bad_files = _load_library_from_all_files(
        path_sims, gwb, hc_ss, hc_bg, sspar, bgpar, log,
    )
    if has_gwb:
        log.info(f"Loaded data from all library files | {holo.utils.stats(gwb)=}")
    param_samples[bad_files] = np.nan

    # ---- Save to concatenated output file ----

    log.info(f"Writing collected data to file {lib_path}")
    with h5py.File(lib_path, 'w') as h5:
        h5.create_dataset('fobs_cents', data=fobs_cents)
        h5.create_dataset('fobs_edges', data=fobs_edges)
        h5.create_dataset('sample_params', data=param_samples)
        if gwb is not None:
            h5.create_dataset('gwb', data=gwb)
        if not gwb_only:
            if has_ss:
                h5.create_dataset('hc_ss', data=hc_ss)
                h5.create_dataset('hc_bg', data=hc_bg)
            if has_params:
                h5.create_dataset('sspar', data=sspar)
                h5.create_dataset('bgpar', data=bgpar)
        h5.attrs['param_names'] = np.array(param_names).astype('S')
        # new in librarian-v1.1
        h5.attrs['parameter_space_class_name'] = pspace.name
        h5.attrs['holodeck_version'] = holo.__version__
        # I'm not sure if this can/will throw errors, but don't let the combination fail if it does.
        try:
            git_hash = holo.utils.get_git_hash()
        except:  # noqa
            git_hash = "None"
        h5.attrs['holodeck_git_hash'] = git_hash
        h5.attrs['holodeck_librarian_version'] = holo.librarian.__version__

    log.warning(f"Saved to {lib_path}, size: {holo.utils.get_file_size(lib_path)}")

    return lib_path


def _check_files_and_load_shapes(log, path_sims, nsamp):
    """Check that all `nsamp` files exist in the given path, and load info about array shapes.

    Arguments
    ---------
    path_sims : str
        Path in which individual simulation files can be found.
    nsamp : int
        Number of simulations/files that should be found.
        This should typically be loaded from the parameter-space object used to generate the library.

    Returns
    -------
    fobs : (F,) ndarray
        Observer-frame frequency bin centers at which GW signals are calculated.
    nreals : int
        Number of realizations in the output files.

    """
    fobs_edges = None
    fobs_cents = None
    nreals = None
    nloudest = None
    has_gwb = False
    has_ss = False
    has_params = False

    log.info(f"Checking {nsamp} files in {path_sims}")
    for ii in tqdm.trange(nsamp):
        temp_fname = libraries._get_sim_fname(path_sims, ii)
        if not temp_fname.exists():
            err = f"Missing at least file number {ii} out of {nsamp} files!  {temp_fname}"
            log.exception(err)
            raise ValueError(err)

        # if we've already loaded all of the necessary info, then move on to the next file
        if (fobs_cents is not None) and (nreals is not None) and (nloudest is not None):
            continue

        temp = np.load(temp_fname)
        data_keys = list(temp.keys())
        log.debug(f"{ii=} {temp_fname.name=} {data_keys=}")

        if fobs_cents is None:
            _fobs = temp.get('fobs', None)
            if _fobs is not None:
                err = "Found `fobs` in data, expected only `fobs_cents` and `fobs_edges`!"
                log.exception(err)
                raise ValueError(err)
            fobs_cents = temp['fobs_cents']
            fobs_edges = temp['fobs_edges']

        if (not has_gwb) and ('gwb' in data_keys):
            has_gwb = True

        if (not has_ss) and ('hc_ss' in data_keys):
            assert 'hc_bg' in data_keys
            has_ss = True

        if (not has_params) and ('sspar' in data_keys):
            assert 'bgpar' in data_keys
            has_params = True

        # find a file that has GWB data in it (not all of them do, if the file was a 'failure' file)
        if (nreals is None):
            nreals_1 = None
            nreals_2 = None
            if ('gwb' in data_keys):
                nreals_1 = temp['gwb'].shape[-1]
                nreals = nreals_1
            if ('hc_bg' in data_keys):
                nreals_2 = temp['hc_bg'].shape[-1]
                nreals = nreals_2
            if (nreals_1 is not None) and (nreals_2 is not None):
                assert nreals_1 == nreals_2

        if (nloudest is None) and ('hc_ss' in data_keys):
            nloudest = temp['hc_ss'].shape[-1]

    return fobs_cents, fobs_edges, nreals, nloudest, has_gwb, has_ss, has_params


def _load_library_from_all_files(path_sims, gwb, hc_ss, hc_bg, sspar, bgpar, log):
    """Load data from all individual simulation files.

    Arguments
    ---------
    path_sims : str
        Path to find individual simulation files.
    gwb : (S, F, R) ndarray
        Array in which to store GWB data from all of 'S' files.
        S: num-samples/simulations,  F: num-frequencies,  R: num-realizations.
    log : `logging.Logger`
        Logging instance.

    """
    if hc_bg is not None:
        nsamp = hc_bg.shape[0]
    elif gwb is not None:
        nsamp = gwb.shape[0]
    else:
        err = "Unable to get shape from either `hc_bg` or `gwb`!"
        log.exception(err)
        raise RuntimeError(err)

    log.info(f"Collecting data from {nsamp} files")
    bad_files = np.zeros(nsamp, dtype=bool)     #: track which files contain UN-useable data
    msg = None
    for pnum in tqdm.trange(nsamp):
        fname = libraries._get_sim_fname(path_sims, pnum)
        temp = np.load(fname, allow_pickle=True)
        # When a processor fails for a given parameter, the output file is still created with the 'fail' key added
        if ('fail' in temp):
            msg = f"file {pnum=:06d} is a failure file, setting values to NaN ({fname})"
            log.warning(msg)
            # set all parameters to NaN for failure files.  Note that this is distinct from gwb=0.0 which can be real.
            if gwb is not None:
                gwb[pnum, :, :] = np.nan
            # `hc_ss` will be set to None if `gwb_only==True`
            if hc_ss is not None:
                hc_ss[pnum, :, :, :] = np.nan
                hc_bg[pnum, :, :] = np.nan

            bad_files[pnum] = True
            continue

        # store the GWB from this file
        if gwb is not None:
            gwb[pnum, :, :] = temp['gwb'][...]
        # `hc_ss` will be set to None if `gwb_only==True`
        if (hc_ss is not None):
            hc_ss[pnum, :, :, :] = temp['hc_ss'][...]
            hc_bg[pnum, :, :] = temp['hc_bg'][...]
        # `hc_ss` will be set to None if `gwb_only==True`
        if bgpar is not None:
            sspar[pnum, :, :, :, :] = temp['sspar'][...]
            bgpar[pnum, :, :, :] = temp['bgpar'][...]

    log.info(f"{holo.utils.frac_str(bad_files)} files are failures")

    return gwb, hc_ss, hc_bg, sspar, bgpar, bad_files


if __name__ == "__main__":
    main()
