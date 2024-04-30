"""Combine output files from individual simulation runs into a single library hdf5 file.

This file can be executed as a script (see the :func:`main` function), and also provides an API
method (:func:`sam_lib_combine`) for programatically combining libraries.  When running as a script
or independent program, it must be run serially (not in parallel).

For command-line usage, run:

    python -m holodeck.librarian.combine -h

"""

import argparse
from pathlib import Path

import numpy as np
import h5py
import tqdm

import holodeck as holo
import holodeck.librarian
import holodeck.librarian.gen_lib
from holodeck.librarian import (
    libraries, DIRNAME_LIBRARY_SIMS, DIRNAME_DOMAIN_SIMS, DomainNotLibraryError
)


def main():
    """Command-line interface executable method for combining holodeck simulation files.
    """

    log = holo.log

    # ---- Make sure we're NOT running in parallel (MPI)

    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
    except ModuleNotFoundError:
        comm = None

    if (comm is not None) and (comm.rank > 0):
        err = f"Cannot run `{__file__}::main()` with multiple processors!"
        log.exception(err)
        raise RuntimeError(err)

    # ---- Setup and parse command-line arguments

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
    parser.add_argument(
        '--verbose', '-v', action='store_true', default=False,
        help="Verbose output in logger ('DEBUG' level)."
    )

    args = parser.parse_args()
    if args.verbose:
        log.setLevel(log.DEBUG)

    log.debug(f"{args=}")
    path = Path(args.path)

    # ---- Combine library files

    for library in [True, False]:
        try:
            sam_lib_combine(path, log, recreate=args.recreate, gwb_only=args.gwb, library=library)
        except DomainNotLibraryError as err:
            pass

    return


def sam_lib_combine(
    path_output, log,
    path_pspace=None, recreate=False, gwb_only=False, library=True
):
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
    library : bool,
        Combine simulations from a library, as opposed to simulations for 'domain' exploration.

    Returns
    -------
    lib_path : Path,
        Path to library output filename (typically ending with 'sam_lib.hdf5').

    """

    # ---- setup paths

    path_output = Path(path_output)
    log.info(f"Path output = {path_output}")
    if library:
        path_sims = path_output.joinpath(DIRNAME_LIBRARY_SIMS)
    else:
        path_sims = path_output.joinpath(DIRNAME_DOMAIN_SIMS)

    # ---- see if a combined library already exists

    lib_path = libraries.get_sam_lib_fname(path_output, gwb_only, library=library)
    if lib_path.exists():
        lvl = log.INFO if recreate else log.WARNING
        log.log(lvl, f"combined library already exists: {lib_path}, run with `-r` to recreate.")
        if not recreate:
            return

        log.log(lvl, "re-combining data into new file")

    # ---- load parameter space from save file

    if path_pspace is None:
        path_pspace = path_output
    pspace, pspace_fname = libraries.load_pspace_from_path(path_pspace, log=log)
    args, args_fname = holo.librarian.gen_lib.load_config_from_path(path_pspace, log=log)

    log.info(f"loaded param space: {pspace} from '{pspace_fname}'")
    param_names = pspace.param_names
    param_samples = pspace.param_samples

    # Standard Library: vary all parameters together
    if library:
        if param_samples == None:   # noqa : use `== None` to match arrays
            log.error(f"`library`={library} but `param_samples`={param_samples}`")
            err = f"`library` is True, but {path_output} looks like it's a domain."
            raise DomainNotLibraryError(err)

        nsamp_all, ndim = param_samples.shape
        log.debug(f"{nsamp_all=}, {ndim=}, {param_names=}")

    # Domain: Vary only one parameter at a time to explore the domain
    else:
        err = f"Expected 'domain' but `param_samples`={param_samples} is not `None`!"
        assert param_samples == None, err   # noqa : use `== None` to match arrays
        ndim = pspace.nparameters
        # for 'domain' simulations, this is the number of samples in each dimension
        nsamp_dim = args.nsamples
        # get the total number of samples
        nsamp_all = ndim * nsamp_dim
        # for 'domain', param_samples will eventually be shaped (ndim, nsamp_dim, ndim), but load
        # the data first as `(nsamp_all, ndim)`, then we will reshape.
        param_samples = np.zeros((nsamp_all, ndim))

    # ---- make sure all files exist; get shape information from files

    log.info(f"checking that all {nsamp_all} files exist")
    fobs_cents, fobs_edges, nreals, nloudest, has_gwb, has_ss, has_params = _check_files_and_load_shapes(
        log, path_sims, nsamp_all, library
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

    gwb = np.zeros((nsamp_all, nfreqs, nreals)) if has_gwb else None

    if (not gwb_only) and has_ss:
        hc_ss = np.zeros((nsamp_all, nfreqs, nreals, nloudest))
        hc_bg = np.zeros((nsamp_all, nfreqs, nreals))
    else:
        hc_ss = None
        hc_bg = None

    if (not gwb_only) and has_params:
        sspar = np.zeros((nsamp_all, 4, nfreqs, nreals, nloudest))
        bgpar = np.zeros((nsamp_all, 7, nfreqs, nreals))
    else:
        sspar = None
        bgpar = None

    gwb, hc_ss, hc_bg, sspar, bgpar, param_samples, bad_files = _load_library_from_all_files(
        path_sims, gwb, hc_ss, hc_bg, sspar, bgpar, param_samples, log, library
    )
    if has_gwb:
        log.info(f"Loaded data from all library files | {holo.utils.stats(gwb)=}")
    if library:
        param_samples[bad_files] = np.nan

    # ---- Save to concatenated output file ----

    log.info(f"Writing collected data to file {lib_path}")
    with h5py.File(lib_path, 'w') as h5:
        h5.create_dataset('fobs_cents', data=fobs_cents)
        h5.create_dataset('fobs_edges', data=fobs_edges)

        if not library:
            new_shape = (ndim, nsamp_dim) + param_samples.shape[1:]
            param_samples = param_samples.reshape(new_shape)
        h5.create_dataset('sample_params', data=param_samples)

        if gwb is not None:
            # if 'domain', reshape to include each dimension
            if not library:
                new_shape = (ndim, nsamp_dim) + gwb.shape[1:]
                gwb = gwb.reshape(new_shape)
            h5.create_dataset('gwb', data=gwb)
        if not gwb_only:
            if has_ss:
                # if 'domain', reshape to include each dimension
                if not library:
                    new_shape = (ndim, nsamp_dim) + hc_ss.shape[1:]
                    hc_ss = hc_ss.reshape(new_shape)
                    new_shape = (ndim, nsamp_dim) + hc_bg.shape[1:]
                    hc_bg = hc_bg.reshape(new_shape)
                h5.create_dataset('hc_ss', data=hc_ss)
                h5.create_dataset('hc_bg', data=hc_bg)
            if has_params:
                # if 'domain', reshape to include each dimension
                if not library:
                    new_shape = (ndim, nsamp_dim) + sspar.shape[1:]
                    sspar = sspar.reshape(new_shape)
                    new_shape = (ndim, nsamp_dim) + bgpar.shape[1:]
                    bgpar = bgpar.reshape(new_shape)
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

    with h5py.File(lib_path, 'r') as h5:
        assert np.all(h5['fobs_cents'][()] > 0.0)
        if has_gwb:
            log.info(f"Checking library file: {holo.utils.stats(gwb)=}")

    return lib_path


def _check_files_and_load_shapes(log, path_sims, nsamp, library):
    """Check that all `nsamp` files exist in the given path, and load info about array shapes.

    Arguments
    ---------
    log : ``logging.Logger`` instance
        Holodeck logger.
    path_sims : str
        Path in which individual simulation files can be found.
    nsamp : int
        Number of simulations/files that should be found.
        This should typically be loaded from the parameter-space object used to generate the library.
    library : bool,
        Whether this is a standard library or a 'domain' exploration.

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
    # num_fail = 0
    # num_good = 0

    log.info(f"Checking {nsamp} files in {path_sims}")
    for ii in tqdm.trange(nsamp):
        temp_fname = libraries._get_sim_fname(path_sims, ii, library=library)
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

        if 'fail' in data_keys:
            err = f"File {ii=} is a failed simulation file.  {temp_fname=}"
            log.error(err)
            log.error(f"Error in file: {temp['fail']}")
            continue

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


def _load_library_from_all_files(
    path_sims, gwb, hc_ss, hc_bg, sspar, bgpar, param_samples,
    log, library,
):
    """Load data from all individual simulation files.

    Arguments
    ---------
    path_sims : str
        Path to find individual simulation files.
    gwb : (S, F, R) ndarray
        Array in which to store GWB data from all of 'S' files.
        S: num-samples/simulations,  F: num-frequencies,  R: num-realizations.
    hc_ss
    hc_bg
    sspar
    bgpar
    param_samples
    log : ``logging.Logger``
        Logging instance.
    library : bool
        Whether this is a standard library or a domain exploration.

    """
    if hc_bg is not None:
        nsamp_all = hc_bg.shape[0]
    elif gwb is not None:
        nsamp_all = gwb.shape[0]
    else:
        err = "Unable to get shape from either `hc_bg` or `gwb`!"
        log.exception(err)
        raise RuntimeError(err)

    log.info(f"Collecting data from {nsamp_all} files")
    bad_files = np.zeros(nsamp_all, dtype=bool)     #: track which files contain UN-useable data
    msg = None
    for pnum in tqdm.trange(nsamp_all):
        fname = libraries._get_sim_fname(path_sims, pnum, library=library)
        temp = np.load(fname, allow_pickle=True)
        # When a processor fails for a given parameter, the output file is still created with the 'fail' key added
        if ('fail' in temp):
            msg = f"file {pnum=:06d} is a failure file, setting values to NaN ({fname})"
            log.info(msg)
            # set all parameters to NaN for failure files.  Note that this is distinct from gwb=0.0 which can be real.
            if gwb is not None:
                gwb[pnum, :, :] = np.nan
            # `hc_ss` will be set to None if `gwb_only==True`
            if hc_ss is not None:
                hc_ss[pnum, :, :, :] = np.nan
                hc_bg[pnum, :, :] = np.nan

            bad_files[pnum] = True
            continue

        # for 'domain' simulations, we need to load the parameters.  For 'library' runs, we
        # already have them.
        if not library:
            #! NOTE: this is just temporary until bug is fixed
            try:
                param_samples[pnum, :] = temp['params'][:]
            except:
                pvals = temp['params'][()]
                param_samples[pnum, :] = np.array([pvals[str(pn)] for pn in temp['param_names']])

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

    return gwb, hc_ss, hc_bg, sspar, bgpar, param_samples, bad_files


if __name__ == "__main__":
    main()
