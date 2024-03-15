"""Script and methods to fit simulated GWB spectra with analytic functions.

Usage
-----
For usage information, run the script with the ``-h`` or ``--help`` arguments, i.e.::

    python -m holodeck.librarian.fit_spectra -h

Typically, the only argument required is the path to the folder containing the combined library
file (``sam_lib.hdf5``).

Notes
-----
As a script, this submodule runs in parallel, with the main processor loading a holodeck library
(from a single, combined HDF5 file), and distributes simulations to the secondary processors.  The
secondary processors then perform analytic fits to the GWB spectra.  What functional forms are fit,
and using how many frequency bins, is easily adjustable.  Currently, the implemented functions are:

* A power-law, with two parameters (amplitude and spectral index),
* A power-law with turnover model, using four parameters: the normal amplitude and spectral index,
  in addition to a break frequency, and a spectral index below the break frequency.

A variety of numbers-of-frequency-bins are also fit, which are specified in the ``FITS_NBINS_PLAW``
and ``FITS_NBINS_TURN`` variables.

The methods used in this submodule are easily adaptable as API methods.

"""

import argparse
from pathlib import Path

import numpy as np
import h5py
import tqdm

import holodeck as holo
import holodeck.librarian
from holodeck.librarian import libraries, FITS_NBINS_PLAW, FITS_NBINS_TURN
from holodeck.constants import YR


def main():
    log = holo.log

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'path', default=None,
        help='library directory to run fits on; must contain the `sam_lib.hdf5` file'
    )
    parser.add_argument(
        '--recreate', '-r', action='store_true', default=False,
        help='recreate/replace existing fits file with a new one.'
    )
    parser.add_argument(
        '--all', '-a', nargs='?', const=True, default=False,
        help=(
            "recursively find all libraries within the given path, and fit them.  "
            "Optional argument is a pattern that all found paths must match, e.g. 'uniform-07'."
        )
    )

    # ---- Run sub-command

    args = parser.parse_args()
    log.debug(f"{args=}")
    path = Path(args.path)

    if args.all is not False:
        pattern = None if args.all is True else args.all
        fit_all_libraries_in_path(path, log, pattern, recreate=args.recreate)
    else:
        fit_library_spectra(path, log, recreate=args.recreate)

    log.debug("done")
    return



def fit_library_spectra(library_path, log, recreate=False):
    """Calculate analytic fits to library spectra using MPI.
    """

    # make sure MPI is working
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
    except Exception as err:
        comm = None
        holo.log.error(f"failed to load `mpi4py` in {__file__}: {err}")
        holo.log.error("`mpi4py` may not be included in the standard `requirements.txt` file")
        holo.log.error("Check if you have `mpi4py` installed, and if not, please install it")
        raise err

    # ---- setup path

    if comm.rank == 0:

        log.info(f"Fitting library from path {library_path}")

        library_path = Path(library_path)
        if library_path.is_dir():
            library_path = libraries.get_sam_lib_fname(library_path, gwb_only=False)
        if not library_path.exists() or not library_path.is_file():
            err = f"{library_path=} must point to an existing library file!"
            log.exception(err)
            raise FileNotFoundError(err)

        log.debug(f"library path = {library_path}")

        # ---- check for existing fits file

        fits_path = libraries.get_fits_path(library_path)
        return_flag = False
        if fits_path.exists():
            lvl = log.INFO if recreate else log.WARNING
            log.log(lvl, f"library fits already exists: {fits_path}")
            if recreate:
                log.log(lvl, "re-fitting data into new file")
            else:
                return_flag = True

        # ---- load library GWB and convert to PSD

        with h5py.File(library_path, 'r') as library:
            fobs = library['fobs'][()]
            psd = holo.utils.char_strain_to_psd(fobs[np.newaxis, :, np.newaxis], library['gwb'][()])

        nsamps, nfreqs, nreals = psd.shape
        log.debug(f"{nsamps=}, {nfreqs=}, {nreals=}")

        # make a copy of the `psd` in the current shape, so that we can confirm shape manipulations work later on
        psd_check = psd.copy()

        # ---- reshape PSD into (N, F) and we will split the N points across all processors

        # (S, F, R)  ==>  (S, R, F)
        psd = np.moveaxis(psd, -1, 1)
        # (S, R, F)  ==>  (S*R, F)
        psd = psd.reshape((-1, nfreqs))

        # total number of spectra that will be fit
        ntot = psd.shape[0]
        indices = range(ntot)
        indices = np.random.permutation(indices)
        indices = np.array_split(indices, comm.size)
        num_ind_per_proc = [len(ii) for ii in indices]
        log.info(f"{ntot=} cores={comm.size} || max runs per core = {np.max(num_ind_per_proc)}")

    else:
        fobs = None
        psd = None
        nsamps = None
        nfreqs = None
        nreals = None
        indices = None
        return_flag = None

    # exit if we're not recreating an existing fits file
    return_flag = comm.bcast(return_flag, root=0)
    if return_flag:
        return

    # distribute quantities to all tasks
    fobs = comm.bcast(fobs, root=0)
    psd = comm.bcast(psd, root=0)
    indices = comm.scatter(indices, root=0)
    comm.barrier()

    # select the PSD spectra for each task
    my_psd = psd[indices]
    # log.info(f"{my_psd.shape=}")

    # ---- Run fits

    nbins_plaw, fits_plaw = fit_spectra_plaw(comm, fobs, my_psd, nbins_list=FITS_NBINS_PLAW)
    nbins_turn, fits_turn = fit_spectra_turn(comm, fobs, my_psd, nbins_list=FITS_NBINS_TURN)

    # ---- gather results and save to output

    comm.barrier()
    all_indices = comm.gather(indices, root=0)
    all_fits_plaw = comm.gather(fits_plaw, root=0)
    all_fits_turn = comm.gather(fits_turn, root=0)
    all_psd = comm.gather(my_psd, root=0)

    if comm.rank == 0:

        # recombine the scatter indices so that we can sort back to the original order of PSD entries
        indices = np.concatenate(all_indices)
        # find the ordering to sort indices
        idx = np.argsort(indices)

        # re-combine all of the separate arrays, [(N1, ...), (N2, ...), ...]  ===>  (N1*N2*etc, ...)
        fits_plaw = np.concatenate(all_fits_plaw, axis=0)
        fits_turn = np.concatenate(all_fits_turn, axis=0)
        all_psd = np.concatenate(all_psd, axis=0)

        # return elements to original order, to match original GWB/PSD
        fits_plaw = fits_plaw[idx]
        fits_turn = fits_turn[idx]
        all_psd = all_psd[idx]

        # confirm that the resorting worked correctly
        assert np.all(all_psd == psd)

        # reshape arrays to convert back to (Samples, Realizations, ...)
        len_nbins_plaw = len(nbins_plaw)
        len_nbins_turn = len(nbins_turn)
        npars_plaw = np.shape(fits_plaw)[-1]
        npars_turn = np.shape(fits_turn)[-1]
        # (S*R, B, P)  ==>  (S, R, B, P)
        fits_plaw = fits_plaw.reshape(nsamps, nreals, len_nbins_plaw, npars_plaw)
        fits_turn = fits_turn.reshape(nsamps, nreals, len_nbins_turn, npars_turn)
        # (S*R, F)  ==>  (S, R, F)
        all_psd = all_psd.reshape(nsamps, nreals, nfreqs)
        # (S, R, F)  ==>  (S, F, R)
        all_psd = np.moveaxis(all_psd, 1, -1)

        # confirm that reshaping worked correctly
        assert np.all(all_psd == psd_check)

        # Report how many fits failed
        fails = np.any(~np.isfinite(fits_plaw), axis=-1)
        lvl = log.INFO if np.any(fails) else log.DEBUG
        log.log(lvl, f"Failed to fit {holo.utils.frac_str(fails)} spectra with power-law model")

        fails = np.any(~np.isfinite(fits_turn), axis=-1)
        lvl = log.INFO if np.any(fails) else log.DEBUG
        log.log(lvl, f"Failed to fit {holo.utils.frac_str(fails)} spectra with turn-over model")

        # --- Save to output file

        np.savez(
            fits_path, fobs=fobs, psd=psd, version=holo.librarian.__version__,
            nbins_plaw=nbins_plaw, fits_plaw=fits_plaw,
            nbins_turn=nbins_turn, fits_turn=fits_turn,
        )
        log.warning(f"Saved fits to {fits_path} size: {holo.utils.get_file_size(fits_path)}")

    return


def fit_all_libraries_in_path(path, log, pattern=None, recreate=False):
    """Recursively find all `sam_lib.hdf5` files in the given path, and construct spectra fits for them.
    """

    path = Path(path)
    msg = "" if pattern is None else f" that match pattern {pattern}"
    log.info(f"fitting all libraries in path {path}" + msg)
    sub_paths = _find_sam_lib_in_path_tree(path, pattern=pattern)
    log.info(f"found {len(sub_paths)} sam_lib files")
    for pp in sub_paths:
        log.info(f"path: {pp}")
        fit_library_spectra(pp, log, recreate=recreate)

    return


def _find_sam_lib_in_path_tree(path, pattern=None):
    """Recursive method to find `sam_lib.hdf5` files anywhere in the given path.
    """

    if path.is_file():
        # if a pattern is given, and it's not in this path, return nothing
        if (pattern is not None) and (pattern not in str(path)):
            return []
        # if we find the library file, return it
        if path.name == "sam_lib.hdf5":
            return [path]
        return []

    # don't recursively follow into these subdirectories
    if path.name in ['sims', 'logs']:
        return []

    # accumulate paths from all subdirectories
    path_list = []
    for pp in path.iterdir():
        path_list += _find_sam_lib_in_path_tree(pp, pattern=pattern)

    return path_list


# ==============================================================================
# ====    Fitting Functions    ====
# ==============================================================================


def _fit_spectra(comm, freqs, psd, nbins_list, fit_npars, fit_func):
    assert np.ndim(psd) == 2
    npoints, nfreqs = np.shape(psd)
    assert len(freqs) == nfreqs
    assert np.ndim(nbins_list) == 1

    bad_pars = [np.nan] * fit_npars

    def fit_if_all_finite(xx, yy):
        if np.any(~np.isfinite(yy)):
            return bad_pars

        sel = (yy > 0.0)
        if np.count_nonzero(sel) < fit_npars:
            return bad_pars

        try:
            pars = fit_func(xx[sel], yy[sel])
        except RuntimeError:
            return bad_pars

        return pars

    len_nbins = len(nbins_list)
    shape_fits = [npoints, len_nbins, fit_npars]
    fits = np.zeros(shape_fits)
    failures = 0
    iterator = tqdm.trange(npoints) if (comm is None) or (comm.rank == 0) else range(npoints)
    for ii in iterator:
        yy = psd[ii, :]
        for nn, nbin in enumerate(nbins_list):
            if nbin > nfreqs:
                raise ValueError(f"Cannot fit for {nbin=} bins, data has {nfreqs=} frequencies!")

            pars = fit_if_all_finite(freqs[:nbin], yy[:nbin])
            fits[ii, nn, :] = pars
            if not np.isfinite(pars[0]):
                failures += 1

    return fits


def fit_spectra_plaw(comm, freqs, psd, nbins_list=FITS_NBINS_PLAW):
    fit_func = lambda xx, yy: holo.utils.fit_powerlaw_psd(xx, yy, 1/YR)[0]
    fit_npars = 2
    fits = _fit_spectra(comm, freqs, psd, nbins_list, fit_npars, fit_func)
    return nbins_list, fits


def fit_spectra_turn(comm, freqs, psd, nbins_list=FITS_NBINS_TURN):
    fit_func = lambda xx, yy: holo.utils.fit_turnover_psd(xx, yy, 1/YR)[0]
    fit_npars = 4
    fits = _fit_spectra(comm, freqs, psd, nbins_list, fit_npars, fit_func)
    return nbins_list, fits


if __name__ == "__main__":
    main()
