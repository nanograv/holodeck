""" Library generation for cw-flows.

Generates a holodeck library containing only what an Atlas CW normalizing flow trains on: the
top-ranked continuous-wave (CW) sources of each Poisson realization and the GWB free spectra.
It's parallelized with ``mpi4py`` exactly like :mod:`~holodeck.librarian.gen_lib` but
instead of storing all ``sspar (4,nfreq,nreal,nloud)`` and ``hc_ss (nfreq,nreal,nloud)``, 
the library is reduced to the flow's columns and stored in the flow's array layout.

Stored layout
-------------
Arrays are ``(nparam, nrank, nreal, nsamp)``, i.e. the *sample* axis is last, so that the flow's
training array is built by broadcasting the astro parameters against the CW columns::

    cw   = np.concatenate([f[k][:] for k in f.attrs['cw_keys']], axis=0)   # (3, K, R, S)
    ast  = np.broadcast_to(f['theta_ast'][:], (nastro,) + cw.shape[1:])    # (6, K, R, S)
    gwb  = np.broadcast_to(f['half_log10rho'][:], (nfreqs,) + cw.shape[1:])
    ncol = nastro + len(cw_keys) + nfreqs
    rows = np.concatenate([ast, cw, gwb], axis=0).reshape(ncol, -1).T      # (N, ncol)
    rows = rows[np.isfinite(rows).all(axis=1)]

This script can be run by executing::

    python -m holodeck.librarian.flows_lib <ARGS>

Run ``python -m holodeck.librarian.flows_lib -h`` for usage information.

Notes
-----
Empty source slots (realizations holding fewer than ``nrank`` live sources) are stored as ``NaN``,
so that final ``isfinite`` mask is the only cleanup needed.  ``half_log10rho`` carries a length-1
rank axis so it broadcasts the way ``theta_ast`` does: every CW source of a realization sees that
realization's background.

The frequency grid is :func:`holodeck.utils.pta_freqs` at ``--dur`` and ``--nfreqs``: bin centers
``k/dur`` for ``k = 1..nfreqs``, bin width ``1/dur``.  The file records it in the ``fobs_cents`` and
``fobs_edges`` datasets and the ``pta_dur`` [sec] and ``df`` [Hz] attrs.

``--rank-exclude-bins`` leaves frequency bins out of the CW ranking (e.g. a bin below ``1/Tspan``
of the data the flow is for), recorded in the ``rank_exclude_bins`` attr.  ``half_log10rho`` still
covers every bin.

"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
import json

import numpy as np
import h5py
import kalepy as kale
import tqdm

import holodeck as holo
from holodeck import cosmo, gravwaves
from holodeck.constants import YR, MSOL, MPC, NWTG, SPLC
import holodeck.librarian
from holodeck import log
from holodeck.librarian import lib_tools, gen_lib, ARGS_CONFIG_FNAME, DEF_PTA_DUR, DEF_NUM_FBINS

# ---- flows-specific defaults ------------------------------------------------

#: Number of top-ranked CW sources kept per realization (fixed once the library is made).
DEF_NUM_RANK = 20

#: All CW columns stored in a library.  ``log10_dc`` is the comoving distance.
STORED_CW_KEYS = ('log10_mc', 'log10_dc', 'log10_h0', 'log10_fo')

#: The CW columns a flow trains on, recorded in the ``cw_keys`` attr.
CW_KEYS = ('log10_mc', 'log10_h0', 'log10_fo')

#: Provenance arrays: not used by the flow, but the CW columns can be re-derived from them.
PROV_KEYS = ('cw_mtot', 'cw_mrat', 'cw_redz', 'cw_hc')

#: Diagnostic index arrays (int16).
IDX_KEYS = ('cw_fidx', 'cw_lidx')

#: The GWB free spectrum, one value per frequency bin.  See :func:`gwb_free_spectrum`.
GWB_KEYS = ('half_log10rho',)

DIRNAME_FLOWS_SIMS = "flows_sims"
FNAME_FLOWS_SIM_FILE = "flows__p{pnum:06d}.npz"
FNAME_FLOWS_COMBINED_FILE = "cw-flows-library"


# ==============================================================================
# ====    Main / CLI    ====
# ==============================================================================


def main():   # noqa : ignore complexity warning
    """Parent method for generating cw-flows libraries from the command-line.

    Mirrors :func:`holodeck.librarian.gen_lib.main`, without the parameter-space 'domain' mode.
    """

    # ---- load mpi4py module

    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        log.info(f"Loaded MPI communicator: {comm.rank=} {comm.size=} {log.comm_rank=}")
    except ModuleNotFoundError as err:
        comm = None
        log.error(f"failed to load `mpi4py` in {__file__}: {err}")
        log.error("`mpi4py` may not be included in the standard `requirements.txt` file.")
        log.error("Check if you have `mpi4py` installed, and if not, please install it.")
        raise err

    # ---- setup arguments / settings, loggers, and outputs

    if comm.rank == 0:
        log.warning(f"Running {__file__} : {comm.rank=} {comm.size=} | {sys.argv=}")
        log.debug("Setting up argparse...")
        args = _setup_argparse()
    else:
        args = None

    # share `args` to all processes from rank=0
    args = comm.bcast(args, root=0)

    # setup log instance, separate for all processes
    log.debug("Setting up log...")
    gen_lib._setup_log(comm, args)

    if comm.rank == 0:

        # get parameter-space class (created new, or load previous save when `args.resume`)
        space = gen_lib._setup_param_space(args)

        # Load arguments/configuration from previous save
        if args.resume:
            args, config_fname = load_config_from_path(args.output, log)
            log.warning(f"Loaded configuration save from {config_fname}")
            # `args.resume` may be set to `False` after loading from save; reset to True
            args.resume = True
        # Save parameter space and args/configuration to output directory
        else:
            space_fname = space.save(args.output)
            log.info(f"Saved parameter space {space} to {space_fname}")

            config_fname = gen_lib._save_config(args)
            log.info(f"Saved configuration to {config_fname}")

        # ---- Split simulations for all processes

        log.info("Constructing library indices")
        indices = range(args.nsamples)
        indices = np.random.permutation(indices)
        indices = np.array_split(indices, comm.size)

        num_per = [len(ii) for ii in indices]
        log.info(f"{args.nsamples=} cores={comm.size} || max sims per core = {np.max(num_per)}")

    else:
        space = None
        indices = None

    # share parameter space across processes
    space = comm.bcast(space, root=0)

    # If we've loaded a new `args`, then share to all processes from rank=0
    if args.resume:
        args = comm.bcast(args, root=0)

    log.info(
        f"param_space={args.param_space}, parameters={space.nparameters}, samples={args.nsamples}, "
        f"sam_shape={args.sam_shape}, nreals={args.nreals}, "
        f"nfreqs={args.nfreqs}, dur={args.dur_yr} [yr], df={args.df*1e9:.4f} [nHz], "
        f"nloudest={args.nloudest}, nrank={args.nrank}, rank_exclude_bins={args.rank_exclude_bins}"
    )

    # ---- distribute jobs to processors

    indices = comm.scatter(indices, root=0)
    iterator = holo.utils.tqdm(indices) if (comm.rank == 0) else np.atleast_1d(indices)

    comm.barrier()

    # ---- iterate over each processors' jobs

    beg = datetime.now()
    log.debug(f"beginning tasks at {beg}")
    for sim_num in iterator:
        log.debug(f"{comm.rank=} {sim_num=}")

        params = space.param_dict(sim_num)
        msg = ", ".join([f"{kk}={vv:.4e}" for kk, vv in params.items()])
        log.debug(msg)

        # a failed simulation saves a 'fail' file, which the combine stores as NaN
        run_cws_at_pspace_params(args, space, sim_num, params)

    end = datetime.now()
    dur = (end - beg)
    log.info(f"\t{comm.rank} done at {str(end)} after {str(dur)} = {dur.total_seconds()}")

    # Make sure all processes are done so that all files are ready for merging
    comm.barrier()

    if (comm.rank == 0):
        log.warning("Combining simulation files into single library file")
        flows_lib_combine(args.output, log)
        log.info("Library combination completed.")

    return


def _setup_argparse(*args, **kwargs):
    """Setup the argument-parser for command-line usage."""

    parser = argparse.ArgumentParser()
    parser.add_argument('param_space', type=str,
                        help="Parameter space class name, found in 'holodeck.librarian'.")

    parser.add_argument('output', metavar='output', type=str,
                        help='output path [created if doesnt exist]')

    # basic parameters
    parser.add_argument('-n', '--nsamples', action='store', dest='nsamples', type=int, default=1000,
                        help='number of parameter space samples')
    parser.add_argument('-r', '--nreals', action='store', dest='nreals', type=int,
                        help='number of realizations', default=holo.librarian.DEF_NUM_REALS)
    parser.add_argument('-s', '--shape', action='store', dest='sam_shape', type=int,
                        help='Shape of SAM grid', default=None)
    parser.add_argument('-l', '--nloudest', action='store', dest='nloudest', type=int,
                        help='Number of loudest single sources per frequency bin',
                        default=DEF_NUM_RANK)

    # ---- frequency grid, from `holo.utils.pta_freqs(dur, nfreqs)`
    parser.add_argument('--dur', action='store', dest='dur_yr', type=float, default=DEF_PTA_DUR,
                        help='Duration [yr] of the frequency grid; bins are centered at k/dur')
    parser.add_argument('-f', '--nfreqs', action='store', dest='nfreqs', type=int,
                        help='Number of frequency bins', default=DEF_NUM_FBINS)

    # ---- what to keep
    parser.add_argument('--nrank', action='store', dest='nrank', type=int, default=DEF_NUM_RANK,
                        help='Number of top-ranked CWs kept per realization')
    parser.add_argument('--rank-exclude-bins', nargs='+', type=int, default=[], metavar='BIN',
                        dest='rank_exclude_bins',
                        help='Frequency bins (0-based) left out of the CW ranking; the GWB keeps them')

    # how to run
    parser.add_argument('--resume', action='store_true', default=False,
                        help='resume production by loading previous parameter-space from output dir')
    parser.add_argument('--recreate', action='store_true', default=False,
                        help='recreate existing simulation files')
    parser.add_argument('--seed', action='store', type=int, default=None,
                        help='Random seed to use')

    parser.add_argument('-v', '--verbose', metavar='LEVEL', type=int, nargs='?', const=20, default=30,
                        help='verbose output level (DEBUG=10, INFO=20, WARNING=30).')

    namespace = argparse.Namespace(**kwargs)
    args = parser.parse_args(*args, namespace=namespace)

    # ---- check / sanitize arguments

    output = Path(args.output).resolve()

    # expected by the `gen_lib` setup functions reused here
    args.domain = False
    args.plot = False

    # sorted and unique, so the config and the attr hold one canonical list
    args.rank_exclude_bins = sorted(set(args.rank_exclude_bins))
    outside = [bb for bb in args.rank_exclude_bins if not 0 <= bb < args.nfreqs]
    if outside:
        raise ValueError(f"`rank_exclude_bins` {outside} are outside bins 0..{args.nfreqs - 1}!")
    nranked = args.nfreqs - len(args.rank_exclude_bins)
    if nranked == 0:
        raise ValueError("`rank_exclude_bins` excludes every frequency bin!")

    if args.nrank > nranked * args.nloudest:
        raise ValueError(
            f"`nrank`={args.nrank} exceeds the {nranked*args.nloudest} available candidates "
            f"({nranked} ranked freqs x {args.nloudest} loudest)!"
        )
    if args.nrank > args.nloudest:
        log.warning(
            f"`nrank`={args.nrank} > `nloudest`={args.nloudest}: the stored pool cannot represent "
            f"a realization whose top {args.nrank} sources all fall in one frequency bin."
        )

    if args.resume:
        if not output.exists() or not output.is_dir():
            err = f"`--resume` is active but output path does not exist! '{output}'"
            raise FileNotFoundError(err)
        if args.recreate:
            raise ValueError("`resume` and `recreate` cannot both be set to True!")

    # [yr] -> [sec]; the bin width is 1/dur
    args.dur = args.dur_yr * YR
    args.df = 1.0 / args.dur

    # ---- Create output directories as needed

    output.mkdir(parents=True, exist_ok=True)
    args.output = output

    output_sims = output.joinpath(DIRNAME_FLOWS_SIMS)
    output_sims.mkdir(parents=True, exist_ok=True)
    args.output_sims = output_sims

    output_logs = output.joinpath("logs")
    output_logs.mkdir(parents=True, exist_ok=True)
    args.output_logs = output_logs

    return args


def load_config_from_path(path, log):
    """Load a saved configuration through this module's argument parser.

    Like :func:`holodeck.librarian.gen_lib.load_config_from_path`.
    """
    fname = Path(path).joinpath(ARGS_CONFIG_FNAME)

    with open(fname, 'r') as inp:
        config = json.load(inp)

    log.info(f"Loaded configuration from {fname}")

    # runs made on the old `nsub` grid can't be resumed or combined by this version
    if config.get('nsub', 1) != 1:
        raise ValueError(f"{fname} has nsub={config['nsub']}, a grid this version cannot reproduce")

    pop_keys = ['holodeck_version', 'holodeck_librarian_version', 'holodeck_git_hash', 'created']
    for pk in pop_keys:
        val = config.pop(pk, None)
        log.info(f"\t{pk}={val}")

    # derived in `_setup_argparse`, so not passed back in
    for pk in ['df', 'dur', 'domain', 'plot', 'output_sims', 'output_logs']:
        config.pop(pk, None)

    pspace = config.pop('param_space')
    output = config.pop('output')

    args = _setup_argparse([pspace, output], **config)

    return args, fname


# ==============================================================================
# ====    Simulation    ====
# ==============================================================================


def gwb_free_spectrum(hc, fobs_cents, df):
    """Convert characteristic strain per frequency bin into the PTA free-spectrum parameter.

    ``rho_i^2 = h_c,i^2 df / (12 pi^2 f_i^3)`` is the power in bin ``i``, in the enterprise/pandora
    convention.

    Arguments
    ---------
    hc : (F, ...) ndarray
        Characteristic strain per bin.
    fobs_cents : (F,) ndarray
        Bin centers [Hz].
    df : float
        Bin width [Hz], which ``rho`` is defined against.  For a library this is ``1/dur``, which
        can be finer than ``1/Tspan`` of the data the prior is used on.

    Returns
    -------
    half_log10rho : (F, ...) ndarray
        ``log10(rho_i)``, or ``NaN`` for a bin with no binaries.

    """
    ff = fobs_cents.reshape((-1,) + (1,) * (hc.ndim - 1))
    rho2 = hc**2 * df / (12.0 * np.pi**2 * ff**3)
    # an empty bin is NaN, so the training array's `isfinite` mask drops it
    return np.where(rho2 > 0.0, 0.5 * np.log10(np.where(rho2 > 0.0, rho2, 1.0)), np.nan)


def run_cws_at_pspace_params(args, space, pnum, params):
    """Run simulation ``pnum`` of the parameter space and save it.

    Like :func:`holodeck.librarian.gen_lib.run_sam_at_pspace_params`: a failed simulation still
    saves a file, holding the single key ``'fail'``.

    Returns
    -------
    rv : bool
        ``True`` if this simulation was successfully run, ``False`` otherwise.
    sim_fname : ``pathlib.Path``
        Path of the simulation save file.

    """

    sim_fname = _get_sim_fname(args.output_sims, pnum)

    beg = datetime.now()
    log.info(f"{pnum=} :: {params=} beginning at {beg}")
    log.info(f"file exists: {sim_fname.is_file()} | '{sim_fname}'")

    if sim_fname.exists():
        log.info(f"Sim file already exists, {args.recreate=} | '{sim_fname}'")
        data = np.load(sim_fname)
        data_keys = list(data.keys())

        if 'fail' in data_keys:
            log.info(f"Existing file was a failure, re-attempting... ({data_keys=})")

        elif not args.recreate:
            # Make sure parameters are consistent with expectations
            params_array = np.array([params[pn] for pn in space.param_names])
            file_params = data['params']
            file_param_names = data['param_names']
            if not np.all([fpn == pn for fpn, pn in zip(file_param_names, space.param_names)]):
                err = f"Mismatch between space param names and loaded parameter names!  {sim_fname=}"
                log.exception(err)
                log.exception(f"{space.param_names=}")
                log.exception(f"{file_param_names=}")
                raise RuntimeError(err)

            if not np.allclose(file_params, params_array):
                err = f"Mismatch between space params and loaded params!  {sim_fname=}"
                log.exception(err)
                raise RuntimeError(err)

            return True, sim_fname

    # ---- run Model

    try:
        log.debug("Selecting `sam` and `hard` instances")
        sam, hard = space.model_for_params(params)

        fobs_cents, fobs_edges = holo.utils.pta_freqs(dur=args.dur, num=args.nfreqs)
        # offset by the sample number, so each simulation gets its own reproducible realizations
        seed = None if (args.seed is None) else (args.seed + pnum)

        data = run_cws(
            sam, hard, fobs_cents, fobs_edges,
            nreals=args.nreals, nloudest=args.nloudest, nrank=args.nrank,
            rank_exclude_bins=args.rank_exclude_bins, seed=seed, log=log,
        )
        data['params'] = np.array([params[pn] for pn in space.param_names])
        data['param_names'] = space.param_names
        rv = True
        log.debug("Completed model successfully.")

    except Exception as err:
        log.exception(f"`run_cws` FAILED on {pnum=} with {params=}")
        log.exception(err)
        rv = False
        data = dict(fail=str(err))

    # ---- save data to file

    log.debug(f"Saving {pnum} to file; data has keys: {list(data.keys())}")
    np.savez(sim_fname, **data)
    log.info(f"Saved to {sim_fname}, size {holo.utils.get_file_size(sim_fname)} "
             f"after {(datetime.now()-beg)}")

    return rv, sim_fname


def run_cws(
    sam, hard, fobs_cents, fobs_edges,
    nreals=holo.librarian.DEF_NUM_REALS,
    nloudest=DEF_NUM_RANK,
    nrank=DEF_NUM_RANK,
    rank_exclude_bins=(),
    seed=None,
    log=None,
):
    """Build a binary population and return its top-ranked CWs and GWB in the flow's columns.

    The population is built as in :func:`holodeck.librarian.lib_tools.run_model`.

    Arguments
    ---------
    sam : :class:`holodeck.sams.sam.Semi_Analytic_Model` instance
    hard : :class:`holodeck.hardening._Hardening` subclass instance
    fobs_cents, fobs_edges : (F,), (F+1,) ndarray
        Observed GW frequency bin centers and edges [Hz], e.g. from :func:`holodeck.utils.pta_freqs`.
    nreals : int
        Number of Poisson realizations.
    nloudest : int
        Number of loudest binaries kept in each frequency bin, the pool ``nrank`` is chosen from.
    nrank : int
        Number of sources kept per realization, ranked across the whole band.
    rank_exclude_bins : sequence of int
        Frequency bins left out of the ranking.  Their sources still count in ``half_log10rho``.
    seed : int or None
        Seed for the Poisson realizations.
    log : ``logging.Logger`` instance

    Returns
    -------
    data : dict
        The :func:`cw_columns` arrays, shaped ``(1, nrank, nreals)``; ``fobs_cents`` and
        ``fobs_edges``; and ``half_log10rho``, shaped ``(nfreqs, nreals)``.

    """

    from holodeck.sams import sam_cyutils

    if not isinstance(hard, (holo.hardening.Fixed_Time_2PL_SAM,
                             holo.hardening.FixedOuterTime_InnerPL_SAM,
                             holo.hardening.Hard_GW)):
        err = (
            f"`sam_cyutils` methods only work with `Fixed_Time_2PL_SAM`, "
            f"`FixedOuterTime_InnerPL_SAM`, or `Hard_GW` hardening models!  Not {hard}!"
        )
        if log is not None:
            log.exception(err)
        raise RuntimeError(err)

    # ---- construct the binary population  (identical to `lib_tools.run_model`)

    # convert from GW to orbital frequencies
    fobs_orb_cents = fobs_cents / 2.0
    fobs_orb_edges = fobs_edges / 2.0

    redz_final, diff_num = sam_cyutils.dynamic_binary_number_at_fobs(
        fobs_orb_cents, sam, hard, cosmo)
    edges = [sam.mtot, sam.mrat, sam.redz, fobs_orb_edges]
    number = sam_cyutils.integrate_differential_number_3dx1d(edges, diff_num)

    # h2fdf = hs^2 * f/df, i.e. the characteristic strain squared of a single source in each bin
    h2fdf = gravwaves.char_strain_sq_from_bin_edges_redz(edges, redz_final)

    # ---- bin-center the redshifts  (identical to `single_sources.ss_gws_redz`)

    # `redz_final` is on the (M, Q, Z) grid edges; midpoints put it on the grid of `number`
    redz = redz_final
    for dd in range(3):
        redz = np.moveaxis(redz, dd, 0)
        redz = kale.utils.midpoints(redz, axis=0)
        redz = np.moveaxis(redz, 0, dd)
    # -1 marks a stalled binary (`cw_columns` turns it into NaN); `~(redz > 0)` also catches NaN
    redz[~(redz > 0.0)] = -1.0

    mt = kale.utils.midpoints(sam.mtot)
    mr = kale.utils.midpoints(sam.mrat)

    # ---- Poisson-realize, and pull the loudest sources out of each frequency bin

    rng = np.random.default_rng(seed)
    hc2ss, bidx, hc2rest = loudest_per_bin(number, h2fdf, nreals, nloudest, rng)

    # ---- rank across the band, and reduce to the flow's columns

    ranked = rank_cws(np.sqrt(hc2ss), bidx, nrank=nrank, exclude_bins=rank_exclude_bins)
    data = cw_columns(ranked, mt, mr, redz, fobs_cents)

    data['fobs_cents'] = fobs_cents
    data['fobs_edges'] = fobs_edges

    # total power per bin, kept CWs included, as a real free-spectrum analysis would see it
    hc_total = np.sqrt(hc2rest + np.sum(hc2ss, axis=-1))            # (F, R)
    data['half_log10rho'] = gwb_free_spectrum(hc_total, fobs_cents, fobs_edges[1] - fobs_edges[0])

    return data


def loudest_per_bin(number, h2fdf, nreals, nloudest, rng):
    """Poisson-realize the population and find the loudest sources in each frequency bin.

    A faster replacement for ``cyutils.loudest_hc_and_par_from_sorted_redz``.  Counts are drawn only
    for cells with ``number > 0`` and ``h2fdf > 0``, since no other cell can hold a source with any
    strain.  Within each frequency bin the cells are sorted loudest-first, and the ``l``-th loudest
    binary is in the cell where the cumulative count first reaches ``l``.

    Arguments
    ---------
    number : (M, Q, Z, F) ndarray
        Expected number of binaries in each bin.
    h2fdf : (M, Q, Z, F) ndarray
        Characteristic strain squared of a single source in each bin.
    nreals : int
        Number of Poisson realizations.
    nloudest : int
        Number of loudest sources to separate in each frequency bin.
    rng : ``numpy.random.Generator``

    Returns
    -------
    hc2ss : (F, R, L) ndarray
        Characteristic strain squared of each loud source; zero where a bin held fewer than
        ``nloudest`` binaries.
    bidx : (F, R, L) ndarray of int
        Flat ``(M, Q, Z)`` index of the cell each source came from; ``-1`` where empty.
    hc2rest : (F, R) ndarray
        Characteristic strain squared of everything except the loud sources (cython's ``hc2bg``).

    Notes
    -----
    Unlike the cython routine, cells are sorted per frequency bin rather than once by the first
    bin, Poisson draws are used at all counts, and the generator is passed in so results are
    reproducible.

    """
    M, Q, Z, F = number.shape
    R, L = nreals, nloudest
    nbins = M * Q * Z

    hc2ss = np.zeros((F, R, L))
    bidx = np.full((F, R, L), -1, dtype=np.int64)
    hc2rest = np.zeros((F, R))

    num_f = number.reshape(nbins, F)
    h2_f = h2fdf.reshape(nbins, F)
    ranks = np.arange(1, L + 1)

    for ff in range(F):
        # ---- keep only cells that can hold a source with nonzero strain
        live = np.nonzero((num_f[:, ff] > 0.0) & (h2_f[:, ff] > 0.0))[0]
        if live.size == 0:
            continue

        # ---- sort loudest-first, so that "the l-th binary" means "the l-th loudest"
        nn = num_f[live, ff]
        hh = h2_f[live, ff]
        order = np.argsort(-hh)
        live = live[order]
        nn = nn[order]
        hh = hh[order]

        # ---- Poisson realizations.  (R, B) so that each realization is contiguous in memory.
        counts = rng.poisson(nn, size=(R, live.size))
        cum = np.cumsum(counts, axis=1)
        tot = cum[:, -1]

        # strain^2 summed from the quiet end, so `csum[k]` is the total of cells k..B-1
        csum = np.concatenate([np.cumsum((counts * hh)[:, ::-1], axis=1)[:, ::-1],
                               np.zeros((R, 1))], axis=1)

        for rr in range(R):
            ntake = min(L, int(tot[rr]))
            if ntake <= 0:
                hc2rest[ff, rr] = csum[rr, 0]
                continue
            # first cell whose cumulative count reaches each rank
            pos = np.searchsorted(cum[rr], ranks[:ntake], side='left')
            hc2ss[ff, rr, :ntake] = hh[pos]
            bidx[ff, rr, :ntake] = live[pos]

            # The rest, summed directly: `total - sum(singles)` leaves float noise where the
            # background is really zero.  `last` is the deepest single's cell; it keeps the
            # binaries not taken, and every cell after it is untouched.
            last = pos[ntake - 1]
            hc2rest[ff, rr] = csum[rr, last + 1] + (cum[rr, last] - ntake) * hh[last]

    return hc2ss, bidx, hc2rest


def rank_cws(hc_ss, bidx, nrank=DEF_NUM_RANK, exclude_bins=()):
    """Keep the top ``nrank`` sources of each realization, ranked across the whole band.

    Arguments
    ---------
    hc_ss : (F, R, L) ndarray
        Characteristic strain of each candidate source.
    bidx : (F, R, L) ndarray of int
        Flat ``(M, Q, Z)`` cell index of each candidate; ``-1`` where empty.
    nrank : int
        Number of sources to keep per realization.
    exclude_bins : sequence of int
        Frequency bins whose candidates are left out of the ranking, as if their slots were empty.

    Returns
    -------
    dict of (R, nrank) arrays, ordered loudest-first along the last axis:
        ``hc``   : characteristic strain of each kept source, ``0`` for an empty slot
        ``bidx`` : its flat (M, Q, Z) cell index, ``-1`` for an empty slot
        ``fidx`` : which frequency bin it came from, ``-1`` for an empty slot
        ``lidx`` : its rank within that bin, ``-1`` for an empty slot

    Notes
    -----
    Ranking is across the whole band, so the flow learns where in frequency the loud sources are.
    It is by characteristic strain rather than S/N, which would put the PTA's sensitivity into
    both the prior and the likelihood.

    """
    nfreq, nreal, nloud = hc_ss.shape
    ncand = nfreq * nloud
    if nrank > ncand:
        raise ValueError(f"nrank={nrank} exceeds {ncand} available candidates "
                         f"({nfreq} freqs x {nloud} loudest)")

    exclude_bins = np.asarray(exclude_bins, dtype=int)
    if np.any((exclude_bins < 0) | (exclude_bins >= nfreq)):
        raise ValueError(f"exclude_bins={exclude_bins.tolist()} are outside bins 0..{nfreq - 1}")

    # a candidate competes if its slot holds a source and its bin is ranked
    ok = bidx >= 0
    ok[exclude_bins] = False

    # empty slots and excluded bins rank last
    stat = np.where(ok, hc_ss, -np.inf)

    # (F, R, L) -> (R, F*L) so all candidates in a realization share one axis.  Flat candidate
    # index c encodes (freq, loudest) as c = f*nloud + l.
    stat_c = np.moveaxis(stat, 0, 1).reshape(nreal, ncand)

    # top-nrank without sorting everything, then sort just the survivors
    part = np.argpartition(-stat_c, nrank - 1, axis=-1)[..., :nrank]
    order = np.argsort(-np.take_along_axis(stat_c, part, axis=-1), axis=-1)
    idx = np.take_along_axis(part, order, axis=-1)          # (R, nrank)

    freq_idx = idx // nloud
    loud_idx = idx % nloud

    # Gather straight from the (F, R, L) arrays by broadcasting the index arrays.
    r_ix = np.arange(nreal)[:, np.newaxis]
    gather = lambda arr: arr[freq_idx, r_ix, loud_idx]      # noqa: E731

    # A realization with fewer than `nrank` live candidates fills the remainder with dead slots.
    # Liveness comes from `ok`, not `bidx`, so those slots are never filled from excluded bins.
    live = gather(ok)

    return dict(
        hc=np.where(live, gather(hc_ss), 0.0),
        bidx=np.where(live, gather(bidx), -1),
        fidx=np.where(live, freq_idx, -1),
        lidx=np.where(live, loud_idx, -1),
    )


#: Geometrized units.
_TSUN = NWTG * MSOL / SPLC ** 3       # G*Msol/c^3 [s]
_MPC2S = MPC / SPLC                   # Mpc in light-seconds [s/Mpc]


def strain_amplitude_h0(mc_msol, dlum_mpc, fobs):
    r"""CW strain amplitude ``h0``, in the convention PTA CW samplers (QuickCW, ATLAS) use.

    .. math::
        h_0 = \frac{2 (G \mathcal{M}_c)^{5/3} (\pi f_\mathrm{gw})^{2/3}}{c^4 D_L}

    :func:`holodeck.utils.gw_strain_source`, and so ``cw_hc``, uses the sky- and
    polarization-averaged amplitude instead, which is larger by ``sqrt(8/5)``.

    Arguments
    ---------
    mc_msol : array_like
        Redshifted chirp mass [Msol].
    dlum_mpc : array_like
        Luminosity distance [Mpc] (``log10_dc`` is comoving; multiply by ``1+z``).
    fobs : array_like
        Observed GW frequency [Hz].

    Returns
    -------
    h0 : array_like
        Dimensionless strain amplitude.

    """
    h0 = (2.0 * (mc_msol * _TSUN) ** (5.0 / 3.0) * (np.pi * fobs) ** (2.0 / 3.0)
          / (dlum_mpc * _MPC2S))
    return h0


def cw_columns(ranked, mt, mr, redz, fobs_cents):
    """Turn ranked sources into the flow's columns, shaped ``(1, nrank, nreals)``.

    Arguments
    ---------
    ranked : dict
        Output of :func:`rank_cws`; arrays shaped ``(R, nrank)``.
    mt, mr : (M-1,), (Q-1,) ndarray
        Total-mass [g] and mass-ratio grid bin centers.
    redz : (M-1, Q-1, Z-1, F) ndarray
        Bin-centered final redshifts, with ``-1`` marking stalled/absent binaries.
    fobs_cents : (F,) ndarray
        Observed GW frequency bin centers [Hz].

    Returns
    -------
    data : dict of (1, nrank, nreals) arrays
        ``log10_mc`` : redshifted chirp mass [Msol]
        ``log10_dc`` : comoving distance [Mpc]
        ``log10_h0`` : strain amplitude, samplers' convention (:func:`strain_amplitude_h0`)
        ``log10_fo`` : observed GW frequency [Hz]
        ``cw_mtot``, ``cw_mrat``, ``cw_redz``, ``cw_hc`` : provenance, in cgs
        ``cw_fidx``, ``cw_lidx`` : int16 diagnostics

    Empty source slots are ``NaN`` in every float column, so one ``isfinite`` mask cleans a
    training array.

    """
    fidx = ranked['fidx']
    live = fidx >= 0

    # unravel the flat (M, Q, Z) cell index back into per-source grid coordinates
    safe_bidx = np.where(live, ranked['bidx'], 0)
    safe_fidx = np.where(live, fidx, 0)
    mm, qq, _zz = np.unravel_index(safe_bidx, redz.shape[:3])

    mtot = mt[mm]
    mrat = mr[qq]
    redz_flat = redz.reshape(-1, redz.shape[3])
    zfin = redz_flat[safe_bidx, safe_fidx]

    # a stalled cell carries the -1 sentinel and has no distance; treat it as an empty slot
    live = live & (zfin > 0.0)

    mchirp = mtot * mrat**(3.0/5.0) / (1.0 + mrat)**(6.0/5.0)          # [g]

    nan = lambda arr: np.where(live, arr, np.nan)                       # noqa: E731
    with np.errstate(divide='ignore', invalid='ignore'):
        dcom = np.asarray(cosmo.z_to_dcom(np.where(live, zfin, 1.0)))
        mc_msol = mchirp * (1.0 + zfin) / MSOL          # REDSHIFTED
        dlum_mpc = (1.0 + zfin) * dcom / MPC            # D_L = (1+z) D_c
        fobs = fobs_cents[safe_fidx]
        log10_mc = nan(np.log10(mc_msol))
        log10_dc = nan(np.log10(dcom / MPC))
        log10_h0 = nan(np.log10(strain_amplitude_h0(mc_msol, dlum_mpc, fobs)))
        log10_fo = nan(np.log10(fobs))

    # (R, K) -> (1, K, R): parameter axis first, sample axis appended at combine time
    slab = lambda arr: np.asarray(arr).T[np.newaxis, :, :]              # noqa: E731

    return dict(
        log10_mc=slab(log10_mc),
        log10_dc=slab(log10_dc),
        log10_h0=slab(log10_h0),
        log10_fo=slab(log10_fo),
        cw_mtot=slab(nan(mtot)),
        cw_mrat=slab(nan(mrat)),
        cw_redz=slab(nan(zfin)),
        cw_hc=slab(nan(ranked['hc'])),
        cw_fidx=slab(np.where(live, fidx, -1)).astype(np.int16),
        cw_lidx=slab(np.where(live, ranked['lidx'], -1)).astype(np.int16),
    )


# ==============================================================================
# ====    Combination    ====
# ==============================================================================


def _get_sim_fname(path, pnum):
    return Path(path).joinpath(FNAME_FLOWS_SIM_FILE.format(pnum=pnum))


def get_flows_lib_fname(path):
    return Path(path).joinpath(FNAME_FLOWS_COMBINED_FILE).with_suffix(".hdf5")


def flows_lib_combine(path_output, log, recreate=False):
    """Combine the simulation files into one library (hdf5) file.

    Like :func:`holodeck.librarian.combine.sam_lib_combine`, for this module's arrays.  Files are
    streamed into the hdf5 file, so memory does not grow with the number of samples.

    Arguments
    ---------
    path_output : str or Path
        Library directory; must contain the ``flows_sims`` subdirectory.
    log : ``logging.Logger``
    recreate : bool
        Replace an existing combined file.

    Returns
    -------
    lib_path : Path

    """
    path_output = Path(path_output)
    log.info(f"Path output = {path_output}")
    path_sims = path_output.joinpath(DIRNAME_FLOWS_SIMS)

    lib_path = get_flows_lib_fname(path_output)
    if lib_path.exists():
        lvl = log.INFO if recreate else log.WARNING
        log.log(lvl, f"Combined library already exists: {lib_path}, run with `-r` to recreate.")
        if not recreate:
            return lib_path
        log.log(lvl, "re-combining data into new file")

    # ---- load parameter space and configuration from save files

    pspace, pspace_fname = lib_tools.load_pspace_from_path(path_output)
    args, args_fname = load_config_from_path(path_output, log)
    log.info(f"loaded param space: {pspace} from '{pspace_fname}'")

    param_names = pspace.param_names
    param_samples = pspace.param_samples[()]
    nsamp_all, ndim = param_samples.shape
    log.debug(f"{nsamp_all=}, {ndim=}, {param_names=}")

    # ---- check that all files exist, and get array shapes from the first good one

    log.info(f"Checking {nsamp_all} files in {path_sims}")
    fobs_cents = None
    fobs_edges = None
    shape = None
    for ii in tqdm.trange(nsamp_all):
        temp_fname = _get_sim_fname(path_sims, ii)
        if not temp_fname.exists():
            err = f"Missing at least file number {ii} out of {nsamp_all} files!  {temp_fname}"
            log.exception(err)
            raise ValueError(err)
        if shape is not None:
            continue
        temp = np.load(temp_fname)
        if 'fail' in list(temp.keys()):
            log.error(f"File {ii=} is a failed simulation file: {temp['fail']}")
            continue
        fobs_cents = temp['fobs_cents']
        fobs_edges = temp['fobs_edges']
        shape = temp['log10_mc'].shape           # (1, nrank, nreals)

    if shape is None:
        err = f"Every one of the {nsamp_all} simulation files is a failure!"
        log.exception(err)
        raise RuntimeError(err)

    nrank, nreals = shape[1], shape[2]
    log.info(f"nfreqs={len(fobs_cents)}, {nrank=}, {nreals=}")

    float_keys = list(STORED_CW_KEYS) + list(PROV_KEYS)
    idx_keys = list(IDX_KEYS)
    gwb_shape = (len(fobs_cents), 1, nreals)

    # ---- stream all simulation files into the output file

    log.info(f"Writing collected data to file {lib_path}")
    bad_files = np.zeros(nsamp_all, dtype=bool)
    with h5py.File(lib_path, 'w') as h5:
        h5.create_dataset('fobs_cents', data=fobs_cents)
        h5.create_dataset('fobs_edges', data=fobs_edges)

        # one chunk per simulation, so each write touches one chunk
        dsets = {}
        for key in float_keys:
            dsets[key] = h5.create_dataset(
                key, shape=shape + (nsamp_all,), dtype='f8', chunks=shape + (1,))
        for key in idx_keys:
            dsets[key] = h5.create_dataset(
                key, shape=shape + (nsamp_all,), dtype='i2', chunks=shape + (1,))
        for key in GWB_KEYS:
            dsets[key] = h5.create_dataset(
                key, shape=gwb_shape + (nsamp_all,), dtype='f8', chunks=gwb_shape + (1,))

        for ii in tqdm.trange(nsamp_all):
            temp = np.load(_get_sim_fname(path_sims, ii))
            if 'fail' in list(temp.keys()):
                bad_files[ii] = True
                for key in float_keys + list(GWB_KEYS):
                    dsets[key][..., ii] = np.nan
                for key in idx_keys:
                    dsets[key][..., ii] = -1
                continue

            for key in float_keys + idx_keys:
                dsets[key][..., ii] = temp[key]
            for key in GWB_KEYS:
                dsets[key][..., ii] = temp[key][:, np.newaxis, :]

        param_samples[bad_files] = np.nan
        h5.create_dataset('sample_params', data=param_samples)
        # (nastro, 1, 1, nsamp): broadcasts against the (1, nrank, nreals, nsamp) CW columns
        h5.create_dataset('theta_ast', data=param_samples.T[:, np.newaxis, np.newaxis, :])

        h5.attrs['param_names'] = np.array(param_names).astype('S')
        h5.attrs['cw_keys'] = np.array(CW_KEYS).astype('S')
        h5.attrs['gwb_keys'] = np.array(GWB_KEYS).astype('S')
        h5.attrs['parameter_space_class_name'] = pspace.name
        h5.attrs['nrank'] = nrank
        h5.attrs['nreals'] = nreals
        h5.attrs['nloudest'] = args.nloudest
        # always written, empty when every bin is ranked; older libraries have no such attr
        h5.attrs['rank_exclude_bins'] = np.array(args.rank_exclude_bins, dtype=np.int16)
        # ranking is always by hc; recorded for readers of older libraries
        h5.attrs['rankby'] = 'hc'
        h5.attrs['df'] = args.df
        h5.attrs['pta_dur'] = args.dur
        h5.attrs['seed'] = -1 if (args.seed is None) else args.seed
        h5.attrs['holodeck_version'] = holo.__version__
        try:
            git_hash = holo.utils.get_git_hash()
        except:  # noqa
            git_hash = "None"
        h5.attrs['holodeck_git_hash'] = git_hash
        h5.attrs['holodeck_librarian_version'] = holo.librarian.__version__

    nbad = int(bad_files.sum())
    log.warning(f"Saved to {lib_path}, size: {holo.utils.get_file_size(lib_path)}")
    log.warning(f"{nbad}/{nsamp_all} simulations failed and are stored as NaN")

    return lib_path


if __name__ == "__main__":
    holo.set_log_level(holo.log.WARNING)
    main()
