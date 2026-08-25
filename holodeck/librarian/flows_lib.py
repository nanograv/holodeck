""" Library generation for cw-flows.

Generates a holodeck library containing ONLY what a ``cw-flows`` normalizing flow trains on: the
top-ranked continuous-wave (CW) sources of each Poisson realization, already reduced to the flow's
own columns and stored in the flow's own array layout.

This script can be run by executing::

    python -m holodeck.librarian.flows_lib <ARGS>

Run ``python -m holodeck.librarian.flows_lib -h`` for usage information.  It is parallelized with
``mpi4py`` exactly like :mod:`~holodeck.librarian.gen_lib`, and takes the same arguments except
where noted below.


How this differs from ``gen_lib``
--------------------------------
:mod:`~holodeck.librarian.gen_lib` stores ``sspar (4,F,R,L)`` and ``hc_ss (F,R,L)`` -- every one of
the ``nloudest`` sources in every one of the ``F`` frequency bins -- and leaves the ranking, the
unit conversions, and the cosmology to a downstream preprocessing pass.  That is ~5.2 MB per
simulation, of which the flow uses ~100 KB.

Here, each simulation is reduced at generation time to the ``nrank`` loudest sources of each
realization, ranked **across the whole band**, and written as ``log10_mc``, ``log10_dc``,
``log10_fo``.  Training-time preprocessing is then a broadcast and a reshape (see 'Stored layout').

Three things are deliberately *not* computed, which is where the speedup comes from:

1. ``bgpar``.  Building it requires ``dcom_final``, ``sepa``, and ``angs`` on the full
   ``(M-1, Q-1, Z-1, F)`` grid, and it is their only consumer
   (``cyutils.loudest_hc_and_par_from_sorted_redz``).  Note this is NOT a change of distance
   convention: the flow's distances come from the interpolated ``cosmo.z_to_dcom``, both here and
   in the old preprocessing path.
2. The separate ``gwb``.  ``cyutils.sam_poisson_gwb`` is a second full Poisson pass over
   ``M*Q*Z*F*R``, and it draws an *independent* realization from the one ``hc_ss``/``hc_bg`` come
   from.  See ``--save-gwb`` for the self-consistent background, which is off by default.
3. Poisson draws for dead grid cells.  Typically only ~5% of ``(M,Q,Z,F)`` cells have
   ``number > 0`` and ``h2fdf > 0`` -- most are zeroed by galaxy-merger-time stalling -- but
   ``cyutils.loudest_hc_and_par_from_sorted_redz`` draws a variate for every cell before it checks.
   :func:`loudest_per_bin` masks to the live cells first, which measured 11.6x faster than the
   cython routine at F=30, R=100 on a 61^3 grid.


Stored layout
-------------
Arrays are ``(nparam, nrank, nreal, nsamp)``, i.e. the *sample* axis is last, so that the flow's
training array is built by broadcasting the astro parameters against the CW columns::

    cw   = np.concatenate([f[k][:] for k in f.attrs['cw_keys']], axis=0)   # (3, K, R, S)
    ast  = np.broadcast_to(f['theta_ast'][:], (nastro,) + cw.shape[1:])    # (6, K, R, S)
    rows = np.concatenate([ast, cw], axis=0).reshape(nastro + 3, -1).T     # (N, 9)
    rows = rows[np.isfinite(rows).all(axis=1)]

Empty source slots (realizations holding fewer than ``nrank`` live sources) are stored as ``NaN``,
so that final ``isfinite`` mask is the only cleanup needed.

"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
import json
import shutil

import numpy as np
import h5py
import kalepy as kale
import tqdm

import holodeck as holo
from holodeck import cosmo, gravwaves
from holodeck.constants import YR, MSOL, MPC
import holodeck.librarian
from holodeck import log
from holodeck.librarian import lib_tools, gen_lib, ARGS_CONFIG_FNAME, DEF_PTA_DUR

#: maximum number of failed simulations before task terminates with error (`None`: no limit)
MAX_FAILURES = None

FILES_COPY_TO_OUTPUT = []

# ---- flows-specific defaults ------------------------------------------------

#: Number of top-ranked CW sources kept per Poisson realization.  Frozen into the library: the
#  ranking happens at generation time, so this cannot be raised without regenerating.
DEF_NUM_RANK = 20

#: Statistic used to rank sources across the band.  See :func:`cw_rank_stat`.
DEF_RANKBY = 'hc'

#: Default frequency bin width [Hz].  4x finer than the standard PTA Nyquist spacing, so that every
#  4th bin lands exactly on the real ``DEF_PTA_DUR``-year PTA grid.  A finer grid changes the prior,
#  not the data: what it buys is the ability to represent a CW *between* real PTA bins.
DEF_DF = 1.0 / (4.0 * DEF_PTA_DUR * YR)

#: Default number of frequency bins.  With ``DEF_DF`` this gives f_max = 29.65 nHz.
DEF_NUM_FBINS = 60

#: The CW columns the flow trains on, in order.  Distance, never strain amplitude: h0 is a
#  deterministic function of (Mc, d, f), so a flow over (mc, h0, fo) would fit a density on a
#  manifold the physics already pins down.  See the ``cw_mtot``/``cw_mrat``/``cw_redz`` provenance
#  arrays if you need to recover h0.
CW_KEYS = ('log10_mc', 'log10_dc', 'log10_fo')

#: Provenance arrays: not used by the flow, but they make the CW columns re-derivable and are what
#  ``h0`` would have to be reconstructed from.  ~half the library size.
PROV_KEYS = ('cw_mtot', 'cw_mrat', 'cw_redz', 'cw_hc')

#: Diagnostic index arrays, stored as int16.
IDX_KEYS = ('cw_fidx', 'cw_lidx')

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

        copy_files = FILES_COPY_TO_OUTPUT
        if (not args.resume) and (copy_files is not None):
            for fname in copy_files:
                src_file = Path(fname)
                dst_file = args.output.joinpath("runtime_" + src_file.name)
                shutil.copyfile(src_file, dst_file)
                log.info(f"Copied {fname} to {dst_file}")

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
        f"nfreqs={args.nfreqs}, df={args.df*1e9:.4f} [nHz], "
        f"nloudest={args.nloudest}, nrank={args.nrank}, rankby={args.rankby}"
    )

    # ---- distribute jobs to processors

    indices = comm.scatter(indices, root=0)
    iterator = holo.utils.tqdm(indices) if (comm.rank == 0) else np.atleast_1d(indices)

    comm.barrier()

    # ---- iterate over each processors' jobs

    beg = datetime.now()
    log.debug(f"beginning tasks at {beg}")
    failures = 0
    for sim_num in iterator:
        log.debug(f"{comm.rank=} {sim_num=}")

        params = space.param_dict(sim_num)
        msg = ", ".join([f"{kk}={vv:.4e}" for kk, vv in params.items()])
        log.debug(msg)

        rv, _sim_fname = run_cws_at_pspace_params(args, space, sim_num, params)

        if rv is False:
            failures += 1

        if (MAX_FAILURES is not None) and (failures > MAX_FAILURES):
            log.error("\n\n")
            err = f"Failed {failures} times on rank:{comm.rank}!"
            log.exception(err)
            raise RuntimeError(err)

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

    # ---- frequency grid
    #
    # NOTE: given as a bin WIDTH, not as an observing duration.  `utils.pta_freqs` derives the
    # spacing from the duration (df = 1/dur) and its `cad` argument only sets the number of bins
    # (the Nyquist cutoff), so a 4x finer grid there requires claiming a 4x longer observation.
    # Nothing downstream needs a duration -- `integrate_differential_number_3dx1d` and
    # `char_strain_sq_from_bin_edges_redz` take bin-edge arrays -- so specify the grid directly.
    parser.add_argument('--df', action='store', dest='df_nhz', type=float, default=DEF_DF*1e9,
                        help='Frequency bin width [nHz]')
    parser.add_argument('-f', '--nfreqs', action='store', dest='nfreqs', type=int,
                        help='Number of frequency bins', default=DEF_NUM_FBINS)

    # ---- what to keep
    parser.add_argument('--nrank', action='store', dest='nrank', type=int, default=DEF_NUM_RANK,
                        help='Number of top-ranked CWs kept per realization (frozen into library)')
    parser.add_argument('--rankby', action='store', dest='rankby', type=str, default=DEF_RANKBY,
                        choices=['hc', 'resid'],
                        help='Statistic used to rank CWs across the band (frozen into library)')
    parser.add_argument('--save-gwb', dest='save_gwb', default=False, action='store_true',
                        help="also store `hc_rest`, the background plus all non-selected sources")

    # how to run
    parser.add_argument('--resume', action='store_true', default=False,
                        help='resume production by loading previous parameter-space from output dir')
    parser.add_argument('--recreate', action='store_true', default=False,
                        help='recreate existing simulation files')
    parser.add_argument('--seed', action='store', type=int, default=None,
                        help='Random seed to use')
    parser.add_argument('--TEST', action='store_true', default=False,
                        help='Run in test mode (NOTE: this resets other values)')

    parser.add_argument('-v', '--verbose', metavar='LEVEL', type=int, nargs='?', const=20, default=30,
                        help='verbose output level (DEBUG=10, INFO=20, WARNING=30).')

    namespace = argparse.Namespace(**kwargs)
    args = parser.parse_args(*args, namespace=namespace)

    # ---- check / sanitize arguments

    output = Path(args.output).resolve()
    if not output.is_absolute:
        output = Path('.').resolve() / output
        output = output.resolve()

    # `gen_lib._setup_param_space` and `gen_lib._setup_log` branch on these; there is no 'domain'
    # mode and no per-simulation plotting here, but those functions are reused as-is.
    args.domain = False
    args.plot = False

    if args.nrank > args.nfreqs * args.nloudest:
        raise ValueError(
            f"`nrank`={args.nrank} exceeds the {args.nfreqs*args.nloudest} available candidates "
            f"({args.nfreqs} freqs x {args.nloudest} loudest)!"
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

    # run in test mode
    if args.TEST:
        msg = "==== WARNING: running in test mode, other settings being overridden! ===="
        print("\n" + "=" * len(msg))
        print(msg)
        print("=" * len(msg) + "\n")

        global MAX_FAILURES
        MAX_FAILURES = 0
        args.nsamples = 10
        args.nreals = 3
        args.nfreqs = 5
        args.sam_shape = (11, 12, 13)
        args.nloudest = 2
        args.nrank = 2

        if not output.name.startswith("_"):
            output = output.with_name("_" + output.name)
            print(f"WARNING: changed output to '{output}'\n")

        if args.resume:
            raise RuntimeError("Cannot use `resume` in TEST mode!")

    # convert the human-facing [nHz] to [Hz] once, here
    args.df = args.df_nhz * 1e-9

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
    """Load a previously-saved configuration.

    The ``flows_lib`` analog of :func:`holodeck.librarian.gen_lib.load_config_from_path`; it must
    round-trip through *this* module's argument parser, not ``gen_lib``'s.
    """
    fname = Path(path).joinpath(ARGS_CONFIG_FNAME)

    with open(fname, 'r') as inp:
        config = json.load(inp)

    log.info(f"Loaded configuration from {fname}")

    pop_keys = ['holodeck_version', 'holodeck_librarian_version', 'holodeck_git_hash', 'created']
    for pk in pop_keys:
        val = config.pop(pk, None)
        log.info(f"\t{pk}={val}")

    # these are derived inside `_setup_argparse`; passing them back in would shadow the derivation
    for pk in ['df', 'domain', 'plot', 'output_sims', 'output_logs']:
        config.pop(pk, None)

    pspace = config.pop('param_space')
    output = config.pop('output')

    args = _setup_argparse([pspace, output], **config)

    return args, fname


# ==============================================================================
# ====    Simulation    ====
# ==============================================================================


def flow_freqs(df, nfreqs):
    """Observed GW frequency bin centers and edges for a grid of the given bin width.

    Arguments
    ---------
    df : float
        Frequency bin width [Hz].
    nfreqs : int
        Number of frequency bins.

    Returns
    -------
    cents : (F,) ndarray
        Bin centers, ``f_i = (i+1) * df`` for ``i`` in ``[0, nfreqs)`` [Hz].
    edges : (F+1,) ndarray
        Bin edges, ``(i+0.5) * df`` [Hz].

    Notes
    -----
    Numerically identical to ``utils.pta_freqs(dur=1/df, num=nfreqs)``, but parameterized by the
    quantity that actually matters here.  Note that ``h_c^2 = h_s^2 * f/df``, so the number of
    single sources resolved per unit frequency scales with the bin width: ``nloudest`` is applied
    per bin, so a 4x finer grid separates 4x more single sources from the background.

    """
    cents = np.arange(1, nfreqs + 1) * df
    edges = (np.arange(nfreqs + 1) + 0.5) * df
    return cents, edges


def run_cws_at_pspace_params(args, space, pnum, params):
    """Run simulation number ``pnum`` of the ``space`` parameter-space and save its CW data.

    The ``flows_lib`` analog of :func:`holodeck.librarian.gen_lib.run_sam_at_pspace_params`; same
    contract, including that a failed simulation still produces an output file, containing the
    single key ``'fail'``.

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

        fobs_cents, fobs_edges = flow_freqs(args.df, args.nfreqs)
        # Offset the seed by the simulation number, so that realizations are reproducible without
        # every simulation in the library sharing one Poisson realization.
        seed = None if (args.seed is None) else (args.seed + pnum)

        data = run_cws(
            sam, hard, fobs_cents, fobs_edges,
            nreals=args.nreals, nloudest=args.nloudest, nrank=args.nrank, rankby=args.rankby,
            seed=seed, save_gwb=args.save_gwb, log=log,
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
    rankby=DEF_RANKBY,
    seed=None,
    save_gwb=False,
    log=None,
):
    """Build a binary population and return its top-ranked CW sources, in the flow's columns.

    The ``flows_lib`` analog of :func:`holodeck.librarian.lib_tools.run_model`.  The binary
    population is constructed identically; what differs is that only the CW sources are kept, and
    that they are reduced to the flow's columns here rather than downstream.

    Arguments
    ---------
    sam : :class:`holodeck.sams.sam.Semi_Analytic_Model` instance
    hard : :class:`holodeck.hardening._Hardening` subclass instance
    fobs_cents, fobs_edges : (F,), (F+1,) ndarray
        Observed GW frequency bin centers and edges [Hz], e.g. from :func:`flow_freqs`.
    nreals : int
        Number of Poisson realizations.
    nloudest : int
        Number of loudest binaries separated from the background in each frequency bin.  This is
        the candidate pool from which ``nrank`` sources are then chosen.
    nrank : int
        Number of top-ranked sources kept per realization, ranked across the whole band.
    rankby : str
        Ranking statistic, see :func:`cw_rank_stat`.
    seed : int or None
        Seed for the Poisson realizations.  Given a seed, the output is reproducible.
    save_gwb : bool
        Also return ``hc_rest``, the background plus every source not kept.
    log : ``logging.Logger`` instance

    Returns
    -------
    data : dict
        Arrays shaped ``(1, nrank, nreals)``, plus ``fobs_cents``/``fobs_edges``.  See the module
        docstring for the layout, and :func:`cw_columns` for the column definitions.

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

    # `redz_final` is defined on the (M, Q, Z) grid EDGES; take midpoints in those three dimensions
    # to land on the same (M-1, Q-1, Z-1, F) grid as `number` and `h2fdf`.
    redz = redz_final
    for dd in range(3):
        redz = np.moveaxis(redz, dd, 0)
        redz = kale.utils.midpoints(redz, axis=0)
        redz = np.moveaxis(redz, 0, dd)
    # NOTE: -1 is holodeck's sentinel for a stalled/absent binary.  It must survive to
    # `cw_columns`, which turns it into NaN rather than handing a negative redshift to the
    # cosmology.  Written as `~(redz > 0)` rather than `redz <= 0` so that NaN is caught too,
    # matching `single_sources.ss_gws_redz`.
    redz[~(redz > 0.0)] = -1.0

    mt = kale.utils.midpoints(sam.mtot)
    mr = kale.utils.midpoints(sam.mrat)

    # ---- Poisson-realize, and pull the loudest sources out of each frequency bin

    rng = np.random.default_rng(seed)
    hc2ss, bidx, hc2rest = loudest_per_bin(number, h2fdf, nreals, nloudest, rng, totals=save_gwb)

    # ---- rank across the band, and reduce to the flow's columns

    ranked = rank_cws(np.sqrt(hc2ss), bidx, fobs_cents, nrank=nrank, rankby=rankby)
    data = cw_columns(ranked, mt, mr, redz, fobs_cents)

    data['fobs_cents'] = fobs_cents
    data['fobs_edges'] = fobs_edges

    if save_gwb:
        # Everything that is NOT one of the `nrank` kept sources: the sub-threshold background,
        # plus every per-bin loudest source that did not win a global slot.  Built by ADDING the
        # unselected singles, never by subtracting the selected ones from a total -- that
        # difference cancels catastrophically in bins where the singles are most of the
        # population.
        live = ranked['fidx'] >= 0
        r_ix = np.broadcast_to(np.arange(nreals)[:, np.newaxis], live.shape)
        taken = np.zeros(hc2ss.shape, dtype=bool)                       # (F, R, L)
        taken[ranked['fidx'][live], r_ix[live], ranked['lidx'][live]] = True
        data['hc_rest'] = np.sqrt(hc2rest + np.sum(np.where(taken, 0.0, hc2ss), axis=-1))

    return data


def loudest_per_bin(number, h2fdf, nreals, nloudest, rng, totals=False):
    """Poisson-realize the population and identify the loudest sources in each frequency bin.

    Vectorized replacement for ``cyutils.loudest_hc_and_par_from_sorted_redz``, keeping only what
    the CW columns need.  It is faster for one reason: typically only ~5% of grid cells have
    ``number > 0`` and ``h2fdf > 0``, and the cython routine draws a Poisson variate for every cell
    before checking, while this masks to the live cells first.

    Within a frequency bin the live cells are sorted loudest-first and Poisson-sampled; the
    ``(l+1)``-th individual binary is then the one in the cell where the cumulative count first
    reaches ``l+1``, found by binary search.

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
    totals : bool
        Also return the sub-threshold background per (frequency, realization).

    Returns
    -------
    hc2ss : (F, R, L) ndarray
        Characteristic strain squared of each loud single source.  Zero where a bin held fewer than
        ``nloudest`` binaries.
    bidx : (F, R, L) ndarray of int
        Flat ``(M, Q, Z)`` index of the cell each single source came from; ``-1`` where empty.
    hc2rest : (F, R) ndarray or None
        Characteristic strain squared of everything EXCEPT the ``nloudest`` single sources -- the
        same quantity cython calls ``hc2bg``.  ``None`` unless ``totals`` is set.  Accumulated
        directly, not as ``total - sum(singles)``: in the sparse high-frequency bins every binary
        is extracted as a single, the true background is exactly zero, and the subtraction returns
        float noise at ~1e-18 of the total instead.

    Notes
    -----
    Three deliberate differences from ``cyutils.loudest_hc_and_par_from_sorted_redz``:

    1. Cells are sorted by ``h2fdf`` **per frequency bin**.  ``single_sources.ss_gws_redz`` sorts
       once by ``h2fdf[..., 0]`` and reuses that order for every bin, which can mis-identify the
       loudest source at high frequency.
    2. Poisson sampling is used at all counts; there is no Gaussian approximation above a
       ``normal_threshold``.  ``numpy`` samples large ``lambda`` efficiently.
    3. The generator is passed in, so realizations are reproducible.  The cython routine seeds
       ``PCG64()`` from OS entropy on every call.

    """
    M, Q, Z, F = number.shape
    R, L = nreals, nloudest
    nbins = M * Q * Z

    hc2ss = np.zeros((F, R, L))
    bidx = np.full((F, R, L), -1, dtype=np.int64)
    hc2rest = np.zeros((F, R)) if totals else None

    num_f = number.reshape(nbins, F)
    h2_f = h2fdf.reshape(nbins, F)
    ranks = np.arange(1, L + 1)

    for ff in range(F):
        # ---- keep only cells that can actually hold a source
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

        if totals:
            # Strain-squared accumulated from the END of the sorted list, so that `csum[k]` is the
            # contribution of cells k..B-1.  Summing the (tiny) quiet cells first also keeps the
            # partial sums well conditioned.
            csum = np.concatenate([np.cumsum((counts * hh)[:, ::-1], axis=1)[:, ::-1],
                                   np.zeros((R, 1))], axis=1)

        for rr in range(R):
            ntake = min(L, int(tot[rr]))
            if ntake <= 0:
                if totals:
                    hc2rest[ff, rr] = csum[rr, 0]
                continue
            # first cell whose cumulative count reaches each rank
            pos = np.searchsorted(cum[rr], ranks[:ntake], side='left')
            hc2ss[ff, rr, :ntake] = hh[pos]
            bidx[ff, rr, :ntake] = live[pos]

            if totals:
                # Everything NOT extracted, accumulated directly rather than as
                # `total - sum(extracted)`.  The subtraction catastrophically cancels wherever the
                # singles are most of the population -- the sparse high-frequency bins, where the
                # background is genuinely zero and the difference returns float noise at ~1e-18 of
                # the total.  `last` is the cell the deepest single came from; every cell after it
                # is untouched, and it keeps whatever binaries were not taken.
                last = pos[ntake - 1]
                hc2rest[ff, rr] = csum[rr, last + 1] + (cum[rr, last] - ntake) * hh[last]

    return hc2ss, bidx, hc2rest


def cw_rank_stat(hc_ss, fobs_cents, rankby=DEF_RANKBY):
    """Ranking statistic for every candidate source.

    Arguments
    ---------
    hc_ss : (F, R, L) ndarray
        Characteristic strain of each candidate source.
    fobs_cents : (F,) ndarray
        Observed GW frequency bin centers [Hz].
    rankby : str
        ``'hc'``    : characteristic strain, as stored.
        ``'resid'`` : induced timing-residual amplitude, ``h0 / (2 pi f)`` -- the quantity a CW
                      search builds its waveform from, and the geometry-averaged Fourier
                      coefficient up to a universal constant.

    Returns
    -------
    stat : (F, R, L) ndarray

    Notes
    -----
    The statistic is only ever used for ordering, so it is defined up to an arbitrary positive
    constant; the strain-convention factor relating holodeck's sky- and polarization-averaged
    ``h_s`` to a CW search's ``h0`` is such a constant and is therefore omitted.

    The ranking is deliberately NOT by signal-to-noise.  An S/N ranking would apply the sensitivity
    curve twice -- once in the prior and again in the likelihood -- and would tie the prior to one
    particular pulsar set.

    """
    if rankby == 'hc':
        return hc_ss

    if rankby == 'resid':
        ff = fobs_cents[:, np.newaxis, np.newaxis]
        df = fobs_cents[1] - fobs_cents[0]
        # h_s = h_c sqrt(df/f) recovers the orientation-averaged amplitude; /(2 pi f) turns that
        # into a timing residual.
        return hc_ss * np.sqrt(df / ff) / (2.0 * np.pi * ff)

    raise ValueError(f"Ranking by '{rankby}' is not supported!")


def rank_cws(hc_ss, bidx, fobs_cents, nrank=DEF_NUM_RANK, rankby=DEF_RANKBY):
    """Keep the top ``nrank`` sources of each realization, ranked across the whole band.

    Arguments
    ---------
    hc_ss : (F, R, L) ndarray
        Characteristic strain of each candidate source.
    bidx : (F, R, L) ndarray of int
        Flat ``(M, Q, Z)`` cell index of each candidate; ``-1`` where empty.
    fobs_cents : (F,) ndarray
        Observed GW frequency bin centers [Hz].
    nrank : int
        Number of sources to keep per realization.
    rankby : str
        Passed to :func:`cw_rank_stat`.

    Returns
    -------
    dict of (R, nrank) arrays, ordered loudest-first along the last axis:
        ``hc``   : characteristic strain of each kept source, ``0`` for an empty slot
        ``bidx`` : its flat (M, Q, Z) cell index, ``-1`` for an empty slot
        ``fidx`` : which frequency bin it came from, ``-1`` for an empty slot
        ``lidx`` : its rank within that bin, ``-1`` for an empty slot

    Notes
    -----
    Sources are ranked across the whole band, NOT per frequency bin.  ``nloudest`` sources are
    stored in every bin, so a per-bin selection would have a uniform frequency marginal by
    construction, and the flow would learn nothing about where in frequency the loud sources are.

    Uses ``argpartition`` (O(ncand)) to find the top ``nrank``, then sorts only those, instead of a
    full ``argsort`` over all candidates.

    """
    nfreq, nreal, nloud = hc_ss.shape
    ncand = nfreq * nloud
    if nrank > ncand:
        raise ValueError(f"nrank={nrank} exceeds {ncand} available candidates "
                         f"({nfreq} freqs x {nloud} loudest)")

    stat = cw_rank_stat(hc_ss, fobs_cents, rankby)
    # Empty slots must never win a slot over a real source.
    stat = np.where(bidx >= 0, stat, -np.inf)

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

    out_bidx = gather(bidx)
    # a realization with fewer than `nrank` live candidates fills the remainder with dead slots
    live = out_bidx >= 0

    return dict(
        hc=np.where(live, gather(hc_ss), 0.0),
        bidx=out_bidx,
        fidx=np.where(live, freq_idx, -1),
        lidx=np.where(live, loud_idx, -1),
    )


def cw_columns(ranked, mt, mr, redz, fobs_cents):
    """Turn ranked sources into the flow's columns, shaped ``(1, nrank, nreals)``.

    This is the step that used to live in a downstream preprocessing pass.

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
        ``log10_mc`` : log10 of the REDSHIFTED chirp mass [Msol].  Redshifted because it is what
                       pairs with the observed frequency in the waveform.
        ``log10_dc`` : log10 of the COMOVING distance [Mpc], from the source's final redshift --
                       this is the distance holodeck's strain was built from.
        ``log10_fo`` : log10 of the observed GW frequency [Hz].
        ``cw_mtot``, ``cw_mrat``, ``cw_redz``, ``cw_hc`` : provenance, in cgs.
        ``cw_fidx``, ``cw_lidx`` : int16 diagnostics.

    Notes
    -----
    Empty source slots become ``NaN`` in every float column, so that a single
    ``np.isfinite(...).all(axis=0)`` is all the cleanup a training array needs.  ``NaN`` rather
    than ``-inf`` because an ``-inf`` poisons a per-column min/max normalization.

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
        log10_mc = nan(np.log10(mchirp * (1.0 + zfin) / MSOL))
        log10_dc = nan(np.log10(dcom / MPC))
        log10_fo = nan(np.log10(fobs_cents[safe_fidx]))

    # (R, K) -> (1, K, R): parameter axis first, sample axis appended at combine time
    slab = lambda arr: np.asarray(arr).T[np.newaxis, :, :]              # noqa: E731

    return dict(
        log10_mc=slab(log10_mc),
        log10_dc=slab(log10_dc),
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
    """Combine individual simulation files into a single cw-flows library (hdf5) file.

    The ``flows_lib`` analog of :func:`holodeck.librarian.combine.sam_lib_combine`, which cannot be
    reused because it hardcodes the standard library's array names and shapes.

    Simulation files are streamed into pre-created datasets rather than accumulated in memory, so
    peak memory is flat in the number of samples.

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
    log.info(f"nfreqs={len(fobs_cents)}, {nrank=}, {nreals=}, save_gwb={args.save_gwb}")

    float_keys = list(CW_KEYS) + list(PROV_KEYS)
    idx_keys = list(IDX_KEYS)
    gwb_shape = (len(fobs_cents), nreals)

    # ---- stream all simulation files into the output file

    log.info(f"Writing collected data to file {lib_path}")
    bad_files = np.zeros(nsamp_all, dtype=bool)
    with h5py.File(lib_path, 'w') as h5:
        h5.create_dataset('fobs_cents', data=fobs_cents)
        h5.create_dataset('fobs_edges', data=fobs_edges)

        # Sample is the last and therefore fastest-varying axis, so a per-simulation write
        # `[..., pnum]` is a strided slab; one chunk per simulation makes each write land in
        # exactly one chunk.  Training reads the whole array anyway.
        dsets = {}
        for key in float_keys:
            dsets[key] = h5.create_dataset(
                key, shape=shape + (nsamp_all,), dtype='f8', chunks=shape + (1,))
        for key in idx_keys:
            dsets[key] = h5.create_dataset(
                key, shape=shape + (nsamp_all,), dtype='i2', chunks=shape + (1,))
        if args.save_gwb:
            dsets['hc_rest'] = h5.create_dataset(
                'hc_rest', shape=gwb_shape + (nsamp_all,), dtype='f8', chunks=gwb_shape + (1,))

        for ii in tqdm.trange(nsamp_all):
            temp = np.load(_get_sim_fname(path_sims, ii))
            if 'fail' in list(temp.keys()):
                bad_files[ii] = True
                for key in float_keys:
                    dsets[key][..., ii] = np.nan
                for key in idx_keys:
                    dsets[key][..., ii] = -1
                if args.save_gwb:
                    dsets['hc_rest'][..., ii] = np.nan
                continue

            for key in float_keys + idx_keys:
                dsets[key][..., ii] = temp[key]
            if args.save_gwb:
                dsets['hc_rest'][..., ii] = temp['hc_rest']

        param_samples[bad_files] = np.nan
        h5.create_dataset('sample_params', data=param_samples)
        # (nastro, 1, 1, nsamp): broadcasts against the (1, nrank, nreals, nsamp) CW columns
        h5.create_dataset('theta_ast', data=param_samples.T[:, np.newaxis, np.newaxis, :])

        h5.attrs['param_names'] = np.array(param_names).astype('S')
        h5.attrs['cw_keys'] = np.array(CW_KEYS).astype('S')
        h5.attrs['parameter_space_class_name'] = pspace.name
        h5.attrs['nrank'] = nrank
        h5.attrs['nreals'] = nreals
        h5.attrs['nloudest'] = args.nloudest
        h5.attrs['rankby'] = args.rankby
        h5.attrs['df'] = args.df
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
