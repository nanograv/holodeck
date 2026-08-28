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
realization, ranked **across the whole band**, and written as ``log10_mc``, ``log10_fo``, and both
amplitude parameterizations, ``log10_h0`` and ``log10_dc``.  Training-time preprocessing is then a
broadcast and a reshape (see 'Stored layout').

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
    gwb  = np.broadcast_to(f['half_log10rho'][:], (nfreqs,) + cw.shape[1:])
    ncol = nastro + len(cw_keys) + nfreqs
    rows = np.concatenate([ast, cw, gwb], axis=0).reshape(ncol, -1).T      # (N, ncol)
    rows = rows[np.isfinite(rows).all(axis=1)]

Empty source slots (realizations holding fewer than ``nrank`` live sources) are stored as ``NaN``,
so that final ``isfinite`` mask is the only cleanup needed.  ``half_log10rho`` carries a length-1
rank axis so it broadcasts the way ``theta_ast`` does: every CW source of a realization sees that
realization's background.

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
from holodeck.constants import YR, MSOL, MPC, NWTG, SPLC
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

#: Number of frequency sub-bins per PTA Fourier bin.  Must be an INTEGER: the grid is anchored at
#  ``1/T`` (see :func:`flow_freqs`), so the real PTA frequency ``k/T`` is bin index ``(k-1)*nsub``
#  exactly, with no tolerance check -- which is what lets the CW columns and a PTA-convention GWB
#  free spectrum index the same grid.  A finer grid changes the prior, not the data: what it buys
#  is the ability to represent a CW *between* real PTA bins.
DEF_NSUB = 4

#: Highest PTA harmonic on the default grid: the band runs from ``1/T`` to ``DEF_KMAX/T``.
DEF_KMAX = 15

#: Default number of frequency bins; f_max = 29.65 nHz over 57 bins at ``DEF_PTA_DUR``.
DEF_NUM_FBINS = DEF_NSUB * (DEF_KMAX - 1) + 1

#: All CW columns stored in a library; the ``cw_keys`` attr says which triple a flow trains on.
STORED_CW_KEYS = ('log10_mc', 'log10_dc', 'log10_h0', 'log10_fo')

#: The two interchangeable CW triples.  Converting between them needs ``cw_redz``: ``log10_dc`` is
#  COMOVING, while ``log10_mc`` is REDSHIFTED and ``log10_fo`` OBSERVED.
CW_KEYS_DCOM = ('log10_mc', 'log10_dc', 'log10_fo')
CW_KEYS_STRAIN = ('log10_mc', 'log10_h0', 'log10_fo')

#: Declared by default: strain, the parameterization QuickCW and ATLAS sample in.  ``--cw-keys
#  dcom`` declares distance instead; both columns are stored either way.
CW_KEYS = CW_KEYS_STRAIN

#: Provenance arrays: not used by the flow, but they make the CW columns re-derivable.  ``cw_hc``
#  is an independent cross-check on ``log10_h0``.  ~half the library size.
PROV_KEYS = ('cw_mtot', 'cw_mrat', 'cw_redz', 'cw_hc')

#: Diagnostic index arrays, stored as int16.
IDX_KEYS = ('cw_fidx', 'cw_lidx')

#: The GWB columns the flow trains on: the free spectrum in the enterprise/pandora convention,
#  one value per frequency bin of the library's own grid.  See :func:`gwb_free_spectrum`.
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
        f"nfreqs={args.nfreqs}, nsub={args.nsub}, dur={args.dur_yr} [yr], "
        f"df={args.df*1e9:.4f} [nHz], "
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
    # NOTE: the anchor and the resolution are set SEPARATELY.  `utils.pta_freqs` derives its
    # spacing from the duration (df = 1/dur), so reaching a finer grid there means claiming a
    # longer observation -- which drags the lowest frequency down with it.  Here `dur` fixes the
    # lowest frequency at 1/dur and `nsub` subdivides, so a finer grid never invents bins below
    # what the PTA can resolve.  See `flow_freqs`.
    parser.add_argument('--dur', action='store', dest='dur_yr', type=float, default=DEF_PTA_DUR,
                        help='PTA observing duration [yr]; sets the lowest frequency, 1/dur')
    parser.add_argument('--nsub', action='store', dest='nsub', type=int, default=DEF_NSUB,
                        help='Frequency sub-bins per PTA Fourier bin (1 = the real PTA grid)')
    parser.add_argument('-f', '--nfreqs', action='store', dest='nfreqs', type=int,
                        help='Number of frequency bins', default=DEF_NUM_FBINS)

    # ---- what to keep
    parser.add_argument('--nrank', action='store', dest='nrank', type=int, default=DEF_NUM_RANK,
                        help='Number of top-ranked CWs kept per realization (frozen into library)')
    parser.add_argument('--rankby', action='store', dest='rankby', type=str, default=DEF_RANKBY,
                        choices=['hc', 'resid'],
                        help='Statistic used to rank CWs across the band (frozen into library)')
    parser.add_argument('--cw-keys', dest='cw_keys', choices=('strain', 'dcom'), default='strain',
                        help="which CW triple the library declares in `cw_keys`, i.e. what a flow "
                             "trains on by default.  Both columns are stored regardless. [strain]")
    parser.add_argument('--no-gwb', dest='save_gwb', default=True, action='store_false',
                        help="skip the `half_log10rho` free-spectrum columns (CW columns only)")

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

    if args.nsub < 1:
        raise ValueError(f"`nsub`={args.nsub} must be a positive integer!")

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

    # convert the human-facing [yr] to [sec] and derive the bin width once, here
    args.dur = args.dur_yr * YR
    args.df = 1.0 / (args.nsub * args.dur)

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
    for pk in ['df', 'dur', 'domain', 'plot', 'output_sims', 'output_logs']:
        config.pop(pk, None)

    pspace = config.pop('param_space')
    output = config.pop('output')

    args = _setup_argparse([pspace, output], **config)

    return args, fname


# ==============================================================================
# ====    Simulation    ====
# ==============================================================================


def flow_freqs(nfreqs, nsub=DEF_NSUB, dur=DEF_PTA_DUR*YR):
    """Observed GW frequency bin centers and edges, anchored to the PTA's lowest frequency.

    Arguments
    ---------
    nfreqs : int
        Number of frequency bins.
    nsub : int
        Number of sub-bins per PTA Fourier bin; the bin width is ``df = 1/(nsub*dur)``.
        ``nsub=1`` is the real PTA grid.
    dur : float
        PTA observing duration [sec].  Sets the lowest frequency, ``f_0 = 1/dur``.

    Returns
    -------
    cents : (F,) ndarray
        Bin centers, ``f_i = (1 + i/nsub) / dur`` for ``i`` in ``[0, nfreqs)`` [Hz].
    edges : (F+1,) ndarray
        Bin edges, ``(1 + (j - 0.5)/nsub) / dur`` for ``j`` in ``[0, nfreqs]`` [Hz].

    """
    f0 = 1.0 / dur
    df = f0 / nsub
    cents = f0 + np.arange(nfreqs) * df
    edges = f0 + (np.arange(nfreqs + 1) - 0.5) * df
    return cents, edges


def gwb_free_spectrum(hc, fobs_cents, df):
    """Convert a characteristic strain per frequency bin into the PTA free-spectrum parameter.

    Arguments
    ---------
    hc : (F, ...) ndarray
        Characteristic strain per bin.
    fobs_cents : (F,) ndarray
        Bin centers [Hz].
    df : float
        Bin width [Hz].  This is what ``rho`` is defined against, and it is the same ``df`` the
        likelihood must be handed -- pandora's GWB models all take it as a free argument rather
        than assuming ``1/Tspan``.

    Returns
    -------
    half_log10rho : (F, ...) ndarray
        ``log10(rho_i)``, or ``NaN`` for a bin holding no binaries at all.

    Notes
    -----
    ``rho_i^2 = h_c,i^2 / (12 pi^2 f_i^3) * df`` is the power in bin ``i`` -- the prior variance
    of the corresponding Fourier coefficient of the timing residuals.

    The grid is deliberately finer than ``1/Tspan``.  The data alone cannot separate modes spaced
    more finely than that -- they are not orthogonal over the observing span -- but an
    astrophysically informed prior over the whole vector, which is what this library is for, can.
    A coarser convention would also force the CW and GWB columns onto different grids.

    """
    ff = fobs_cents.reshape((-1,) + (1,) * (hc.ndim - 1))
    rho2 = hc**2 * df / (12.0 * np.pi**2 * ff**3)
    # An empty bin becomes NaN, never a sentinel value: the same choice `cw_columns` makes for
    # empty source slots, and for the same reason -- a finite stand-in orders of magnitude below
    # the real values would survive the training array's `isfinite` mask and then dominate any
    # per-column normalization.  Empty bins cluster in degenerate parameter samples rather than
    # scattering, so this drops bad samples, not good rows.
    return np.where(rho2 > 0.0, 0.5 * np.log10(np.where(rho2 > 0.0, rho2, 1.0)), np.nan)


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

        fobs_cents, fobs_edges = flow_freqs(args.nfreqs, nsub=args.nsub, dur=args.dur)
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
    save_gwb=True,
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
        Also return ``half_log10rho``, the free spectrum of the whole population.
    log : ``logging.Logger`` instance

    Returns
    -------
    data : dict
        Arrays shaped ``(1, nrank, nreals)``, plus ``fobs_cents``/``fobs_edges``, plus
        ``half_log10rho`` shaped ``(nfreqs, nreals)`` when ``save_gwb``.  See the module docstring
        for the layout, and :func:`cw_columns` for the column definitions.

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
        # The TOTAL power in each bin: background plus every single source, the kept CWs
        # included.  They are not subtracted -- the free spectrum a real analysis recovers holds
        # whatever is in the band, and this library supplies a prior, not a likelihood.  One
        # consequence is that `nloudest` does not enter here at all: it only controls how the
        # same total is split between `hc2rest` and `hc2ss`.
        hc_total = np.sqrt(hc2rest + np.sum(hc2ss, axis=-1))            # (F, R)
        data['half_log10rho'] = gwb_free_spectrum(hc_total, fobs_cents, fobs_edges[1] - fobs_edges[0])

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


#: Geometrized units, derived rather than imported so this does not depend on the holodeck branch.
_TSUN = NWTG * MSOL / SPLC ** 3       # G*Msol/c^3 [s]
_MPC2S = MPC / SPLC                   # Mpc in light-seconds [s/Mpc]


def strain_amplitude_h0(mc_msol, dlum_mpc, fobs):
    r"""CW strain amplitude ``h0``, in the convention PTA CW SAMPLERS use.

    .. math::
        h_0 = \frac{2 (G \mathcal{M}_c)^{5/3} (\pi f_\mathrm{gw})^{2/3}}{c^4 D_L}

    **This is not holodeck's strain convention.**  :func:`holodeck.utils.gw_strain_source` returns
    the sky- and polarization-averaged (Sesana / Enoki) amplitude
    ``h_s = (8/sqrt(10)) (G Mc)^(5/3) (pi f)^(2/3) / (c^4 D_L)``, which is larger by
    ``sqrt(8/5)`` = +0.1021 dex, because it folds the orientation average into the amplitude.  The
    form here leaves inclination and polarization to the waveform instead.  ``cw_hc`` in the same
    library is in holodeck's convention; converting between them is that ``sqrt(8/5)``.

    Arguments
    ---------
    mc_msol : array_like
        REDSHIFTED chirp mass, ``Mc*(1+z)`` [Msol] -- the pairing that goes with `fobs`.
    dlum_mpc : array_like
        LUMINOSITY distance [Mpc].  NOT the comoving distance ``log10_dc`` stores: multiply that
        by ``(1+z)`` first.
    fobs : array_like
        OBSERVED GW frequency [Hz].

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
        ``log10_h0`` : log10 strain amplitude, PTA-sampler convention (:func:`strain_amplitude_h0`)
                       -- NOT holodeck's sky-averaged convention, which ``cw_hc`` uses.
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
        # `log10_h0` postdates some libraries; do not demand it when re-combining an older run.
        stored_cw = [key for key in STORED_CW_KEYS if key in temp]

    if shape is None:
        err = f"Every one of the {nsamp_all} simulation files is a failure!"
        log.exception(err)
        raise RuntimeError(err)

    nrank, nreals = shape[1], shape[2]
    log.info(f"nfreqs={len(fobs_cents)}, {nrank=}, {nreals=}, save_gwb={args.save_gwb}")

    missing_cw = [key for key in STORED_CW_KEYS if key not in stored_cw]
    if missing_cw:
        log.warning(f"simulation files carry no {missing_cw}; combining without them")

    float_keys = list(stored_cw) + list(PROV_KEYS)
    idx_keys = list(IDX_KEYS)
    gwb_shape = (len(fobs_cents), 1, nreals)

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
            for key in GWB_KEYS:
                dsets[key] = h5.create_dataset(
                    key, shape=gwb_shape + (nsamp_all,), dtype='f8', chunks=gwb_shape + (1,))

        for ii in tqdm.trange(nsamp_all):
            temp = np.load(_get_sim_fname(path_sims, ii))
            if 'fail' in list(temp.keys()):
                bad_files[ii] = True
                for key in float_keys:
                    dsets[key][..., ii] = np.nan
                for key in idx_keys:
                    dsets[key][..., ii] = -1
                if args.save_gwb:
                    for key in GWB_KEYS:
                        dsets[key][..., ii] = np.nan
                continue

            for key in float_keys + idx_keys:
                dsets[key][..., ii] = temp[key]
            if args.save_gwb:
                for key in GWB_KEYS:
                    dsets[key][..., ii] = temp[key][:, np.newaxis, :]

        param_samples[bad_files] = np.nan
        h5.create_dataset('sample_params', data=param_samples)
        # (nastro, 1, 1, nsamp): broadcasts against the (1, nrank, nreals, nsamp) CW columns
        h5.create_dataset('theta_ast', data=param_samples.T[:, np.newaxis, np.newaxis, :])

        h5.attrs['param_names'] = np.array(param_names).astype('S')
        # Declaring a triple whose column is absent would produce an unloadable file.
        declared = CW_KEYS_STRAIN if getattr(args, 'cw_keys', 'strain') == 'strain' else CW_KEYS_DCOM
        if not all(key in stored_cw for key in declared):
            other = CW_KEYS_DCOM if declared is CW_KEYS_STRAIN else CW_KEYS_STRAIN
            log.warning(f"cannot declare {declared} -- not all columns are stored; "
                        f"declaring {other} instead")
            declared = other
        h5.attrs['cw_keys'] = np.array(declared).astype('S')
        if args.save_gwb:
            h5.attrs['gwb_keys'] = np.array(GWB_KEYS).astype('S')
        h5.attrs['parameter_space_class_name'] = pspace.name
        h5.attrs['nrank'] = nrank
        h5.attrs['nreals'] = nreals
        h5.attrs['nloudest'] = args.nloudest
        h5.attrs['rankby'] = args.rankby
        h5.attrs['df'] = args.df
        h5.attrs['nsub'] = args.nsub
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
