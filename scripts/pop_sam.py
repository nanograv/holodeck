"""Generation a SAM population and save.
"""

import argparse
import subprocess
import warnings
from pathlib import Path

import numpy as np

import kalepy as kale

import holodeck as holo
import holodeck.sam
import holodeck.relations
import holodeck.evolution
import holodeck.utils
from holodeck.constants import YR, MPC
from holodeck import cosmo

log = holo.log

CWD = Path('.').resolve()

# DEBUG = False

# ---- Fail on warnings
# err = 'ignore'
err = 'raise'
np.seterr(divide=err, invalid=err, over=err)
warn_err = 'error'
# warnings.filterwarnings(warn_err, category=UserWarning)
warnings.filterwarnings(warn_err)

# ---- Setup ArgParse

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--reals', action='store', dest='reals', type=int,
                    help='number of realizations', default=10)
parser.add_argument('-s', '--shape', action='store', dest='shape', type=int,
                    help='shape of SAM grid', default=50)
parser.add_argument('-t', '--threshold', action='store', dest='threshold', type=float,
                    help='sample threshold', default=100.0)
parser.add_argument('-d', '--dur', action='store', dest='dur', type=float,
                    help='PTA observing duration [yrs]', default=20.0)
parser.add_argument('-c', '--cad', action='store', dest='cad', type=float,
                    help='PTA observing cadence [yrs]', default=0.1)
parser.add_argument('--debug', action='store_true', default=False, dest='debug',
                    help='run in DEBUG mode')
parser.add_argument('-v', '--verbose', action='store_true', default=False, dest='verbose',
                    help='verbose output [INFO]')
# parser.add_argument('-d', '--debug', action='store_true', default=False, dest='debug',
#                     help='run in DEBUG mode')
# parser.add_argument('--version', action='version', version='%(prog)s 1.0')

args = parser.parse_args()

lvl = log.INFO if args.verbose else log.WARNING
log.setLevel(lvl)


def main():
    """

    Parameters
    ----------
    dur : _type_, optional
    cad : _type_, optional
    nreals : int, optional
    sample_threshold : _type_, optional

    Returns
    -------
    _type_

    """
    fobs = holo.utils.nyquist_freqs(args.dur * YR, args.cad * YR)
    if args.debug:
        log.warning("!RUNNING IN 'DEBUG' MODE!")
        args.shape = 20 if (20 < args.shape) else args.shape

    # ---- Construct Semi-Analytic Models

    gsmf = holo.sam.GSMF_Schechter()               # Galaxy Stellar-Mass Function (GSMF)
    gpf = holo.sam.GPF_Power_Law()                 # Galaxy Pair Fraction         (GPF)
    gmt = holo.sam.GMT_Power_Law()                 # Galaxy Merger Time           (GMT)
    mmbulge = holo.relations.MMBulge_Standard()    # M-MBulge Relation            (MMB)
    hard = holo.evolution.Hard_GW()

    sam = holo.sam.Semi_Analytic_Model(gsmf=gsmf, gpf=gpf, gmt=gmt, mmbulge=mmbulge, shape=args.shape)

    # ---- Calculate Number of Binaries

    edges, dnum = sam.diff_num_from_hardening(hard, fobs=fobs)

    edges_integrate = [np.copy(ee) for ee in edges]
    edges_sample = [np.log10(edges[0]), edges[1], edges[2], np.log(edges[3])]

    # Sample redshift by first converting to comoving volume, sampling, then convert back
    redz = edges[2]
    volume = cosmo.comoving_volume(redz).to('Mpc3').value   # NOTE: units of [Mpc^3]

    # convert from dN/dz to dN/dVc, dN/dVc = (dN/dz) * (dz/dVc) = (dN/dz) / (dVc/dz)
    dvcdz = cosmo.dVcdz(redz, cgs=False).value    # NOTE: units of [Mpc^3]
    dnum = dnum / dvcdz[np.newaxis, np.newaxis, :, np.newaxis]

    # change variable from redshift to comoving-volume, both sampling and integration
    edges_sample[2] = volume
    edges_integrate[2] = volume

    if args.debug:
        log.warning("DECREASING DENSITY FOR DEBUG MODE!")
        dnum = dnum / 10.0

    # Find the 'mass' (total number of binaries in each bin) by multiplying each bin by its volume
    # NOTE: this needs to be done manually, instead of within kalepy, because of log-spacings
    mass = holo.sam._integrate_differential_number(edges_integrate, dnum, freq=True)

    # ---- Prepare filenames

    # command = ["git", "describe"]
    command = ['git', 'rev-parse', '--short', 'HEAD']
    hash = subprocess.check_output(command).decode('ascii').strip()

    path_output = CWD

    base_name = (
        f"pop-sam_v{holo.__version__}_{hash}"
        "__"
        f"d{(args.dur):.1f}_c{(args.cad):.3f}_sh{args.shape:04d}_st{np.log10(args.threshold):+.3f}"
    )

    def get_path(real, basic=False):
        path = path_output.joinpath(base_name)
        temp = f"{real:04}.npz"
        if args.debug:
            temp = "debug_" + temp

        path = path.joinpath(temp)
        parent = path.parent
        parent.mkdir(parents=True, exist_ok=True)

        if basic:
            return path

        temp = path
        while temp.exists():
            real = real + 1
            temp = get_path(real, basic=True)

        path = temp
        return real, path

    # ---- Save basic SAM information

    path_info = get_path(0, basic=True)
    temp = 'sam.npz'
    if args.debug:
        temp = "debug_" + temp
    path_info = path_info.parent.joinpath(temp)
    np.savez(path_info, fobs=fobs, dnum=dnum, mass=mass, edges=np.array(edges_integrate, dtype=object))
    log.info(f"Saved SAM data to '{path_info}' size '{holo.utils.get_file_size(path_info)}'")

    # ---- Construct realizations

    real = 0
    for rr in holo.utils.tqdm(range(args.reals)):
        real, path = get_path(real)

        vals, weights = kale.sample_outliers(
            edges_sample, dnum, args.threshold, mass=mass, poisson_inside=True, poisson_outside=True
        )

        vals[0] = 10.0 ** vals[0]
        vals[2] = np.power(vals[2] / (4.0*np.pi/3.0), 1.0/3.0)
        vals[2] = cosmo.dcom_to_z(vals[2] * MPC)
        vals[3] = np.e ** vals[3]

        np.savez(path, mtot=vals[0], mrat=vals[1], redz=vals[2], fobs=vals[3], weights=weights)
        msg = (
            f"Saved {rr}/{args.reals} to '{path}' size '{holo.utils.get_file_size(path)}'"
            f" ({weights.size:.8e}, {weights.sum():.8e})"
        )
        log.info(msg)

    return


if __name__ == "__main__":
    header = f"holodeck :: {__file__}"
    print(header)
    print("=" * len(header) + "\n")
    print(f"args={args}")

    main()