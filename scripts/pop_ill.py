"""Generation a SAM population and save.
"""

import argparse
import subprocess
import warnings
from pathlib import Path

import numpy as np

import holodeck as holo
import holodeck.sam
import holodeck.relations
import holodeck.evolution
import holodeck.utils
from holodeck.constants import YR, MSOL, GYR

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
parser.add_argument('-s', '--sample', action='store', dest='resamp', type=float,
                    help='resample initial illustris population', default=5.0)
parser.add_argument('-d', '--dur', action='store', dest='dur', type=float,
                    help='PTA observing duration [yrs]', default=20.0)
parser.add_argument('-c', '--cad', action='store', dest='cad', type=float,
                    help='PTA observing cadence [yrs]', default=0.1)
parser.add_argument('--debug', action='store_true', default=False, dest='debug',
                    help='run in DEBUG mode')
parser.add_argument('-v', '--verbose', action='store_true', default=False, dest='verbose',
                    help='verbose output [INFO]')

parser.add_argument('-m', '--mamp', action='store', dest='mamp', type=float,
                    help='M-MBulge ampliture [Msol] (MM2013: 2.884e8)', default=7e8)

args = parser.parse_args()

lvl = log.INFO if args.verbose else log.WARNING
log.setLevel(lvl)


def main():
    fobs = holo.utils.nyquist_freqs(args.dur * YR, args.cad * YR)
    if args.debug:
        log.warning("!RUNNING IN 'DEBUG' MODE!")
        DOWN = 100.0
        args.resamp = 1.0
    else:
        DOWN = None

    # ---- Construct Illustris Models

    pop = holo.population.Pop_Illustris()
    mod_resamp = holo.population.PM_Resample(resample=args.resamp)
    pop.modify(mod_resamp)

    # default mamp is  10.0 ** 8.46 = 2.884e8
    mmbulge = holo.relations.MMBulge_MM2013(mamp=args.mamp*MSOL)
    mod_MM2013 = holo.population.PM_Mass_Reset(mmbulge, scatter=True)
    pop.modify(mod_MM2013)

    fixed = holo.hardening.Fixed_Time.from_pop(pop, 2.0 * GYR)
    evo = holo.evolution.Evolution(pop, fixed)
    evo.evolve()

    # ---- Prepare filenames

    command = ['git', 'rev-parse', '--short', 'HEAD']
    hash = subprocess.check_output(command).decode('ascii').strip()

    path_output = CWD

    base_name = (
        f"pop-ill_v{holo.__version__}_{hash}"
        "__"
        f"d{(args.dur):.1f}_c{(args.cad):.3f}_mamp{np.log10(args.mamp):+.6f}_rs{args.resamp:.2f}"
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

    # ---- Construct realizations

    real = 0
    for rr in holo.utils.tqdm(range(args.reals)):
        real, path = get_path(real)

        vals = evo.sample_full_population(fobs, DOWN=DOWN)

        np.savez(path, freq_bins=fobs, mtot=vals[0], mrat=vals[1], redz=vals[2], fobs=vals[3])
        msg = (
            f"Saved {rr}/{args.reals} to '{path}' size '{holo.utils.get_file_size(path)}'"
            f" ({vals.shape[1]:.8e})"
        )
        log.info(msg)

    return


if __name__ == "__main__":
    header = f"holodeck :: {__file__}"
    print(header)
    print("=" * len(header) + "\n")
    print(f"args={args}")

    main()

    pass
