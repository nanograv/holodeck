"""Combine the GWB-only data in the target path folders.
"""

from pathlib import Path

import holodeck as holo

path = "/global/scratch/users/lzkelley"
#lib_names = [
#    "uniform-07a-rot_new_n1000_r1000_f40",
#]

path = Path(path)
if not path.is_dir():
    raise FileNotFoundError(f"Base path {path} is not a directory!")

pattern = "uniform-07*"
lib_names = sorted(list(path.glob(pattern)))
print(f"Found {len(lib_names)} libraries matching {pattern=}")

for lib in lib_names:
    lib_path = path.joinpath(lib)
    if not lib_path.is_dir():
        err = f"library path {lib_path} is not a directory!"
        print(err)
        # raise FileNotFoundError(err)
        continue
    test = lib_path.joinpath("sims")
    if not test.is_dir():
        err = f"library path does not contain a `sims` director {test}!"
        print(err)
        # raise FileNotFoundError(err)
        continue

    try:
        lib_path = holo.librarian.sam_lib_combine(lib_path, holo.log, gwb_only=True)
    except Exception as err:
        print(err)
        continue

