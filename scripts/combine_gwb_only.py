"""Combine the GWB-only data in the target path folders.
"""

from pathlib import Path

import holodeck as holo

path = "/Users/lzkelley/Programs/nanograv/15yr_astro_libraries"
lib_names = [
    "uniform-07a_new_n1000_r1000_f40",
    "uniform-07a_new_n2000_r2000_f40",
    "uniform-07a-rot_new_n1000_r1000_f40",
]

path = Path(path)
if not path.is_dir():
    raise FileNotFoundError(f"Base path {path} is not a directory!")

for lib in lib_names:
    lib_path = path.joinpath(lib)
    if not lib_path.is_dir():
        raise FileNotFoundError(f"library path {lib_path} is not a directory!")
    test = lib_path.joinpath("sims")
    if not test.is_dir():
        raise FileNotFoundError(f"library path does not contain a `sims` director {test}!")

    lib_path = holo.librarian.sam_lib_combine(lib_path, holo.log, gwb_only=True)

