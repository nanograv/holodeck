"""holodeck: Massive black hole (MBH) binary simulator for pulsar timing array signals.


To build cython library in-place:
    $ python setup.py build_ext -i

"""

from pathlib import Path
from os.path import abspath, join
from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy as np

SETUP_PATH = Path(__file__).absolute().resolve().parent
print(f"{SETUP_PATH=}")

# ---- Prepare Meta-Data ----

# NOTE: `short_description` gets first line of `__doc__` only (linebreaks not allowed by setuptools)
short_description = __doc__.strip().split('\n')[0]

fname_desc = SETUP_PATH.joinpath("README.md")
with open(fname_desc, "r") as handle:
    long_description = handle.read()

fname_reqs = SETUP_PATH.joinpath("requirements.txt")
with open(fname_reqs, "r") as handle:
    requirements = handle.read()

fname_vers = SETUP_PATH.joinpath('./holodeck/version.txt')
with open(fname_vers) as handle:
    version = handle.read().strip()


# ---- Handle cython submodules ----

fname_cyutils = SETUP_PATH.joinpath("holodeck", "cyutils.pyx")
ext_cyutils = Extension(
    "holodeck.cyutils",    # specify the resulting name/location of compiled extension
    sources=[str(fname_cyutils)],   # location of source code
    # define parameters external libraries
    include_dirs=[
        np.get_include()
    ],
    library_dirs=[
        abspath(join(np.get_include(), '..', '..', 'random', 'lib')),
        abspath(join(np.get_include(), '..', 'lib'))
    ],
    libraries=['npyrandom', 'npymath'],

    # Silence some undesired warnings
    define_macros=[('NPY_NO_DEPRECATED_API', 0)],
    extra_compile_args=['-Wno-unreachable-code-fallthrough', '-Wno-unused-function'],
)

fname_sam_cyutils = SETUP_PATH.joinpath("holodeck", "sams", "sam_cyutils.pyx")
ext_sam_cyutils = Extension(
    "holodeck.sams.sam_cyutils",    # specify the resulting name/location of compiled extension
    sources=[str(fname_sam_cyutils)],   # location of source code
    # define parameters external libraries
    include_dirs=[
        np.get_include()
    ],
    library_dirs=[
        abspath(join(np.get_include(), '..', '..', 'random', 'lib')),
        abspath(join(np.get_include(), '..', 'lib'))
    ],
    libraries=['npyrandom', 'npymath'],

    # Silence some undesired warnings
    define_macros=[('NPY_NO_DEPRECATED_API', 0)],
    extra_compile_args=['-Wno-unreachable-code-fallthrough', '-Wno-unused-function'],
)

cython_modules = cythonize(
    [ext_cyutils, ext_sam_cyutils],
    compiler_directives={"language_level": "3"},
    annotate=True,   # create html output about cython files
)


# ---- Perform Setup ----

setup(
    name='holodeck-gw',
    author='NANOGrav',
    author_email='luke.kelley@nanograv.org',
    description=short_description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=version,
    license='MIT',
    url="https://github.com/NANOGrav/holodeck/",

    # External dependencies loaded from 'requirements.txt'
    install_requires=requirements,

    # Which Python importable modules should be included when your package is installed
    # Handled automatically by setuptools. Use 'exclude' to prevent some specific
    # subpackage(s) from being added, if needed
    packages=find_packages(),

    # Optional include package data to ship with your package
    # Customize MANIFEST.in if the general case does not suit your needs
    # Comment out this line to prevent the files from being packaged with your software
    include_package_data=True,

    # Additional entries you may want simply uncomment the lines you want and fill in the data
    # url='http://www.my_package.com',  # Website
    python_requires=">=3.9",          # Python version restrictions

    ext_modules=cython_modules,
)
