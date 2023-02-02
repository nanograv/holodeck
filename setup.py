"""holodeck: Massive black hole (MBH) binary simulator for pulsar timing array signals.


To build cython library in-place:
    $ python setup.py build_ext -i

"""

from os.path import abspath, join
from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy as np

# ---- Prepare Meta-Data ----

# NOTE: `short_description` gets first line of `__doc__` only (linebreaks not allowed by setuptools)
short_description = __doc__.strip().split('\n')[0]

with open("README.md", "r") as handle:
    long_description = handle.read()

with open("requirements.txt", "r") as handle:
    requirements = handle.read()

with open('holodeck/version.txt') as handle:
    version = handle.read().strip()


# ---- Handle cython submodules ----

ext_cyutils = Extension(
    "holodeck.cyutils",    # specify the resulting name/location of compiled extension
    sources=[join('.', 'holodeck', 'cyutils.pyx')],   # location of source code
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
    [ext_cyutils],
    compiler_directives={"language_level": "3"},
    annotate=True,   # create html output about cython files
)


# ---- Perform Setup ----

setup(
    name='holodeck',
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
    python_requires=">=3.8",          # Python version restrictions

    ext_modules=cython_modules,
)
