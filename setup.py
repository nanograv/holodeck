"""
holodeck

Supermassive binary black hole simulator for pulsar timing array signals and galaxy population  statistics.

"""
import sys
from setuptools import setup, find_packages

import versioneer

short_description = __doc__.split("\n")

# from https://github.com/pytest-dev/pytest-runner#conditional-requirement
needs_pytest = {'pytest', 'test', 'ptr'}.intersection(sys.argv)
pytest_runner = ['pytest-runner'] if needs_pytest else []

with open("README.md", "r") as handle:
    long_description = handle.read()

with open("requirements.txt", "r") as handle:
    requirements = handle.read()


setup(
    # Self-descriptive entries which should always be present
    name='holodeck',
    author='NANOGrav',
    author_email='luke.kelley@nanograv.org',
    description=short_description[0],
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
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

    # Allows `setup.py test` to work correctly with pytest
    setup_requires=[] + pytest_runner,

    # Additional entries you may want simply uncomment the lines you want and fill in the data
    # url='http://www.my_package.com',  # Website
    python_requires=">=3.4",          # Python version restrictions

    # Manual control if final package is compressible or not, set False to prevent the .egg from being made
    # zip_safe=False,

)
