"""holodeck

Massive black hole (MBH) binary simulator for pulsar timing array (and associated) signals.

"""

from setuptools import setup, find_packages

short_description = __doc__.strip()

with open("README.md", "r") as handle:
    long_description = handle.read()

with open("requirements.txt", "r") as handle:
    requirements = handle.read()

with open('holodeck/version.txt') as handle:
    version = handle.read().strip()

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

)
