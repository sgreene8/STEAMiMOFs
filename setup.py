# Adopted from https://github.com/navdeep-G/samplemod

from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='STEAMiMOFs',
    version='0.1.0',
    description='Package for simulating the adsorption of small molecules in MOFs using Allegro potentials',
    long_description=readme,
    author='Samuel M. Greene',
    author_email='samuel.greene@austin.utexas.edu',
    url='https://github.com/sgreene8/STEAMiMOFs',
    license=license,
    packages=find_packages(include=["STEAMiMOFs", "STEAMiMOFs.*", "STEAMiMOFs.packages.*"]),
    entry_points={
        # make the scripts available as command line scripts
        "console_scripts": [
            "STEAMiMOFs-nvtw = STEAMiMOFs.nvtw:main",
            "STEAMiMOFs-enthalpy = STEAMiMOFs.enthalpy:main"
        ]
    },
    install_requires=[
        "argparse",
        "mir-allegro",
        "emcee"
    ]
)