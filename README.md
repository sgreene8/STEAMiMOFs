# STEAMiMOFs
This python-based package can be used to predict the adsorption properties of metal-organic frameworks or other nanoporous materials. It implements Monte Carlo simulations for calculating adsorption isoterms and enthalpies of adsorption using machine-learned interatomic potentials based on the [Allegro framework](https://www.nature.com/articles/s41467-023-36329-y).

## Installation

`STEAMiMOFs` requires the `allegro` package and its dependencies, which can be installed following the instructions at https://github.com/mir-group/allegro/.

Once `allegro` is installed, `STEAMiMOFs` can be installed by first downloading the source code:
```bash
git clone --depth 1 https://github.com/sgreene8/STEAMiMOFs/
```
and then installing using `pip`:
```bash
cd STEAMiMOFs
pip install .
cd ..
```

## Usage
The `nvtw.py` script can be used to calculate free energies as a function of temperature and external partial pressure, which can in turn be used to calculate adsorption isotherms.