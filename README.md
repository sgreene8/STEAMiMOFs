# STEAMiMOFs: Simulaton Tools for Energies of Adsorption of Molecules in Metal-Organic Frameworks
This python-based package can be used to predict the adsorption properties of metal-organic frameworks or other nanoporous materials. It implements Monte Carlo simulations for calculating adsorption isoterms and enthalpies of adsorption using machine-learned interatomic potentials based on the [Allegro framework](https://www.nature.com/articles/s41467-023-36329-y).

## Installation

`STEAMiMOFs` requires the `allegro` package and its dependencies, which can be installed by following the instructions at https://github.com/mir-group/allegro/.

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
Following installation, the script named `STEAMiMOFs-nvtw` will be added to your `PATH`. This script can be used to perform a Monte Carlo calculation using the [flat-histogram NVT+W method](https://pubs.acs.org/doi/10.1021/acs.jpcc.0c11082). For example, the command
```bash
STEAMiMOFs-nvtw --NNP_path deployed.pth --MOF_structure_path UiO-66.cif --num_h2o 10
```
will launch a NVT+W calculation with 10 H2O molecules in the structure in the cif file `UiO-66.cif` using the Allegro model saved in `deployed.pth`. Simply running `STEAMiMOFs-nvtw` will print a detailed description of all arguments.

The script `STEAMiMOFs-enthalpy` can be used to calculate adsorption enthalpies using the output files `insert_*` and `remove_*`. See argument help for more information.

The script `STEAMiMOFs-isotherm` can be used to calculate adsorption isotherms using the output files `energy_*`. See argument help for more information.