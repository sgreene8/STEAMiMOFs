import argparse
import pathlib
import sys
import emcee
import numpy as np
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(
        description="Calculate the enthalpy of adsorption as a function of number of adsorbate molecules.")

    parser.add_argument(
        "--energy_files",
        help="List of text files containing energies of adsorbed molecules.",
        type=argparse.FileType('r'),
        nargs='+',
        default=None,
        required=True
    )

    parser.add_argument(
        "--loadings",
        help="List of the numbers of adsorbed molecules respresented in each of the files in the energy_files argument.",
        type=int,
        nargs='+',
        default=None,
        required=True
    )

    parser.add_argument(
        "--MOF_energy",
        help="Energy of the empty MOF",
        type=float,
        default=None,
        required=True
    )

    parser.add_argument(
        "--H2O_energy",
        help="Internal energy of a free H2O molecule, usually calculated using the same DFT setup used to generate training data for the potential",
        type=float,
        required=True
    )

    parser.add_argument(
        "--results_dir",
        help="Path to the directory in which results, including enthalpy data file and plot, should be saved",
        type=pathlib.Path,
        default='.',
        required=False
    )

    parser.add_argument(
        "--burn_in",
        help="The burn-in times that should be used for each loading. If only 1 value is provided, this value will be used for all loadings.",
        type=int,
        nargs='+',
        default=[0],
        required=False
    )

    if len(sys.argv) == 1:
        parser.print_help()
        parser.exit()
    
    args = parser.parse_args()

    n_loadings = len(args.loadings)
    assert(n_loadings == len(args.energy_files))

    burn_ins = args.burn_in
    if len(burn_ins) == 1:
        burn_ins = burn_ins * n_loadings

    outfile = open(args.results_dir / 'enthalpies.txt', 'w')
    outfile.write('Loading    Mean     Uncertainty\n')

    en_outfile = open(args.results_dir / 'energies.txt', 'w')
    en_outfile.write('Loading    Mean     Uncertainty     Converged?\n')

    en_fig, en_ax = plt.subplots(nrows=n_loadings, figsize=(5, 3 * n_loadings))
    energies = np.zeros(n_loadings + 1)
    energies[0] = args.MOF_energy
    errors = np.zeros(n_loadings + 1)
    converged = np.ones(n_loadings, dtype=bool)

    for loading_idx in range(n_loadings):
        loading = args.loadings[loading_idx]
        file = args.energy_files[loading_idx]
        print('Analyzing {} with {} molecules'.format(file.name, loading))
        data = np.genfromtxt(file)[burn_ins[loading_idx]:]
        num_en = data.shape[0]

        try:
            iat = emcee.autocorr.integrated_time(data, quiet=False)[0]
        except:
            converged[loading_idx] = False
            iat = emcee.autocorr.integrated_time(data, quiet=True)[0]
        
        en_ax[loading_idx].set_title('{} molecules'.format(loading))
        en_ax[loading_idx].set_ylabel('Ads. energy per molec.')
        en_ax[loading_idx].plot((data - args.MOF_energy) / loading - args.H2O_energy)

        energies[loading_idx + 1] = np.mean(data)
        errors[loading_idx + 1] = (np.var(data) * iat / data.shape[0])**0.5
    
    all_loadings = np.append(0, args.loadings)
    enthalpies = (energies[1:] - energies[:-1]) / (all_loadings[1:] - all_loadings[:-1]) - args.H2O_energy
    enthalpy_err = (errors[1:]**2 + errors[:-1]**2)**0.5 / (all_loadings[1:] - all_loadings[:-1])

    for loading_idx in range(n_loadings):
        loading = args.loadings[loading_idx]
        print('Enthalpy : {} ± {}'.format(enthalpies[loading_idx], enthalpy_err[loading_idx]))
        outfile.write('{} {} {}\n'.format(loading, enthalpies[loading_idx], enthalpy_err[loading_idx]))
        en_outfile.write('{} {} {} {}\n'.format(loading, (energies[loading_idx + 1] - args.MOF_energy) / loading - args.H2O_energy, errors[loading_idx + 1] / loading, 'Y' if converged[loading_idx] else 'N'))
    
    outfile.close()
    en_outfile.close()
    en_fig.savefig('energies.pdf', bbox_inches='tight')

if __name__ == "__main__":
    main()