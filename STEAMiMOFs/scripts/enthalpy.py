import argparse
import pathlib
import sys
import emcee
import numpy as np
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(
        description="Calculate the enthalpy of adsorption as a function of number of adsorbate molecules. Uses the procedure from [JCTC 12, 1799-1805 (2016)] to automatically select equilibration time for averaging at each loading.")

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
        "--results_dir",
        help="Path to the directory in which results, including isotherm data file and plot, should be saved",
        type=pathlib.Path,
        default='.',
        required=False
    )

    parser.add_argument(
        "--burn_in_slope",
        help="Sets a lower bound on the range of burn-in values to try. The burn-in will be chosen to exclude any region at the beginning of the trajectory where the magnitude of the slope is greater than this value.",
        type=float,
        default=0.03,
        required=False
    )

    if len(sys.argv) == 1:
        parser.print_help()
        parser.exit()
    
    args = parser.parse_args()

    n_loadings = len(args.loadings)
    assert(n_loadings == len(args.energy_files))

    outfile = open(args.results_dir / 'enthalpies.txt', 'w')
    outfile.write('Loading    Mean     Uncertainty\n')

    n_burn_in = 10 # number of different burn-in times to try
    iat_fig, iat_ax = plt.subplots(nrows=n_loadings, figsize=(5, 3 * n_loadings), squeeze=False)
    en_fig, en_ax = plt.subplots(nrows=n_loadings, figsize=(5, 3 * n_loadings), squeeze=False)
    en_slope_fig, en_slope_ax = plt.subplots(nrows=n_loadings, figsize=(5, 3 * n_loadings), squeeze=False)
    slope_separation = 10

    for loading_idx in range(n_loadings):
        loading = args.loadings[loading_idx]
        file = args.energy_files[loading_idx]
        print('Analyzing {} with {} molecules'.format(file.name, loading))
        energies = np.genfromtxt(file)
        num_en = energies.shape[0]
        slope = (energies[slope_separation:] - energies[:-slope_separation]) / slope_separation

        burn_in_start = np.max(np.nonzero(np.abs(slope) > args.burn_in_slope * loading)[0]) + slope_separation
        burn_in = np.arange(burn_in_start, num_en // 2, (num_en // 2 - burn_in_start) // n_burn_in, dtype=int)[:n_burn_in]
        iats = np.zeros(n_burn_in)
        for burn_in_index in range(n_burn_in):
            try:
                iats[burn_in_index] = emcee.autocorr.integrated_time(energies[burn_in[burn_in_index]:])
            except:
                print('The amount of data for burn_in = {} is not sufficient to get an accurate estimate of the uncertainty. Treat this estimate with caution.'.format(burn_in[burn_in_index]))
                iats[burn_in_index] = emcee.autocorr.integrated_time(energies[burn_in[burn_in_index]:], c=0.01)
        eff_samp_sizes = (num_en - burn_in) / iats
        optimal_idx = np.argmax(eff_samp_sizes)
        nonoptimal_mask = np.ones(burn_in.shape[0], dtype=bool)
        nonoptimal_mask[optimal_idx] = False
        optimal_burnin = burn_in[optimal_idx]

        iat_ax[loading_idx, 0].set_title('{} molecules'.format(loading))
        en_ax[loading_idx, 0].set_title('{} molecules'.format(loading))
        en_slope_ax[loading_idx, 0].set_title('{} molecules'.format(loading))

        en_ax[loading_idx, 0].set_ylabel('Energy')
        iat_ax[loading_idx, 0].set_ylabel('IAT')
        en_slope_ax[loading_idx, 0].set_ylabel('Energy/iteration')

        iat_ax[loading_idx, 0].scatter(burn_in[nonoptimal_mask], iats[nonoptimal_mask], color='k')
        iat_ax[loading_idx, 0].scatter(optimal_burnin, iats[optimal_idx], color='r')
        print('Optimal burn-in: {}'.format(optimal_burnin))

        en_ax[loading_idx, 0].plot(energies)
        en_ax[loading_idx, 0].plot(2 * [optimal_burnin], [np.min(energies), np.max(energies)], 'k--')

        en_slope_ax[loading_idx, 0].fill_between(range(slope_separation + 50, num_en), np.min(slope[50:]), -args.burn_in_slope * loading, color=(0.5, 0.5, 0.5, 0.5))
        en_slope_ax[loading_idx, 0].plot(range(slope_separation + 50, num_en), slope[50:])

        mean_en = np.mean(energies[optimal_burnin:])
        std_err = (np.var(energies[optimal_burnin:]) / eff_samp_sizes[optimal_idx])**0.5
        print('Energy: {} ± {}'.format(mean_en, std_err))
        outfile.write('{} {} {}\n'.format(loading, mean_en, std_err))
    
    outfile.close()
    iat_fig.savefig('IAT.pdf', bbox_inches='tight')
    en_fig.savefig('energies.pdf', bbox_inches='tight')
    en_slope_fig.savefig('slope.pdf', bbox_inches='tight')

if __name__ == "__main__":
    main()