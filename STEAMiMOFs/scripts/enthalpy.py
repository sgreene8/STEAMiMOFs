import argparse
import pathlib
import sys
import emcee
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

    if len(sys.argv) == 1:
        parser.print_help()
        parser.exit()
    
    args = parser.parse_args()

    n_loadings = len(args.loadings)
    assert(n_loadings == len(args.energy_files))

    outfile = open(args.results_dir / 'enthalpies.txt', 'w')
    outfile.write('Loading Mean Uncertainty')

    n_burn_in = 10 # number of different burn-in times to try
    iat_fig, iat_ax = plt.subplots(nrows=n_loadings, figsize=(5, 3 * n_burn_in))
    en_fig, en_ax = plt.subplots(nrows=n_loadings, figsize=(5, 3 * n_burn_in))

    for loading_idx, file in zip(args.loadings, args.energy_files):
        loading = args.loadings[loading_idx]
        file = args.energy_file[loading_idx]
        print('Analyzing {} with {} molecules'.format(file, loading))
        energies = np.genfromtxt(file)
        num_en = energies.shape[0]
        burn_in = np.arange(0, num_en // 2, num_en // (n_burn_in * 2), dtype=int)
        iats = np.zeros(n_burn_in)
        for burn_in_index in range(n_burn_in):
            iats[burn_in_index] = emcee.autocorrelation.integrated_time(energies[burn_in[burn_in_index]:])
        eff_samp_sizes = (num_en - burn_in) / iats
        optimal_idx = np.argmax(eff_samp_sizes)
        nonoptimal_mask = np.ones(burn_in.shape[0], dtype=bool)
        nonoptimal_mask[optimal_idx] = False
        iat_ax[loading_idx].scatter(burn_in[nonoptimal_mask], iats[nonoptimal_mask], color='k')
        iat_ax[loading_idx].scatter(burn_in[optimal_idx], iats[optimal_idx], color='r')
        print('Optimal burn-in: {}'.format(burn_in[optimal_idx]))
        mean_en = np.mean(energies[burn_in[optimal_idx]:])
        std_err = (np.var(energies[burn_in[optimal_idx]:]) / eff_samp_sizes[optimal_idx])**0.5
        print('Energy: {} ± {}'.format(mean_en, std_err))
        outfile.write('{} {} {}'.format(loading, mean_en, std_err))
    
    outfile.close()

if __name__ == "__main__":
    main()