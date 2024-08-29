import argparse
from pathlib import Path
import sys
from STEAMiMOFs import structure

def main():
    parser = argparse.ArgumentParser(
        description="Perform a NVT+W simulation to determine the loading of a MOF at a target temperature and adsorbate pressure.")

    parser.add_argument(
        "--NNP_path",
        help="Path to a deployed Allegro neural network potential",
        type=Path,
        default=None,
        required=True
    )

    parser.add_argument(
        "--structure_path",
        help="Path to the structure file for the MOF containing a certain number of H2O molecules",
        type=Path,
        default=None,
        required=True
    )

    parser.add_argument(
        "--results_dir",
        help="Path to the directory in which results, including NVT+W insertion and removal probabilities, and PDB trajectory should be saved",
        type=Path,
        default=None,
        required=True
    )

    parser.add_argument(
        "--temperature",
        help="Temperature of the simulation in K",
        default=298.,
        required=False
    )

    parser.add_argument(
        "--H2O_energy",
        help="Internal energy of a free H2O molecule, usually calculated using the same DFT setup used to generate training data for the potential",
        type=float,
        default=-.16875515E+02, # vdW-DF
        required=True
    )

    parser.add_argument(
        "--translate_probability",
        help="The probability of attempting a rigid translation move in each step. Probabilities need not be normalized.",
        type=float,
        default=1.,
        required=False
    )

    parser.add_argument(
        "--rotate_probability",
        help="The probability of attempting a rigid rotation move in each step. Probabilities need not be normalized.",
        type=float,
        default=1.,
        required=False
    )

    parser.add_argument(
        "--nvtw_insert_probability",
        help="The probability of attempting a NVT+W insertion move in each step. Probabilities need not be normalized.",
        type=float,
        default=1.,
        required=False
    )

    parser.add_argument(
        "--nvtw_remove_probability",
        help="The probability of attempting a NVT+W removal move in each step. Probabilities need not be normalized.",
        type=float,
        default=1.,
        required=False
    )

    parser.add_argument(
        "--num_h2o",
        help="The number of H2O molecules to add to the simulation cell upon launch",
        type=int,
        default=0,
        required=False
    )

    parser.add_argument(
        "--num_cycles",
        help="Target number of MC cycles to run",
        type=int,
        default=100,
        required=False
    )
    

    if len(sys.argv) == 1:
        parser.print_help()
        parser.exit()
    
    args = parser.parse_args()

    mof_ads = structure.MOFWithAds(args.NNP_path, args.structure_path, args.results_dir, args.temperature, args.H2O_energy)

    # Normalize probabilities
    tot_probability = args.translate_probability + args.rotate_probability + args.nvtw_insert_probability + args.nvtw_remove_probability
    trans_prob = args.translate_probability / tot_probability
    rotate_prob = args.rotate_probability / tot_probability
    nvtw_ins_prob = args.nvtw_insert_probability / tot_probability
    nvtw_rem_prob = args.nvtw_remove_probability / tot_probability

    # Create output file handles
    nvtw_ins_file = open(args.results_dir + 'insert_prob.txt', 'a')
    nvtw_rem_file = open(args.results_dir + 'remove_prob.txt', 'a')

    mof_ads.insert_h2o(args.num_h2o)

    mc_steps_per_cycle = max(20, mof_ads.nh2o)

if __name__ == "__main__":
    main()