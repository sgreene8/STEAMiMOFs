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
        "--H2O_energy",
        help="Internal energy of a free H2O molecule, usually calculated using the same DFT setup used to generate training data for the potential",
        type=float,
        default=-.16875515E+02, # vdW-DF
        required=True
    )

    if len(sys.argv) == 1:
        parser.print_help()
        parser.exit()
    
    args = parser.parse_args()

    nn_model = structure.MOFWithAds(args.NNP_path, args.structure_path)

if __name__ == "__main__":
    main()