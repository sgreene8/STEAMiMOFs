import argparse
from pathlib import Path
import sys
import json
from STEAMiMOFs import structure

def main():
    parser = argparse.ArgumentParser(
        description="Perform a NVT+W simulation to determine the thermodynamic properties of a MOF at a target temperature and loading.")

    parser.add_argument(
        "--NNP_paths",
        help="JSON string representing a dictionary whose keys represent paths to deployed Allegro neural network potentials and whose " +
             "values are lists containing the min and max number of adsorbate molecules to which the potential can be applied",
        type=json.loads,
        default=None,
        required=True
    )

    parser.add_argument(
        "--MOF_structure_path",
        help="Path to the structure file for the MOF without any H2O molecules",
        type=Path,
        default=None,
        required=True
    )

    parser.add_argument(
        "--H2O_structure_path",
        help="Path to a structure file for adsorbed H2O molecules. Useful for restarting a calculation. If the file contains multiple frames, only the last frame will be read.",
        type=Path,
        default=None,
        required=False
    )

    parser.add_argument(
        "--results_dir",
        help="Path to the directory in which results, including NVT+W insertion and removal probabilities, and PDB trajectory should be saved",
        type=Path,
        default='.',
        required=False
    )

    parser.add_argument(
        "--temperature",
        help="Temperature of the simulation in K",
        default=298.,
        type=float,
        required=False
    )

    parser.add_argument(
        "--H2O_DFT_data_path",
        help="YAML file containing the DFT energy of the H2O molecule, the O-H bond length, and the H-O-H angle. The file should contain the following keys: 'energy', 'rOH', and 'aHOH'.",
        type=Path,
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
        help="The number of H2O molecules to add to the simulation cell upon launch, in addition to those in the file specified by H2O_structure_path",
        type=int,
        default=0,
        required=False
    )

    parser.add_argument(
        "--num_cycles",
        help="Target number of MC cycles to run",
        type=int,
        default=100000,
        required=False
    )

    parser.add_argument(
        "--rng_seed",
        help="Seed for the random number generator, can be either a path to a file containing the state of the random number generator, or an integer. Useful for debugging purposes.",
        type=str,
        default=None,
        required=False
    )

    parser.add_argument(
        "--sampling_scheme",
        choices=["Metropolis", "MALA"],
        default="MALA",
        help="Sampling scheme, either standard Metropolis with uniform proposal densities, or Metropolis-adjusted Langevin algorithm with gradients"
    )

    parser.add_argument(
        "--translation_stepsize",
        type=float,
        default=0.1,
        required=False,
        help="Width (in Angstroms) of the normal distribution from which translation moves will be sampled"
    )

    parser.add_argument(
        "--rotation_stepsize",
        type=float,
        default=45,
        required=False,
        help="Width (in degrees) of the normal distribution from which rotation moves will be sampled"
    )

    parser.add_argument(
        "--vibrational_stepsize",
        type=float,
        default=0.05,
        required=False,
        help="Width (in Angstroms) of the normal distribution from which vibrational moves will be sampled"
    )

    parser.add_argument(
        "--ngrid_O",
        type=int,
        default=10
    )

    parser.add_argument(
        "--ngrid_H1",
        type=int,
        default=10
    )

    parser.add_argument(
        "--ngrid_H2",
        type=int,
        default=10
    )
    

    if len(sys.argv) == 1:
        parser.print_help()
        parser.exit()
    
    args = parser.parse_args()

    mof_ads = structure.MOFWithAds(args.NNP_paths, args.MOF_structure_path, args.H2O_DFT_data_path, args.H2O_structure_path, args.results_dir, 
                                   args.temperature, args.translation_stepsize, args.rotation_stepsize, args.vibrational_stepsize,
                                   args.ngrid_O, args.ngrid_H1, args.ngrid_H2)

    if args.rng_seed is not None:
        try:
            seed = int(args.rng_seed)
        except:
            seed = args.rng_seed
        mof_ads.seed_rng_state(seed)

    # Normalize probabilities
    tot_probability = args.translate_probability + args.rotate_probability + args.nvtw_insert_probability + args.nvtw_remove_probability
    trans_prob = args.translate_probability / tot_probability
    rotate_prob = args.rotate_probability / tot_probability
    nvtw_ins_prob = args.nvtw_insert_probability / tot_probability
    nvtw_rem_prob = args.nvtw_remove_probability / tot_probability

    cumu_rotate = trans_prob + rotate_prob
    cumu_insert = cumu_rotate + nvtw_ins_prob

    num_trans_accepted = 0
    num_trans_attempted = 0
    num_rot_accepted = 0
    num_rot_attempted = 0

    if args.num_h2o > 0:
        mof_ads.insert_h2o_lowen(args.num_h2o)
        mof_ads.check_h2o_geom()

    # Create output file handles
    if nvtw_ins_prob > 0:
        nvtw_ins_file = open(args.results_dir / 'insert_{:d}'.format(mof_ads.nh2o), 'a')
    if nvtw_rem_prob > 0:
        nvtw_rem_file = open(args.results_dir / 'remove_{:d}'.format(mof_ads.nh2o), 'a')
    
    energy_file = open(args.results_dir / 'energy_{:d}'.format(mof_ads.nh2o), 'a')

    mc_steps_per_cycle = max(20, mof_ads.nh2o)

    for cycle in range(args.num_cycles):
        print("Cycle {} of {}: energy = {} eV".format(cycle, args.num_cycles, mof_ads.current_potential_en))
        energy_file.write(str(mof_ads.current_potential_en) + '\n')
        energy_file.flush()
        if nvtw_ins_prob > 0:
            nvtw_ins_file.flush()
        if nvtw_rem_prob > 0:
            nvtw_rem_file.flush()
        mof_ads.write_to_traj()
        mof_ads.save_rng_state('rng_state.dat')

        if num_trans_attempted != 0:
            print("Translation moves accepted: {} of {} ({:.2f}%)".format(num_trans_accepted, num_trans_attempted, float(num_trans_accepted) / num_trans_attempted * 100))
        if num_rot_attempted != 0:
            print("Rotation moves accepted: {} of {} ({:.2f}%)".format(num_rot_accepted, num_rot_attempted, float(num_rot_accepted) / num_rot_attempted * 100))
        
        for step in range(mc_steps_per_cycle):
            rn = mof_ads.rng.random(1)
            if rn < trans_prob:
                num_trans_attempted += 1
                success = mof_ads.translate_h2o(sampling=args.sampling_scheme)
                if success:
                    num_trans_accepted += 1
            elif rn < cumu_rotate:
                num_rot_attempted += 1
                success = mof_ads.rotate_h2o(sampling=args.sampling_scheme)
                if success:
                    num_rot_accepted += 1
            elif rn < cumu_insert:
                nvtw_prob = mof_ads.insert_h2o(keep=False)
                nvtw_ins_file.write(str(nvtw_prob) + '\n')
            else:
                nvtw_prob = mof_ads.remove_h2o(put_back=True)
                nvtw_rem_file.write(str(nvtw_prob) + '\n')
    

    print("Cycle {} of {}: energy = {} eV".format(args.num_cycles, args.num_cycles, mof_ads.current_potential_en))
    if (num_trans_attempted > 0):
        trans_accept = float(num_trans_accepted) / num_trans_attempted
    else:
        trans_accept = float('NaN')
    print("Translation moves accepted: {} of {} ({:.2f}%)".format(num_trans_accepted, num_trans_attempted, trans_accept))
    if (num_rot_attempted > 0):
        rot_accept = float(num_rot_accepted) / num_rot_attempted
    else:
        rot_accept = float('NaN')
    print("Rotation moves accepted: {} of {} ({:.2f}%)".format(num_rot_accepted, num_rot_attempted, rot_accept))
    energy_file.write(str(mof_ads.current_potential_en) + '\n')
    energy_file.flush()
    if nvtw_ins_prob > 0:
       nvtw_ins_file.flush()
    if nvtw_rem_prob > 0:
        nvtw_rem_file.flush()
    mof_ads.write_to_traj()

    print("Simulation finished")

if __name__ == "__main__":
    main()