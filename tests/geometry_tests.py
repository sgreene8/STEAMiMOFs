import STEAMiMOFs.structure
import numpy as np
from pathlib import Path

def test_rotation_inversion():
    h2oMOF = STEAMiMOFs.structure.MOFWithAds(None, Path('RUBTAK01_SL.cif'))
    h2oMOF.insert_h2o(10)

    index = 5
    h2o = h2oMOF._atoms[(h2oMOF.n_MOF_atoms + 3 * index):(h2oMOF.n_MOF_atoms + 3 * index + 3)]
    orig_pos = h2o.get_positions()
    for trial in range(100):
        rot_vector = h2oMOF.rng.standard_normal(3)
        rot_vector /= np.linalg.norm(rot_vector)
        rot_angle = h2oMOF.rng.standard_normal() * np.pi
        if rot_angle > np.pi:
            rot_angle -= 2 * np.pi
        if rot_angle < -np.pi:
            rot_angle += 2 * np.pi
        new_pos = h2oMOF._calculate_rot_positions(index, rot_angle, rot_vector)

        angle = STEAMiMOFs.structure._calc_rot_angle(orig_pos, new_pos)
        assert(abs(angle - abs(rot_angle)) < 1e-5)

test_rotation_inversion()