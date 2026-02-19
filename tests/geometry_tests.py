import STEAMiMOFs.structure
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib import cm, colors, colormaps

def test_rotation_inversion():
    h2oMOF = STEAMiMOFs.structure.MOFWithAds(None, Path('RUBTAK01_SL.cif'))
    h2oMOF.insert_h2o(10)

    index = 5
    h2o = h2oMOF._atoms[(h2oMOF.n_MOF_atoms + 3 * index):(h2oMOF.n_MOF_atoms + 3 * index + 3)]
    orig_pos = h2o.get_positions()
    com = h2o.get_center_of_mass()
    orig_rel = orig_pos - com
    for trial in range(100):
        rot_vector = h2oMOF.rng.standard_normal(3)
        rot_vector /= np.linalg.norm(rot_vector)
        rot_angle = h2oMOF.rng.standard_normal() * np.pi
        if rot_angle > np.pi:
            rot_angle -= 2 * np.pi
        if rot_angle < -np.pi:
            rot_angle += 2 * np.pi
        rot_matrix = STEAMiMOFs.structure._calc_rot_matrix(rot_angle, rot_vector)
        new_pos = orig_rel @ rot_matrix.T + com

        angle = STEAMiMOFs.structure._calc_rot_angle(orig_pos, new_pos)
        assert(abs(angle - abs(rot_angle)) < 1e-5)

def test_pbc_equivalence():
    h2oMOF = STEAMiMOFs.structure.MOFWithAds(Path('model.pth'), Path('RUBTAK01_SL.cif'))
    h2oMOF.insert_h2o(5)

    index = 0
    h2o = h2oMOF._atoms[(h2oMOF.n_MOF_atoms + 3 * index):(h2oMOF.n_MOF_atoms + 3 * index + 3)]
    orig_pos = h2o.get_positions()
    orig_scaled = h2o.get_scaled_positions()

    en1 = h2oMOF.current_potential_en
    print(f'original energy: {en1}')

    for x_shift in [-1, 0, 1]:
        for y_shift in [-1, 0, 1]:
            for z_shift in [-1, 0, 1]:
                new_scaled = np.array(orig_scaled)
                new_scaled[:, 0] += x_shift
                new_scaled[:, 1] += y_shift
                new_scaled[:, 2] += z_shift
                new_pos = h2oMOF._atoms.cell.cartesian_positions(new_scaled)
                for atom_idx in range(3):
                    h2oMOF._atoms[(h2oMOF.n_MOF_atoms + 3 * index + atom_idx)].position = new_pos[atom_idx]
                en2, _ = h2oMOF._evaluate_potential()
                print(f'new energy: {en2}')

def visualize_input_grid():
    h2oMOF = STEAMiMOFs.structure.MOFWithAds(Path('model.pth'), Path('RUBTAK01_SL.cif'))
    h2oMOF.insert_h2o(3)
    h2oMOF.write_to_traj(with_MOF=True)

    insert_probs = h2oMOF._insert_probs_grid()
    n_grid = insert_probs.shape[2]
    cell = h2oMOF._atoms.get_cell()
    grid = np.mgrid[0:n_grid] * 1.0 / n_grid

    # first two cell vectors lie in xy-plane
    assert(abs(cell[0, 2]) < 1e-8) 
    assert(abs(cell[1, 2]) < 1e-8)

    dx = cell[0, :2] / n_grid
    dy = cell[1, :2] / n_grid
    parallelogram = np.array([[0, 0], dx, dx + dy, dy])
    fig, ax = plt.subplots(nrows=n_grid, figsize=(5, 5 * n_grid))

    max_prob = np.max(insert_probs)
    norm = colors.Normalize(0, max_prob)
    cmap = colormaps["Spectral"]

    for z_idx in range(n_grid):
        for x_idx in range(n_grid):
            for y_idx in range(n_grid):
                color = cmap(norm(insert_probs[x_idx, y_idx, z_idx]))
                poly = Polygon(parallelogram + x_idx * dx + y_idx * dy, facecolor=color)
                ax[z_idx].add_patch(poly)
        ax[z_idx].set_xlim(0, parallelogram[2, 0] * n_grid)
        ax[z_idx].set_ylim(0, parallelogram[2, 1] * n_grid)
        ax[z_idx].set_xticks([])
        ax[z_idx].set_yticks([])
        ax[z_idx].set_title(f'z={z_idx / n_grid}')
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax[z_idx])
    fig.savefig('input_probs.pdf', bbox_inches='tight')

test_rotation_inversion()
test_pbc_equivalence()
visualize_input_grid()