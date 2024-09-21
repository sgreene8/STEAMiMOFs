"""
Tests whether the Monte Carlo components of the code are sampling the correct distributions.
Each test writes a .txt file with number of samples vs. sampling error. The sampling error should decrease asymptotically as N^{-1/2}.
A log-log plot of number of samples vs. sampling error should asymptotically exhibit a slope of -1/2.
"""
import STEAMiMOFs.structure
import numpy as np
import matplotlib.pyplot as plt
from ase.calculators.lj import LennardJones

def test_H2O_insertion(tot_samples=100000):
    h2oMOF = STEAMiMOFs.structure.MOFWithAds('deployed_model_50_sam.pth', 'RUBTAK01_SL.cif')
    n_MOF_atoms = h2oMOF.n_MOF_atoms
    h2oMOF.insert_h2o(tot_samples)

    n_cart = 5
    n_theta = 5
    n_phi = 10
    o_grid = np.zeros([tot_samples] + [n_cart] * 3)
    h_spherical_grid = np.zeros([tot_samples * 2, n_theta, n_phi]) # (theta, phi) spherical coordinates

    o_frac = h2oMOF._atoms.get_scaled_positions()[n_MOF_atoms::3]
    assert(np.all(o_frac > 0))
    assert(np.all(o_frac < 1))
    h2o_cart = h2oMOF._atoms.get_positions()[n_MOF_atoms:]
    h2o_cart.shape = (tot_samples, 3, 3)
    h_relative_cart = h2o_cart[:, 1:] - h2o_cart[:, 0][:, np.newaxis]
    h_relative_cart /= 0.95720 # OH bond length should be 0.95720 angstroms
    hoh_angle = np.arccos(np.einsum('ac,ac->a', h_relative_cart[:, 0], h_relative_cart[:, 1]))
    h_relative_cart.shape = (2 * tot_samples, 3)
    assert(np.allclose(np.linalg.norm(h_relative_cart, axis=1), 1))
    # H-O-H angle should  be 104.52 degrees
    assert(np.allclose(hoh_angle, 104.52 / 180 * np.pi))
    h_theta = np.arccos(h_relative_cart[:, 2])
    xy_norm = np.linalg.norm(h_relative_cart[:, :2], axis=1)
    h_phi = np.sign(h_relative_cart[:, 1]) * np.arccos(h_relative_cart[:, 0] / xy_norm) + np.pi

    # create the histograms
    o_hist_idx = (o_frac * n_cart).astype(int)
    o_hist_idx = (np.arange(tot_samples, dtype=int), o_hist_idx[:, 0], o_hist_idx[:, 1], o_hist_idx[:, 2])
    np.add.at(o_grid, o_hist_idx, 1)
    o_grid = np.cumsum(o_grid, axis=0) / (np.arange(tot_samples) + 1)[:, np.newaxis, np.newaxis, np.newaxis]
    o_err = np.amax(np.abs(o_grid - 1. / n_cart**3), axis=(1, 2, 3))
    np.savetxt('o_insert_err.txt', o_err)

    h_theta_idx = (h_theta / np.pi * n_theta).astype(int)
    h_phi_idx = (h_phi / 2 / np.pi * n_phi).astype(int)
    h_hist_idx = (np.arange(tot_samples * 2, dtype=int), h_theta_idx, h_phi_idx)
    np.add.at(h_spherical_grid, h_hist_idx, 1 / np.sin(h_theta))
    h_spherical_grid = np.cumsum(h_spherical_grid, axis=0)[::2] / 2 / (np.arange(tot_samples) + 1)[:, np.newaxis, np.newaxis]
    h_err = np.amax(np.abs(h_spherical_grid * 2 / np.pi - 1. / n_theta / n_phi ), axis=(1, 2))
    np.savetxt('h_insert_err.txt', h_err)

def test_H2O_rotation(tot_samples=100000):
    h2oMOF = STEAMiMOFs.structure.MOFWithAds('deployed_model_50_sam.pth', 'RUBTAK01_SL.cif')
    h2oMOF._atoms.calc = LennardJones(epsilon=0, sigma=1)
    n_MOF_atoms = h2oMOF.n_MOF_atoms
    h2oMOF.insert_h2o(1)
    h2o_com = h2oMOF._atoms[n_MOF_atoms:(n_MOF_atoms + 3)].get_center_of_mass()

    n_theta = 5
    n_phi = 10
    spherical_grid = np.zeros([n_theta, n_phi]) # (theta, phi) spherical coordinates
    err = np.zeros(tot_samples)

    for sample in range(tot_samples):
        assert(h2oMOF.rotate_h2o())
        h2o_rel_pos = h2oMOF._atoms[n_MOF_atoms:(n_MOF_atoms + 3)].get_positions() - h2o_com
        h2o_rel_pos /= np.linalg.norm(h2o_rel_pos, axis=1)[:, np.newaxis]
        theta = np.arccos(h2o_rel_pos[:, 2])
        xy_norm = np.linalg.norm(h2o_rel_pos[:, :2], axis=1)
        phi = np.sign(h2o_rel_pos[:, 1]) * np.arccos(h2o_rel_pos[:, 0] / xy_norm) + np.pi
        theta_idx = (theta / np.pi * n_theta).astype(int)
        phi_idx = (phi / 2 / np.pi * n_phi).astype(int)
        np.add.at(spherical_grid, (theta_idx, phi_idx), 1. / np.sin(theta))
        err[sample] = np.amax(np.abs(spherical_grid * 2 / np.pi / 3 / (sample + 1) - 1. / n_theta / n_phi))
    np.savetxt('rotate_err.txt', err)

def test_H2O_translation(tot_samples=100000):
    h2oMOF = STEAMiMOFs.structure.MOFWithAds('deployed_model_50_sam.pth', 'RUBTAK01_SL.cif')
    h2oMOF._atoms.calc = LennardJones(epsilon=0, sigma=1)
    n_MOF_atoms = h2oMOF.n_MOF_atoms
    h2oMOF.insert_h2o(1)
    orig_pos = h2oMOF._atoms[n_MOF_atoms:(n_MOF_atoms + 3)].get_positions()

    n_grid = 5
    grid = np.zeros([n_grid] * 3)
    err = np.zeros(tot_samples)

    for sample in range(tot_samples):
        assert(h2oMOF.translate_h2o())
        new_pos = h2oMOF._atoms[n_MOF_atoms:(n_MOF_atoms + 3)].get_positions()
        trans_vector = new_pos - orig_pos
        orig_pos = new_pos
        assert(np.allclose(trans_vector[1:] - trans_vector[0], 0))
        grid_idx = ((trans_vector[0] + h2oMOF.trans_max) / h2oMOF.trans_max / 2 * n_grid).astype(int)
        
        grid[grid_idx[0], grid_idx[1], grid_idx[2]] += 1
        err[sample] = np.amax(np.abs(grid / (sample + 1) - 1. / n_grid**3))
    np.savetxt('translate_err.txt', err)

def plot_loglog_mc(prefix):
    err = np.genfromtxt(prefix + '.txt')
    fig, ax = plt.subplots()
    max_idx = err.shape[0]
    ax.plot(np.log10(np.arange(max_idx) + 1), np.log10(err))
    # dashed line with slope of -1/2
    ax.plot([0, np.log10(max_idx)], [np.log10(err[-1]) + np.log10(max_idx) / 2, np.log10(err[-1])], 'k--')
    ax.set_ylabel('Log(error)')
    ax.set_xlabel('Log(# samples)')
    fig.savefig(prefix + '.png', bbox_inches='tight')

# test_H2O_insertion()
# plot_loglog_mc('o_insert_err')
# plot_loglog_mc('h_insert_err')

# test_H2O_rotation(10000)
# plot_loglog_mc('rotate_err')

# test_H2O_translation(1000)
# plot_loglog_mc('translate_err')