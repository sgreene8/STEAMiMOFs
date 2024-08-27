from pathlib import Path
import torch
from nequip.ase import NequIPCalculator
import ase.io
from ase import Atoms
import numpy as np

class MOFWithAds:
    """
    A class for representing the positions of adsorbate molecules within
    a MOF structure and calculating their energy and forces using an 
    Allegro model.

    """

    def __init__(self, model_path : Path, structure_path : Path, temperature=298.):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('Using device {}'.format(device.type))

        self._atoms = ase.io.read(structure_path)
        self._atoms.calc = NequIPCalculator.from_deployed_model(
            model_path = model_path,
            species_to_type_name = {
                "H" : "H",
                "C" : "C",
                "O" : "O",
                "Zr" : "Zr",
            }
        )
        self.n_MOF_atoms = len(self._atoms)
        self.nh2o = 0
        self.rot_max = 2 * np.pi # maximum angle for MC rotation
        self.trans_max = 1.
        self.temperature = temperature

        self._free_H2O_en = -.16875515E+02 # vdW-DF

    def insert_h2o(self, number=1):
        """
        Insert H2O molecules with random position and orientation in the simulation cell
        Returns:
            True if the insertion was accepted according to GCMC Metropolis criterion, False otherwise
        """
        cell = self._atoms.get_cell()
        en_before = self._atoms.get_potential_energy()

        O_frac_coords = np.random.rand(number, 3)
        O_cart_coords = np.einsum('vc,nv->nc', cell, O_frac_coords)

        # see https://math.stackexchange.com/questions/1585975/how-to-generate-random-points-on-a-sphere
        OH_vectors = np.random.randn(number, 3)
        is_zero = np.linalg.norm(OH_vectors, axis=1) < 1e-8
        while (np.any(is_zero)):
            num_zero = np.sum(is_zero)
            new_vecs = np.random.randn(num_zero, 3)
            OH_vectors[is_zero] = new_vecs
            is_zero = np.linalg.norm(OH_vectors, axis=1) < 1e-8
        OH_vectors /= np.linalg.norm(OH_vectors, axis=1)[:, np.newaxis]

        arbitrary_vec = OH_vectors[0].copy()
        arbitrary_vec[0] += 0.1
        arbitrary_vec /= np.linalg.norm(arbitrary_vec)
        
        # Make sure arbitrary_vec not parallel or antiparallel to any OH_vectors
        while(np.any(np.abs(np.abs(np.einsum('nc,c->n', OH_vectors, arbitrary_vec)) - 1) < 1e-8)):
            arbitrary_vec[1] += 0.1
            arbitrary_vec /= np.linalg.norm(arbitrary_vec)
        x_vecs = np.cross(OH_vectors, arbitrary_vec)
        x_vecs /= np.linalg.norm(x_vecs, axis=-1)[:, np.newaxis]
        y_vecs = np.cross(OH_vectors, x_vecs)

        H2_angles = np.random.rand(number) * np.pi * 2
        # H-O-H angle should  be 104.52 degrees
        H2_vecs = (np.cos(H2_angles)[:, np.newaxis] * x_vecs + np.sin(H2_angles)[:, np.newaxis] * y_vecs) * np.sin(104.52 / 180 * np.pi) + np.cos(104.52 / 180 * np.pi) * OH_vectors
        H2_vecs /= np.linalg.norm(H2_vecs, axis=1)[:, np.newaxis]
        H2_vecs *= 0.95720
        OH_vectors *= 0.95720 # O-H distance should be 0.95720 angstroms

        h2o_coords = np.zeros([number, 3, 3]) # O, H1, H2
        h2o_coords[:, 0] = O_cart_coords
        h2o_coords[:, 1] = O_cart_coords + OH_vectors
        h2o_coords[:, 2] = O_cart_coords + H2_vecs
        h2o_coords.shape = (number * 3, 3)

        h2o_atoms = Atoms('OHH' * number, positions=h2o_coords)
        self._atoms += h2o_atoms
        self.nh2o += number

        en_after = _atoms.get_potential_energy()
        # Eq 71 in Dubbeldam et al., Molecular Simulation 39, 1253-1292 (2013)
        acc_ratio = np.exp(-(en_after - en_before - self._free_H2O_en * number) / self.temperature) #* 
        if np.random.rand(1) < acc_ratio: # move is accepted
            return True
        else: # move is rejected: remove inserted molecules
            del self._atoms[(-3 * number):]
            return False
        
    
    def rotate_h2o(self, index=None):
        """
        Rotate an h2o molecule in the cell about a randomly chosen axis by a randomly chosen angle.
        Arguments:
            index: If None, an h2o molecule is chosen at random, otherwise the molecule at the specified index is chosen
        Returns:
            True if the rotation was accepted according to Metropolis criterion, False otherwise
        """

        rot_vector = np.random.randn(3)
        rot_vector /= np.linalg.norm(rot_vector)
        rot_angle = np.random.rand(1) * self.rot_max

        if index is None:
            index = np.random.randint(self.nh2o)
        
        h2o = self._atoms[(self.n_MOF_atoms + 3 * index):(self.n_MOF_atoms + 3 * index + 3)]
        com = h2o.get_center_of_mass()
        orig_h2o_pos = h2o.get_positions()

        # see https://stackoverflow.com/questions/6721544/circular-rotation-around-an-arbitrary-axis
        q = np.array([np.cos(rot_angle / 2)] + 3 * [np.sin(rot_angle / 2)])
        q.shape = -1
        q[1:] *= rot_vector
        rot_matrix = np.array([
            [q[0]**2 + q[1]**2 - q[2]**2 - q[3]**2, 2 * (q[1] * q[2] - q[0] * q[3]), 2 * (q[1] * q[3] + q[0] * q[2])],
            [2 * (q[2] * q[1] + q[0] * q[3]), q[0]**2 - q[1]**2 + q[2]**2 - q[3]**2, 2 * (q[2] * q[3] - q[0] * q[1])],
            [2 * (q[3] * q[1] - q[0] * q[2]), 2 * (q[3] * q[2] + q[0] * q[1]), q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2]
        ])

        new_h2o_pos = orig_h2o_pos - com
        new_h2o_pos = new_h2o_pos @ rot_matrix
        new_h2o_pos += com

        # Make sure internal geometry is preserved
        oh_bond_dist = np.linalg.norm(new_h2o_pos[1:] - new_h2o_pos[0], axis=1)
        assert(np.allclose(oh_bond_dist, 0.95720))
        hoh_bond_angle = np.arccos(np.dot(new_h2o_pos[2] - new_h2o_pos[0], new_h2o_pos[1] - new_h2o_pos[0]) / 0.95720**2)
        assert(np.allclose(hoh_bond_angle, 104.52 / 180 * np.pi))

        en_before = self._atoms.get_potential_energy()
        for atom_idx in range(3):
            self._atoms[self.n_MOF_atoms + 3 * index + atom_idx].position = new_h2o_pos[atom_idx]
        en_after = self._atoms.get_potential_energy()

        acc_ratio = np.exp(-(en_after - en_before) / self.temperature)
        if np.random.rand(1) < acc_ratio: # move is accepted
            return True
        else: # move is rejected
            for atom_idx in range(3):
                self._atoms[self.n_MOF_atoms + 3 * index + atom_idx].position = orig_h2o_pos[atom_idx]
            return False
    
    def translate_h2o(self, index=None):
        """
        Translate an h2o molecule in the cell in a randomly chosen direction by a randomly chosen displacement.
        Arguments:
            index: If None, an h2o molecule is chosen at random, otherwise the molecule at the specified index is chosen
        Returns:
            True if the translation was accepted according to Metropolis criterion, False otherwise
        """

        trans_vector = (2 * np.random.rand(3) - 1) * self.trans_max

        if index is None:
            index = np.random.randint(self.nh2o)
        
        h2o = self._atoms[(self.n_MOF_atoms + 3 * index):(self.n_MOF_atoms + 3 * index + 3)]
        orig_h2o_pos = h2o.get_positions()
        new_h2o_pos = orig_h2o_pos + trans_vector

        en_before = self._atoms.get_potential_energy()
        for atom_idx in range(3):
            self._atoms[self.n_MOF_atoms + 3 * index + atom_idx].position = new_h2o_pos[atom_idx]
        en_after = self._atoms.get_potential_energy()

        acc_ratio = np.exp(-(en_after - en_before) / self.temperature)
        if np.random.rand(1) < acc_ratio: # move is accepted
            return True
        else: # move is rejected
            for atom_idx in range(3):
                self._atoms[self.n_MOF_atoms + 3 * index + atom_idx].position = orig_h2o_pos[atom_idx]
            return False
        