from pathlib import Path
import torch
from nequip.ase import NequIPCalculator
from ase.calculators.calculator import Calculator, all_changes
import ase.io
from ase import Atoms
import numpy as np

kb = 8.617333262e-5 # Boltzmann Constant, eV/K

class MOFWithAds:
    """
    A class for representing the positions of adsorbate molecules within
    a MOF structure and calculating their energy and forces using an 
    Allegro model.

    """

    def __init__(self, model_path : Path, MOF_path : Path, H2O_path : Path=None, results_path : Path='.', temperature : float=298., h2o_energy : float=0.):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('Using device {}'.format(device.type))

        self._atoms = ase.io.read(MOF_path)
        if model_path is None:
            self._atoms.calc = NullCalculator()
        else:
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

        if H2O_path is not None:
            H2O_atoms = ase.io.read(H2O_path)
            assert(len(H2O_atoms) % 3 == 0)
            self.nh2o = len(H2O_atoms) // 3
        else:
            self.nh2o = 0
        
        self.rot_max = 45. / 180. * np.pi # maximum angle for MC rotation; originally 30 degrees
        self.trans_max = 0.1 # originally 0.05
        self.temperature = temperature
        self.volume = self._atoms.get_volume() # Angstroms^3

        self._free_H2O_en = h2o_energy
        self.current_potential_en = self._atoms.get_potential_energy()
        self._H2O_forces = self._atoms.get_forces(apply_constraint=False)[self.n_MOF_atoms:]
        assert(self._H2O_forces.shape[1] == 3)

        self._traj_file = open(results_path / 'traj.pdb', 'a')

    def insert_h2o(self, number : int=1, keep=True) -> float:
        """
        Insert H2O molecules with random position and orientation in the simulation cell
        Arguments:
            number: The number of H2O molecules (not atoms) to insert
            keep: Whether the inserted molecules should remain in the simulation cell upon return, or whether they should be deleted (as for the NVT+W method)
        Returns:
            The NVT+W probability divided by the fugacity for the insertion.
        """
        cell = self._atoms.get_cell()

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

        en_after = self._atoms.get_potential_energy()
        # Eq 71 in Dubbeldam et al., Molecular Simulation 39, 1253-1292 (2013)
        # Fugacity is not included to allow to calculate multiple isotherm points in one simulation. Multiply by fugacity in atm to get acceptance probability.
        acc_prob = (np.exp(-(en_after - self.current_potential_en - self._free_H2O_en * number) / self.temperature / kb) * self.volume / kb / self.temperature / self.nh2o
            * (1e-10)**3 # Angstroms to meters
            / 1.602e-19 # eV to J
            * 101325 # J/m^3 to atm
        )
        if keep:
            self.current_potential_en = en_after
            self._H2O_forces = self._atoms.get_forces(apply_constraint=False)[self.n_MOF_atoms:]
        else:
            del self._atoms[(-3 * number):]
            self.nh2o -= number
        return acc_prob
    

    def remove_h2o(self, number : int=1, put_back=False) -> float:
        """
        Remove randomly selected molecules from the simulation cell
        Arguments:
            number: The number of H2O molecules (not atoms) to remove
            put_back: If true, the deleted molecules will be returned to the simulation cell upon return (as for the NVT+W) method
        Returns:
            The NVT+W probability divided by the fugacity for the deletion.
        """
        remove_idx = np.random.choice(self.nh2o, number)
        atom_idx = np.reshape((remove_idx * 3 + np.arange(3)[:, np.newaxis]).T, -1) + self.n_MOF_atoms
        h2o_atoms = self._atoms[atom_idx]
        h2o_coords = h2o_atoms.get_positions()
        del self._atoms[atom_idx]
        en_after = self._atoms.get_potential_energy()
        # Eq 72 in Dubbeldam et al., Molecular Simulation 39, 1253-1292 (2013)
        # Fugacity is not included to allow to calculate multiple isotherm points in one simulation. Divide by fugacity in atm to get acceptance probability.
        acc_prob = (np.exp(-(en_after - self.current_potential_en + self._free_H2O_en * number) / self.temperature / kb) / self.volume * kb * self.temperature * self.nh2o
            / (1e-10)**3 # Angstroms to meters
            * 1.602e-19 # eV to J
            / 101325 # J/m^3 to atm
        )
        if put_back:
            self._atoms += h2o_atoms
        else:
            self.nh2o -= number
            self.current_potential_en = en_after
            self._H2O_forces = self._atoms.get_forces(apply_constraint=False)[self.n_MOF_atoms:]
        return acc_prob
        
    
    def rotate_h2o(self, index=None) -> float:
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

        for atom_idx in range(3):
            self._atoms[self.n_MOF_atoms + 3 * index + atom_idx].position = new_h2o_pos[atom_idx]
        en_after = self._atoms.get_potential_energy()

        acc_ratio = np.exp(-(en_after - self.current_potential_en) / self.temperature / kb)
        if np.random.rand(1) < acc_ratio: # move is accepted
            self.current_potential_en = en_after
            self._H2O_forces = self._atoms.get_forces(apply_constraint=False)[self.n_MOF_atoms:]
            return True
        else: # move is rejected
            for atom_idx in range(3):
                self._atoms[self.n_MOF_atoms + 3 * index + atom_idx].position = orig_h2o_pos[atom_idx]
            return False
    
    def translate_h2o(self, index=None) -> float:
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

        # Periodic boundary conditions
        new_scaled = self._atoms.cell.scaled_positions(new_h2o_pos)
        new_scaled[:, new_scaled[0] > 1.] -= 1
        new_scaled[:, new_scaled[0] < 0.] += 1
        new_h2o_pos = self._atoms.cell.cartesian_positions(new_scaled)

        oh_bond_dist = np.linalg.norm(new_h2o_pos[1:] - new_h2o_pos[0], axis=1)
        assert(np.allclose(oh_bond_dist, 0.95720))
        hoh_bond_angle = np.arccos(np.dot(new_h2o_pos[2] - new_h2o_pos[0], new_h2o_pos[1] - new_h2o_pos[0]) / 0.95720**2)
        assert(np.allclose(hoh_bond_angle, 104.52 / 180 * np.pi))

        for atom_idx in range(3):
            self._atoms[self.n_MOF_atoms + 3 * index + atom_idx].position = new_h2o_pos[atom_idx]
        en_after = self._atoms.get_potential_energy()

        acc_ratio = np.exp(-(en_after - self.current_potential_en) / self.temperature / kb)
        if np.random.rand(1) < acc_ratio: # move is accepted
            self.current_potential_en = en_after
            self._H2O_forces = self._atoms.get_forces(apply_constraint=False)[self.n_MOF_atoms:]
            return True
        else: # move is rejected
            for atom_idx in range(3):
                self._atoms[self.n_MOF_atoms + 3 * index + atom_idx].position = orig_h2o_pos[atom_idx]
            return False
        
    def write_to_traj(self, with_MOF=False):
        """
        Saves a snapshot of the current H2O positions to the trajectory file
        Arguments:
            with_MOF: If true, the MOF atoms will be included in the output file
        """
        if with_MOF:
            ase.io.write(self._traj_file, self._atoms, format='proteindatabank')
        else:
            ase.io.write(self._traj_file, self._atoms[self.n_MOF_atoms:], format='proteindatabank')
        self._traj_file.flush()


class NullCalculator(Calculator):
    """
    ASE Calculator that returns zero energy and forces (for debugging purposes)
    """
    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        self.results = {'energy': 0.0, 'forces': np.zeros((len(atoms), 3))}