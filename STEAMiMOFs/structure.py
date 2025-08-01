from pathlib import Path
import torch
import ase.io
import ase.geometry
from ase import Atoms
import numpy as np
import pickle
import nequip.scripts.deploy
from nequip.data.transforms import TypeMapper
from nequip.data import AtomicData, AtomicDataDict
from typing import Union
import yaml

kb = 8.617333262e-5 # Boltzmann Constant, eV/K

class MOFWithAds:
    """
    A class for representing the positions of adsorbate molecules within
    a MOF structure and calculating their energy and forces using an 
    Allegro model.

    """

    def __init__(self, model_paths : dict, MOF_path : Path, H2O_DFT_path : Path, H2O_path : Path=None, results_path : Path=Path('.'), 
                 temperature : float=298., trans_step : float=0.1, rot_step : float=45, vib_step=0.05,
                 ngrid_O : int=10, ngrid_H1 : int=10, ngrid_H2 : int=10):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('Using device {}'.format(self._device.type))

        self._atoms = ase.io.read(MOF_path)
        self.n_MOF_atoms = len(self._atoms)

        # eq (20), Rappé et al., J. Am. Chem. Soc. 1992, 114, 10024 (UFF)
        # LJ_distance = {'H': 2.886, 'O': 3.5, 'Zr': 3.124, 'C': 3.851} # Angstroms
        # LJ_energy = {'H': 0.044, 'O': 0.060, 'Zr': 0.069, 'C': 0.105} # kcal/mol
        # self._mof_LJ_eps = np.array([LJ_energy[atom.symbol] for atom in self._atoms]) * 0.043 # eV
        # self._mof_LJ_sigma = np.array([LJ_distance[atom.symbol] for atom in self._atoms])

        self._ngrid_O = ngrid_O
        self._ngrid_H1 = ngrid_H1
        self._ngrid_H2 = ngrid_H2

        if H2O_path is not None:
            H2O_atoms = ase.io.read(H2O_path, index=-1)
            assert(len(H2O_atoms) % 3 == 0)
            self.nh2o = len(H2O_atoms) // 3
            self._atoms += H2O_atoms
        else:
            self.nh2o = 0
        
        if model_paths is None:
            self._models = None
        else:
            # Set up potentials
            r_max = 0
            tmp_models = []
            n_models = len(model_paths.keys())
            model_ranges = np.zeros([n_models, 2])
            for idx, path in enumerate(model_paths.keys()):
                model_ranges[idx] = model_paths[path]
               # Code adapted from nequip.ase.nequip_calculator
                this_model, metadata = nequip.scripts.deploy.load_deployed_model(
                                        model_path=path,
                                        device=self._device,
                                        set_global_options="warn",
                                )
                this_r_max = float(metadata[nequip.scripts.deploy.R_MAX_KEY])
                r_max = this_r_max
                # TO-DO: assert consistency in r_max across models
                tmp_models.append(this_model)
            
            # TO-DO: assert disjoint model ranges 
            self._r_max = r_max
            self._models = tmp_models
            self._model_ranges = model_ranges
            
            # build typemapper
            species_to_type_name = {
                        "H" : "H",
                        "C" : "C",
                        "O" : "O",
                        "Zr" : "Zr",
                    }
            type_names = metadata[nequip.scripts.deploy.TYPE_NAMES_KEY].split(" ")
            type_name_to_index = {n: i for i, n in enumerate(type_names)}
            chemical_symbol_to_type = {
                sym: type_name_to_index[species_to_type_name[sym]]
                for sym in ase.data.chemical_symbols
                if sym in type_name_to_index
            }
            assert(len(chemical_symbol_to_type) == len(type_names))
            self._transform = TypeMapper(chemical_symbol_to_type=chemical_symbol_to_type)
        
        if H2O_DFT_path is None:
            raise ValueError('H2O DFT path must be provided')
        else:
            with open(H2O_DFT_path, 'rb') as file:
                h2o_data = yaml.safe_load(file)
                self._free_H2O_en = float(h2o_data['H2O_energy'])
                self._empty_MOF_en = float(h2o_data['MOF_energy']) # TO-DO: fold this into NNP argument
                self._rOH = h2o_data['rOH']
                self._aHOH = h2o_data['aHOH'] / 180. * np.pi
        
        self.rot_step = abs(rot_step) / 180. * np.pi
        assert(self.rot_step < np.pi)
        self.trans_step = abs(trans_step)
        self.vib_step = abs(vib_step)
        self.temperature = temperature
        self.volume = self._atoms.get_volume() # Angstroms^3
        self.rng = np.random.default_rng()

        self.current_potential_en, self._H2O_forces = self._evaluate_potential()
        assert(self._H2O_forces.shape[1] == 3)

        self._traj_file = open(results_path / 'traj.xyz', 'a')
    
    def _evaluate_potential(self):
        if self.nh2o == 0:
            energy = self._empty_MOF_en
            forces = np.zeros([0, 3])
        elif self._models is not None:
            model_idx = (self._model_ranges[:, 0] <= self.nh2o) and (self._model_ranges[:, 1] >= self.nh2o)
            model_idx = np.nonzero(model_idx)[0]
            assert(model_idx.shape[0] == 1)
            model_idx = model_idx[0]

            data = AtomicData.from_ase(atoms=self._atoms, r_max=self._r_max)
            data = self._transform(data)
            data = data.to(self._device)
            data = AtomicData.to_AtomicDataDict(data)
            out = self._models[model_idx](data)
            energy = out[AtomicDataDict.TOTAL_ENERGY_KEY][0, 0].detach().cpu().numpy()
            forces = out[AtomicDataDict.FORCE_KEY][self.n_MOF_atoms:].detach().cpu().numpy()
        else:
            energy = 0.0
            forces = np.zeros([self.nh2o * 3, 3])
        return energy, forces
    
    def _calculate_torque(self, index : int):
        """
        Calculate the torque on a rigid adsorbate and its center of mass
        """
        h2o = self._atoms[(self.n_MOF_atoms + 3 * index):(self.n_MOF_atoms + 3 * index + 3)]
        com = h2o.get_center_of_mass()
        h2o_pos = h2o.get_positions()
        h2o_rel = orig_h2o_pos - orig_com
        torque = np.sum(np.cross(orig_h2o_rel, self._H2O_forces[(3 * index):(3 * index + 3)]), axis=0)
        return com, torque
    
    def check_h2o_geom(self):
        """
        For debugging purposes, check that each H2O molecule has the correct O-H bond lengths and H-O-H bond angle
        """
        for index in range(self.nh2o):
            print('Checking {} geometry'.format(index))
            h2o = self._atoms[(self.n_MOF_atoms + 3 * index):(self.n_MOF_atoms + 3 * index + 3)]
            h2o_pos = h2o.get_positions()
            oh_bond_dist = np.linalg.norm(h2o_pos[1:] - h2o_pos[0], axis=1)
            assert(np.allclose(oh_bond_dist, self._rOH))
            hoh_bond_angle = np.arccos(np.dot(h2o_pos[2] - h2o_pos[0], h2o_pos[1] - h2o_pos[0]) / self._rOH**2)
            assert(np.allclose(hoh_bond_angle, self._aHOH))

    def insert_h2o(self, number : int=1, keep=True, exclusion_radius=1.0) -> float:
        """
        Insert H2O molecules with random position and orientation in the simulation cell
        Arguments:
            number: The number of H2O molecules (not atoms) to insert
            keep: Whether the inserted molecules should remain in the simulation cell upon return, or whether they should be deleted (as for the NVT+W method)
            exclusion_radius: Insertions that place atoms within this distance of other atoms will be rejected and re-tried
        Returns:
            The NVT+W probability divided by the fugacity for the insertion.
        """
        cell = self._atoms.get_cell()

        n_success = 0
        O_cart_coords_keep = np.zeros([n_success, 3])
        while (n_success < number):
            O_frac_coords = self.rng.random((number * 2, 3))
            O_cart_coords = np.einsum('vc,nv->nc', cell, O_frac_coords)
            _, distances = ase.geometry.get_distances(O_cart_coords, self._atoms.arrays['positions'], cell=cell, pbc=self._atoms.pbc)
            keep_bool = distances > exclusion_radius
            keep_coords = O_cart_coords[keep_bool][:(number - n_success)]
            n_success += keep_coords.shape[0]
            O_cart_coords_keep = np.append(O_cart_coords_keep, keep_coords, axis=0)
        O_cart_coords = O_cart_coords_keep

        # see https://math.stackexchange.com/questions/1585975/how-to-generate-random-points-on-a-sphere
        n_success = 0
        OH_vectors_keep = np.zeros([n_success, 3])
        while (n_success < number):
            OH_vectors = self.rng.standard_normal((number * 2, 3))
            is_zero = np.linalg.norm(OH_vectors, axis=1) < 1e-8
            while (np.any(is_zero)):
                num_zero = np.sum(is_zero)
                new_vecs = self.rng.standard_normal(num_zero, 3)
                OH_vectors[is_zero] = new_vecs
                is_zero = np.linalg.norm(OH_vectors, axis=1) < 1e-8
            OH_vectors /= np.linalg.norm(OH_vectors, axis=1)[:, np.newaxis]
            H_coords = O_cart_coords + self._rOH * OH_vectors
            _, distances = ase.geometry.get_distances(H_coords, self._atoms.arrays['positions'], cell=cell, pbc=self._atoms.pbc)
            keep_bool = distances > exclusion_radius
            keep_vecs = OH_vectors[keep_bool][:(number - n_success)]
            n_success += keep_vecs.shape[0]
            OH_vectors_keep = np.append(OH_vectors_keep, keep_vecs, axis=0)
        OH_vectors = OH_vectors_keep

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

        n_success = 0
        H2_vecs_keep = np.zeros([n_success, 3])
        while (n_success < number):
            H2_angles = self.rng.random(number * 2) * np.pi * 2
        
            H2_vecs = (np.cos(H2_angles)[:, np.newaxis] * x_vecs + np.sin(H2_angles)[:, np.newaxis] * y_vecs) * np.sin(self._aHOH) + np.cos(self._aHOH) * OH_vectors
            H2_vecs /= np.linalg.norm(H2_vecs, axis=1)[:, np.newaxis]
            H2_vecs *= self._rOH

            H_coords = O_cart_coords + H2_vecs
            _, distances = ase.geometry.get_distances(H_coords, self._atoms.arrays['positions'], cell=cell, pbc=self._atoms.pbc)
            keep_bool = distances > exclusion_radius
            keep_vecs = H2_vecs[keep_bool][:(number - n_success)]
            n_success += keep_vecs.shape[0]
            H2_vecs_keep = np.append(H2_vecs_keep, keep_vecs, axis=0)
        H2_vecs = H2_vecs_keep

        OH_vectors *= self._rOH

        h2o_coords = np.zeros([number, 3, 3]) # O, H1, H2
        h2o_coords[:, 0] = O_cart_coords
        h2o_coords[:, 1] = O_cart_coords + OH_vectors
        h2o_coords[:, 2] = O_cart_coords + H2_vecs
        h2o_coords.shape = (number * 3, 3)

        h2o_atoms = Atoms('OHH' * number, positions=h2o_coords)
        self._atoms += h2o_atoms
        self.nh2o += number
        
        """
        acc_prob = np.zeros(number)
        rosenbluth_wt = 1
        for h2o_index in range(number):
            h2o_coords, rosen = self._sample_insert_position(n_grid=20)
            rosenbluth_wt *= rosen
            h2o_atoms = Atoms('OHH', positions=h2o_coords)
            self._atoms += h2o_atoms
            self.nh2o += 1
        """
        en_after, forces_after = self._evaluate_potential()

        # Eq 71 in Dubbeldam et al., Molecular Simulation 39, 1253-1292 (2013) * with modification for nonuniform probability
        # Fugacity is not included to allow to calculate multiple isotherm points in one simulation. Multiply by fugacity in atm to get acceptance probability.
        acc_prob = (np.exp(-(en_after - self.current_potential_en - self._free_H2O_en * number) / self.temperature / kb) * self.volume / kb / self.temperature / self.nh2o
            * (1e-10)**3 # Angstroms to meters
            / 1.602e-19 # eV to J
            * 101325 # J/m^3 to atm
            # * rosenbluth_wt
        )
        if keep:
            self.current_potential_en = en_after
            self._H2O_forces = forces_after
        else:
            del self._atoms[(-3 * number):]
            self.nh2o -= number
        return acc_prob
    
    def insert_h2o_lowen(self, number : int=1, n_sample=500, exclusion_radius=1.0):
        for ins_idx in range(number):
            cell = self._atoms.get_cell()

            O_frac_coords = self.rng.random((n_sample, 3))
            O_cart_coords = np.einsum('vc,nv->nc', cell, O_frac_coords)

            # see https://math.stackexchange.com/questions/1585975/how-to-generate-random-points-on-a-sphere
            OH_vectors = self.rng.standard_normal((n_sample, 3))
            is_zero = np.linalg.norm(OH_vectors, axis=1) < 1e-8
            while (np.any(is_zero)):
                num_zero = np.sum(is_zero)
                new_vecs = self.rng.standard_normal(num_zero, 3)
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

            H2_angles = self.rng.random(n_sample) * np.pi * 2
            
            H2_vecs = (np.cos(H2_angles)[:, np.newaxis] * x_vecs + np.sin(H2_angles)[:, np.newaxis] * y_vecs) * np.sin(self._aHOH) + np.cos(self._aHOH) * OH_vectors
            H2_vecs /= np.linalg.norm(H2_vecs, axis=1)[:, np.newaxis]
            H2_vecs *= self._rOH
            OH_vectors *= self._rOH

            h2o_coords = np.zeros([n_sample, 3, 3]) # O, H1, H2
            h2o_coords[:, 0] = O_cart_coords
            h2o_coords[:, 1] = O_cart_coords + OH_vectors
            h2o_coords[:, 2] = O_cart_coords + H2_vecs

            h2o_coords.shape = (n_sample * 3, 3)
            _, distances = ase.geometry.get_distances(h2o_coords, self._atoms.arrays['positions'], cell=cell, pbc=self._atoms.pbc)
            h2o_coords.shape = (n_sample, 3, 3)
            distances.shape = (n_sample, 3, self.n_MOF_atoms)
            keep = np.all(distances > exclusion_radius, axis=(1, 2))
            h2o_coords = h2o_coords[keep]
            n_keep = h2o_coords.shape[0]

            h2o_atoms = Atoms('OHH', positions=h2o_coords[0])
            self._atoms += h2o_atoms
            self.nh2o += 1
            min_energy = np.inf
            for sample_idx in range(n_keep):
                for atom_idx in range(3):
                    self._atoms[-3 + atom_idx].position = h2o_coords[sample_idx, atom_idx]
                en_after, forces_after = self._evaluate_potential()
                if en_after < min_energy:
                    min_energy = en_after
                    argmin_energy = sample_idx
            for atom_idx in range(3):
                self._atoms[-3 + atom_idx].position = h2o_coords[argmin_energy, atom_idx]
            
        en_after, forces_after = self._evaluate_potential()
        self.current_potential_en = en_after
        self._H2O_forces = forces_after

    
    def _sample_insert_position(self, n_grid : int=10):
        offset = self.rng.random(3)
        O_cart_pos, O_probs = self._insert_probs_grid(offset=offset, n_grid=self._ngrid_O)
        O_idx = self._sample_multidimensional_array(O_probs)
        O_cart_pos = O_cart_pos[O_idx]
        rosen = np.sum(O_probs)

        H1_cart_pos, H1_probs = self._insert_probs_sphere(O_cart_pos, n_grid=self._ngrid_H1)
        H1_idx = self._sample_multidimensional_array(H1_probs)
        H1_cart_pos = H1_cart_pos[H1_idx]
        rosen *= np.sum(probs)

        H2_cart_pos, H2_probs = self._insert_probs_ring(O_cart_pos, H1_cart_pos, n_grid=self._ngrid_H2)
        H2_idx = self._sample_multidimensional_array(H2_probs)
        H2_cart_pos = H2_cart_pos[H2_idx]
        rosen *= np.sum(probs)

        h2o_coords = np.array([O_cart_pos, H1_cart_pos, H2_cart_pos])
        # total_probability = O_probs[O_idx] * H1_probs[H1_idx] * H2_probs[H2_idx] * self._ngrid_O**3 * self._ngrid_H1 * self._ngrid_H2
        return h2o_coords, rosen
    
    def _backcalculate_insert_prob(self, cart_coords, n_grid : int=10):
        """
        Calculate the normalized Boltzmann probability associated with inserting a molecule at a specified position.
        Arguments:
            cart_coords: The Cartesian coordinates of the inserted molecule
        """
        cell = self._atoms.get_cell()
        max_cell_len = np.max(np.linalg.norm(cell, axis=1))
        frac_coords = self._atoms.cell.scaled_positions(cart_coords)
        frac_coords[:, frac_coords[0] > 1.] -= 1
        frac_coords[:, frac_coords[0] < 0.] += 1
        cart_coords = self._atoms.cell.cartesian_positions(frac_coords)
        offset = frac_coords[0] - np.round(frac_coords[0] * self._ngrid_O) / self._ngrid_O
        O_cart_pos, O_probs = self._insert_probs_grid(n_grid=self._ngrid_O, offset=offset)
        distances = np.linalg.norm(cart_coords[0] - O_cart_pos, axis=-1)
        O_idx = np.argmin(distances)
        O_idx = np.unravel_index(O_idx, distances.shape)
        assert(distances[O_idx] < max_cell_len / self._ngrid_O)
        O_cart_pos = O_cart_pos[O_idx]
        rosen = np.sum(O_probs)

        H1_cart_pos, H1_probs = self._insert_probs_sphere(cart_coords[0], n_grid=self._ngrid_H1)
        distances = np.linalg.norm(cart_coords[1] - H1_cart_pos, axis=-1)
        H1_idx = np.argmin(distances)
        H1_idx = np.unravel_index(H1_idx, distances.shape)
        assert(distances[H1_idx] < self._rOH * 2 * np.pi / self._ngrid_H1)
        rosen *= np.sum(probs)

        H2_cart_pos, H2_probs = self._insert_probs_ring(cart_coords[0], cart_coords[1], n_grid=self._ngrid_H2, rotation_origin=cart_coords[2])
        distances = np.linalg.norm(cart_coords[2] - H2_cart_pos, axis=-1)
        H2_idx = np.argmin(distances)
        assert(H2_idx == 0) # because rotation_origin was provided as coordinates of H atom in _insert_probs_ring
        assert(distances[H2_idx] < 0.1)
        H2_cart_pos = H2_cart_pos[H2_idx]
        rosen *= np.sum(probs)

        # return O_probs[O_idx] * H1_probs[H1_idx] * H2_probs[H2_idx] * self._ngrid_O**3 * self._ngrid_H1 * self._ngrid_H2
        return rosen

    
    def _insert_probs_ring(self, origin, satellite, radius : float, angle : float, n_grid : int=10, rotation_origin=None):
        """
        Calculate the normalized Boltzmann probabilities associated with inserting a particle at each of the points on a 
        ring at a certain radius from the origin with a certain bond angle with respect to the origin and satellite points
        Arguments:
            origin: The Cartesian coordinates of the central atom
            satellite: The Cartesian coordinates of the existing atom already placed
            radius: The distance from the origin that the new atom should be
            angle: The target bond angle from satellite to origin to the new atom
            n_grid: The number of grid points around the ring
            rotation_origin: If provided, defines the zero angle of rotation on the ring. 
        Returns:
            The Cartesian coordinates and normalized Boltzmann probabilities for each point on the ring
        """
        cell = self._atoms.get_cell()
    
        bond_vector = satellite - origin
        bond_vector /= np.linalg.norm(bond_vector)

        if rotation_origin is None:
            arbitrary_vec = bond_vector.copy()
            arbitrary_vec[0] += 0.1
            arbitrary_vec /= np.linalg.norm(arbitrary_vec)

            # Make sure arbitrary_vec not parallel to bond_vector
            dprod = np.dot(bond_vector, arbitrary_vec)
            if (abs(dprod - 1) < 1e-8):
                arbitrary_vec[1] += 0.1
                arbitrary_vec /= np.linalg.norm(arbitrary_vec)
                dprod = np.dot(bond_vector, arbitrary_vec)
                assert(abs(dprod - 1) > 1e-8)
            
            x_vec = np.cross(bond_vector, arbitrary_vec)
        else:
            x_vec = rotation_origin - origin
            x_vec -= np.dot(x_vec, bond_vector) * bond_vector
        
        x_vec /= np.linalg.norm(x_vec)
        assert(np.dot(x_vec, bond_vector) < 1e-8)
        y_vec = np.cross(bond_vector, x_vec)
        assert(abs(np.linalg.norm(y_vec) - 1) < 1e-8)

        angles = np.linspace(0, np.pi * 2, num=n_grid, endpoint=False)
        new_vecs = (np.cos(angles)[:, np.newaxis] * x_vec + np.sin(angles)[:, np.newaxis] * y_vec) * np.sin(self._aHOH) + np.cos(self._aHOH) * bond_vector
        assert(np.allclose(np.arccos(np.einsum('vc,c->v', new_vecs, bond_vector)), self._aHOH))
        cart_coords = origin + new_vecs * self._rOH

        frac_grid = np.linalg.solve(cell.T, cart_coords.T).T
        probs = self._calc_LJ_Boltzmann_probs(frac_grid, 'H')
        return cart_coords, probs
    
    def _insert_probs_sphere(self, origin, radius : float, n_grid : int=10) -> np.ndarray:
        """ 
        Calculate the normalized Boltzmann probabilities associated with inserting a particle at each of the points on a spherical grid
        surrounding the origin point with the given radius
        Arguments:
            origin: The origin point in Cartesian coordinates
            radius: The radius of the spherical surface on which to construct the grid
            n_grid: The number of grid points along each of the 2 angular spherical coordinates
        Returns:
            The Cartesian coordinates and normalized Boltzmann probabilities for each point on the grid
        """
        cell = self._atoms.get_cell()

        theta_grid = np.linspace(0, np.pi, num=n_grid, endpoint=False)
        phi_grid = np.linspace(0, np.pi * 2, num=n_grid, endpoint=False)
        sphere_grid = np.meshgrid(theta_grid, phi_grid)
        # Jacobian = np.sin(sphere_grid[0] + np.pi / n_grid / 2)
        sphere_grid[0].shape = -1
        sphere_grid[1].shape = -1
        # shape (3, n_grid**2)
        cart_coords = origin[:, np.newaxis] + radius * np.array([np.sin(sphere_grid[0]) * np.cos(sphere_grid[1]), np.sin(sphere_grid[0]) * np.sin(sphere_grid[1]), np.cos(sphere_grid[0])])
        frac_grid = np.linalg.solve(cell.T, cart_coords).T

        probs = self._calc_LJ_Boltzmann_probs(frac_grid, 'H')
        probs.shape = (n_grid, n_grid)
        # probs *= Jacobian
        # probs /= np.sum(probs)
        cart_coords = cart_coords.T
        cart_coords.shape = (n_grid, n_grid, 3)
        assert(np.allclose(np.linalg.norm(cart_coords - origin, axis=-1), radius))
        return cart_coords, probs

    
    def _insert_probs_grid(self, n_grid : int=10, offset=0) -> np.ndarray:
        """
        Calculate the normalized Boltzmann probabilities associated with inserting a particle at each of the points on a grid 
            within the cell (rectilinear in fractional coordinates), according to a H-atom UFF LJ potential
        Arguments:
            n_grid: The number of grid points to use in each of the 3 dimensions
            offset(scalar or 3-d array): A shift for the grid points in fractional coordinates
        Returns:
            The Cartesian coordinates and normalized Boltzmann probabilities for each point on the grid
        """

        frac_grid = np.mgrid[0:n_grid, 0:n_grid, 0:n_grid]
        frac_grid.shape = (3, -1)
        frac_grid = frac_grid.T
        frac_grid = frac_grid * 1.0 / n_grid
        frac_grid += offset
        frac_grid[frac_grid < 0] += 1

        probs = self._calc_LJ_Boltzmann_probs(frac_grid, 'O')
        probs.shape = (n_grid, n_grid, n_grid)

        cart_coords = np.einsum('vc,gv->gc', self._atoms.get_cell(), frac_grid)
        cart_coords.shape = (n_grid, n_grid, n_grid, 3)
        return cart_coords, probs

    def _calc_LJ_Boltzmann_probs(self, frac_coords, atom_type):
        """ 
        Calculate the Boltzmann factors associated with inserting a particle at each of the points
        in the input array
        Arguments:
            frac_coords: The fractional coordinates of each of the points, with shape (*, 3)
        Returns:
            The Boltzmann factors
        """
        cell = self._atoms.get_cell()

        mof_frac_coords = self._atoms.get_scaled_positions()
        differences = mof_frac_coords[:, np.newaxis] - frac_coords[np.newaxis, :]
        differences -= np.round(differences) # PBC
        cart_differences = np.einsum('vc,agv->agc', cell, differences)
        # shape: (n_atoms, frac_coords.shape[0])
        distances = np.linalg.norm(cart_differences, axis=-1)

        O_charge = -0.834
        H_charge = -O_charge / 2
        O_dist = distances[(self.n_MOF_atoms + 0)::3]
        H1_dist = distances[(self.n_MOF_atoms + 1)::3]
        H2_dist = distances[(self.n_MOF_atoms + 2)::3]

        if atom_type == 'O':
            tip3p_eps = 595.0**2 / 4 / 582e3 * 0.043
            tip3p_sigma = (2 * 582e3 / 595.0)**(1./6)
            sigma_mix = 0.5 * (self._mof_LJ_sigma + tip3p_sigma)
            eps_mix = (tip3p_eps * self._mof_LJ_eps)**0.5
            mof_dist = distances[:self.n_MOF_atoms]
            scaled_dist = sigma_mix[:, None] / mof_dist
            lj_mof = np.sum(eps_mix[:, None] * (-2 * scaled_dist**6 + scaled_dist**12), axis=0)
            lj_OO = np.sum(tip3p_eps * (-2 * (tip3p_sigma / O_dist)**6 + (tip3p_sigma / O_dist)**12), axis=0)
            self_charge = O_charge
        elif atom_type == 'H':
            lj_mof = 0
            lj_OO = 0
            self_charge = H_charge

        coulomb_h2o = 332.1 * 0.043 * self_charge * np.sum(O_charge / O_dist + H_charge /  H1_dist + H_charge /  H2_dist, axis=0)
        overlap = coulomb_h2o < -(50 * kb * self.temperature) # detect overlaps; np.exp blows up at 100
        coulomb_h2o[overlap] = 1e20

        total_energy = lj_mof + lj_OO + coulomb_h2o
        boltzmann = np.exp(-total_energy / self.temperature / kb)
        # if normalize:
        #     boltzmann /= np.sum(boltzmann)
        return boltzmann
    
    def _sample_multidimensional_array(self, arr):
        cs = np.cumsum(arr)
        rand_num = self.rng.random() # [0, 1)
        sampled_idx = np.sum(cs < rand_num)
        return np.unravel_index(sampled_idx, arr.shape)

    def remove_h2o(self, number : int=1, put_back=False, index : Union[int, list[int]]=None) -> float:
        """
        Remove randomly selected molecules from the simulation cell
        Arguments:
            number: The number of H2O molecules (not atoms) to remove
            put_back: If true, the deleted molecules will be returned to the simulation cell upon return (as for the NVT+W) method
            index: If none, indices will be selected randomly. Otherwise, molecules at the specified indices will be removed.
        Returns:
            The NVT+W probability divided by the fugacity for the deletion.
        """
        if index is None:
            remove_idx = self.rng.choice(self.nh2o, number)
        else:
            remove_idx = np.array(index)
        atom_idx = np.reshape((remove_idx * 3 + np.arange(3)[:, np.newaxis]).T, -1) + self.n_MOF_atoms
        original_atoms = self._atoms.copy()
        del self._atoms[atom_idx]
        en_after, forces_after = self._evaluate_potential()

        # rosenbluth_wt = self._backcalculate_insert_prob(original_atoms[atom_idx].get_positions(), n_grid=20)
        # Eq 72 in Dubbeldam et al., Molecular Simulation 39, 1253-1292 (2013)
        # Fugacity is not included to allow to calculate multiple isotherm points in one simulation. Divide by fugacity in atm to get acceptance probability.
        acc_prob = (np.exp(-(en_after - self.current_potential_en + self._free_H2O_en * number) / self.temperature / kb) / self.volume * kb * self.temperature * self.nh2o
            / (1e-10)**3 # Angstroms to meters
            * 1.602e-19 # eV to J
            / 101325 # J/m^3 to atm
            # / rosenbluth_wt
        )
        if put_back:
            self._atoms = original_atoms
        else:
            self.nh2o -= number
            self.current_potential_en = en_after
            self._H2O_forces = forces_after
        return acc_prob
    

    def eval_H2O_energies(self) -> np.ndarray:
        """
        Evaluate the energy required to remove each H2O molecule in the simulation cell, for debugging purposes
        Returns:
            The energy of each H2O molecule in the simulation cell
        """
        energies = np.zeros(self.nh2o)
        original_atoms = self._atoms.copy()
        for idx in range(self.nh2o):
            del self._atoms[idx]
            en_after, _ = self._evaluate_potential()
            energies[idx] = en_after - self.current_potential_en + self._free_H2O_en
            self._atoms = original_atoms.copy()
        return energies
    
    def rotate_h2o(self, index : int=-1, sampling : str="MALA") -> bool:
        """
        Rotate an h2o molecule in the cell about a randomly chosen axis by a randomly chosen angle.
        Arguments:
            index: If None, an h2o molecule is chosen at random, otherwise the molecule at the specified index is chosen
            sampling: Monte Carlo sampling scheme to use, either MALA (Metropolis-adjusted Langevin) or Metropolis
        Returns:
            True if the rotation was accepted according to Metropolis-Hastings criterion, False otherwise
        """

        if not(sampling == "MALA" or sampling == "Metropolis"):
            raise ValueError("MOFWithAds.rotate_h2o sampling argument must be either 'MALA' or 'Metropolis'")

        rot_vector = self.rng.standard_normal(3)
        rot_angle = self.rng.standard_normal() * self.rot_step
        while(abs(rot_angle) > np.pi):
            print('angle too large')
            rot_angle = self.rng.standard_normal() * self.rot_step
        rot_matrix_rand = _calc_rot_matrix(rot_angle, rot_vector)

        if index == -1:
            index = self.rng.integers(self.nh2o)
        
        h2o = self._atoms[(self.n_MOF_atoms + 3 * index):(self.n_MOF_atoms + 3 * index + 3)]
        orig_com = h2o.get_center_of_mass()
        orig_h2o_pos = h2o.get_positions()
        orig_OH_bond_dist = np.linalg.norm(orig_h2o_pos[1:] - orig_h2o_pos[0], axis=1)
        orig_HOH_bond_angle = np.arccos(np.dot(orig_h2o_pos[2] - orig_h2o_pos[0], orig_h2o_pos[1] - orig_h2o_pos[0]) / orig_OH_bond_dist[0] / orig_OH_bond_dist[1])
        orig_h2o_rel = orig_h2o_pos - orig_com
        new_h2o_rel = orig_h2o_rel

        if sampling=="MALA":
            orig_torque = np.sum(np.cross(orig_h2o_rel, self._H2O_forces[(3 * index):(3 * index + 3)]), axis=0)
            torque_angle = np.linalg.norm(orig_torque) * self.rot_step**2 / 2 / kb / self.temperature
            rot_matrix_torque = _calc_rot_matrix(torque_angle, orig_torque)
            new_h2o_rel = orig_h2o_rel @ rot_matrix_torque.T

        new_h2o_rel = new_h2o_rel @ rot_matrix_rand.T
        new_h2o_pos = orig_com + new_h2o_rel

        for atom_idx in range(3):
            self._atoms[self.n_MOF_atoms + 3 * index + atom_idx].position = new_h2o_pos[atom_idx]

        # Make sure internal geometry is preserved
        oh_bond_dist = np.linalg.norm(new_h2o_pos[1:] - new_h2o_pos[0], axis=1)
        assert(np.allclose(oh_bond_dist, orig_OH_bond_dist))
        hoh_bond_angle = np.arccos(np.dot(new_h2o_pos[2] - new_h2o_pos[0], new_h2o_pos[1] - new_h2o_pos[0]) / oh_bond_dist[0] / oh_bond_dist[1])
        assert(np.allclose(hoh_bond_angle, orig_HOH_bond_angle))
        new_com = self._atoms[(self.n_MOF_atoms + 3 * index):(self.n_MOF_atoms + 3 * index + 3)].get_center_of_mass()
        assert(np.allclose(new_com, orig_com))

        en_after, forces_after = self._evaluate_potential()
        exp_argument = -(en_after - self.current_potential_en) / self.temperature / kb # numerically more stable to calculate exp(a+b) than exp(a) exp(b)

        if sampling=="MALA":
            new_torque = np.sum(np.cross(new_h2o_rel, forces_after[(3 * index):(3 * index + 3)]), axis=0)
            new_torque_angle = np.linalg.norm(new_torque) * self.rot_step**2 / 2 / kb / self.temperature
            rot_matrix_newtorque = _calc_rot_matrix(new_torque_angle, new_torque)
            new_rel_torqued = new_h2o_rel @ rot_matrix_newtorque.T
            new_rot_angle = _calc_rot_angle(new_rel_torqued, orig_h2o_rel)
            # new_prob = np.exp(-rot_angle**2 / 2 / self.rot_step**2) # Prob. of sampling new_h2o_pos given orig_h2o_pos
            # old_prob = np.exp(-new_rot_angle**2 / 2 / self.rot_step**2) # Prob. of sampling orig_h2o_pos given new_h2o_pos
            # proposal_ratio = old_prob / new_prob
            exp_argument += (-new_rot_angle**2 + rot_angle**2) / 2 / self.rot_step**2

        acc_ratio = np.exp(exp_argument)
        if self.rng.random(1) < acc_ratio: # move is accepted
            self.current_potential_en = en_after
            self._H2O_forces = forces_after
            return True
        else: # move is rejected
            for atom_idx in range(3):
                self._atoms[self.n_MOF_atoms + 3 * index + atom_idx].position = orig_h2o_pos[atom_idx]
            return False

    def translate_h2o(self, index : int=None, sampling : str="MALA") -> bool:
        """
        Translate an h2o molecule in the cell in a randomly chosen direction by a randomly chosen displacement.
        Arguments:
            index: If None, an h2o molecule is chosen at random, otherwise the molecule at the specified index is chosen
            sampling: Monte Carlo sampling scheme to use, either MALA (Metropolis-adjusted Langevin) or Metropolis
        Returns:
            True if the translation was accepted according to Metropolis-Hastings criterion, False otherwise
        """

        if not(sampling == "MALA" or sampling == "Metropolis"):
            raise ValueError("MOFWithAds.rotate_h2o sampling argument must be either 'MALA' or 'Metropolis'")
        
        trans_vector = self.rng.standard_normal(3) * self.trans_step

        if index is None:
            index = self.rng.integers(self.nh2o)
        
        h2o = self._atoms[(self.n_MOF_atoms + 3 * index):(self.n_MOF_atoms + 3 * index + 3)]
        orig_com = h2o.get_center_of_mass()
        orig_h2o_pos = h2o.get_positions()
        orig_OH_bond_dist = np.linalg.norm(orig_h2o_pos[1:] - orig_h2o_pos[0], axis=1)
        orig_HOH_bond_angle = np.arccos(np.dot(orig_h2o_pos[2] - orig_h2o_pos[0], orig_h2o_pos[1] - orig_h2o_pos[0]) / orig_OH_bond_dist[0] / orig_OH_bond_dist[1])

        new_h2o_pos = orig_h2o_pos + trans_vector
        if sampling=="MALA":
            orig_com_force = np.sum(self._H2O_forces[(3 * index):(3 * index + 3)], axis=0) # Force on center of mass
            new_h2o_pos += self.trans_step**2 / 2 / kb / self.temperature * orig_com_force

        # Periodic boundary conditions
        new_scaled = self._atoms.cell.scaled_positions(new_h2o_pos)
        new_scaled[:, new_scaled[0] > 1.] -= 1
        new_scaled[:, new_scaled[0] < 0.] += 1
        new_h2o_pos = self._atoms.cell.cartesian_positions(new_scaled)

        for atom_idx in range(3):
            self._atoms[self.n_MOF_atoms + 3 * index + atom_idx].position = new_h2o_pos[atom_idx]

        # Make sure internal geometry is preserved
        oh_bond_dist = np.linalg.norm(new_h2o_pos[1:] - new_h2o_pos[0], axis=1)
        assert(np.allclose(oh_bond_dist, orig_OH_bond_dist))
        hoh_bond_angle = np.arccos(np.dot(new_h2o_pos[2] - new_h2o_pos[0], new_h2o_pos[1] - new_h2o_pos[0]) / oh_bond_dist[0] / oh_bond_dist[1])
        assert(np.allclose(hoh_bond_angle, orig_HOH_bond_angle))

        en_after, forces_after = self._evaluate_potential()
        exp_argument = -(en_after - self.current_potential_en) / self.temperature / kb # numerically more stable to calculate exp(a+b) than exp(a) exp(b)

        if sampling=="MALA":
            new_com_force = np.sum(forces_after[(3 * index):(3 * index + 3)], axis=0)
            new_trans_vector = new_h2o_pos - orig_h2o_pos + self.trans_step**2 / 2 / kb / self.temperature * new_com_force
            new_trans_vector = new_trans_vector[0]
            # new_prob = np.exp(-np.linalg.norm(trans_vector)**2 / 2 / self.trans_step**2) # Prob. of sampling new_h2o_pos given orig_h2o_pos
            # old_prob = np.exp(-np.linalg.norm(new_trans_vector)**2 / 2 / self.trans_step**2)  # Prob. of sampling orig_h2o_pos given new_h2o_pos
            # proposal_ratio = old_prob / new_prob
            exp_argument += (-np.linalg.norm(new_trans_vector)**2 + np.linalg.norm(trans_vector)**2) / 2 / self.trans_step**2

        acc_ratio = np.exp(exp_argument)
        if self.rng.random(1) < acc_ratio: # move is accepted
            self.current_potential_en = en_after
            self._H2O_forces = forces_after
            return True
        else: # move is rejected
            for atom_idx in range(3):
                self._atoms[self.n_MOF_atoms + 3 * index + atom_idx].position = orig_h2o_pos[atom_idx]
            return False
    
    def vibrate_h2o(self, index : int=None, sampling : str="MALA") -> bool:
        """
        Changes the internal geometry of an h2o molecule in the cell by a random amount.
        Arguments:
            index: If None, an h2o molecule is chosen at random, otherwise the molecule at the specified index is chosen
            sampling: Monte Carlo sampling scheme to use, either MALA (Metropolis-adjusted Langevin) or Metropolis
        Returns:
            True if the change was accepted according to Metropolis-Hastings criterion, False otherwise
        """

        if not(sampling == "MALA" or sampling == "Metropolis"):
            raise ValueError("MOFWithAds.vibrate_h2o sampling argument must be either 'MALA' or 'Metropolis'")
        
        if index is None:
            index = self.rng.integers(self.nh2o)
        
        h2o = self._atoms[(self.n_MOF_atoms + 3 * index):(self.n_MOF_atoms + 3 * index + 3)]
        orig_h2o_pos = h2o.get_positions()
        displ_vec = self.rng.standard_normal([3, 3]) * self.vib_step
        new_h2o_pos = orig_h2o_pos + displ_vec

        if sampling=="MALA":
            orig_com_force = np.sum(self._H2O_forces[(3 * index):(3 * index + 3)], axis=0) # Force on center of mass
            internal_force = self._H2O_forces[(3 * index):(3 * index + 3)] - orig_com_force
            new_h2o_pos += self.vib_step**2 / 2 / kb / self.temperature * internal_force
        
        for atom_idx in range(3):
            self._atoms[self.n_MOF_atoms + 3 * index + atom_idx].position = new_h2o_pos[atom_idx]

        en_after, forces_after = self._evaluate_potential()
        exp_argument = -(en_after - self.current_potential_en) / self.temperature / kb # numerically more stable to calculate exp(a+b) than exp(a) exp(b)

        if sampling=="MALA":
            new_com_force = np.sum(forces_after[(3 * index):(3 * index + 3)], axis=0)
            new_internal_force = forces_after[(3 * index):(3 * index + 3)] - new_com_force
            new_displ_vec = new_h2o_pos - orig_h2o_pos + self.vib_step**2 / 2 / kb / self.temperature * new_internal_force
            # new_prob = np.exp(-np.linalg.norm(displ_vec)**2 / 2 / self.vib_step**2) # Prob. of sampling new_h2o_pos given orig_h2o_pos
            # old_prob = np.exp(-np.linalg.norm(new_displ_vec)**2 / 2 / self.vib_step**2) # Prob. of sampling orig_h2o_pos given new_h2o_pos
            # proposal_ratio = old_prob / new_prob
            exp_argument += (-np.linalg.norm(new_displ_vec)**2 + np.linalg.norm(displ_vec)**2) / 2 / self.vib_step**2

        acc_ratio = np.exp(exp_argument)
        if self.rng.random(1) < acc_ratio: # move is accepted
            self.current_potential_en = en_after
            self._H2O_forces = forces_after
            return True
        else: # move is rejected
            for atom_idx in range(3):
                self._atoms[self.n_MOF_atoms + 3 * index + atom_idx].position = orig_h2o_pos[atom_idx]
            return False
        
    def write_to_traj(self, with_MOF: bool=False):
        """
        Saves a snapshot of the current H2O positions to the trajectory file
        Arguments:
            with_MOF: If true, the MOF atoms will be included in the output file
        """
        # xyz gives more sig figs than PDB
        if with_MOF:
            ase.io.write(self._traj_file, self._atoms, format='xyz')
        else:
            ase.io.write(self._traj_file, self._atoms[self.n_MOF_atoms:], format='xyz')
        self._traj_file.flush()

    def save_rng_state(self, fname : str):
        """
        Saves the current state of the random number generator to a file
        """
        with open(fname, 'wb') as f:
            pickle.dump(self.rng.bit_generator.state, f, pickle.HIGHEST_PROTOCOL)
    
    def seed_rng_state(self, state : Union[str, int]):
        """
        If state is a Path, loads the state of the random number generator from that path.
        If state is an int, seeds the random number generator with that int
        """
        if isinstance(state, str):
            with open(fname, 'rb') as f:
                self.rng.bit_generator.state = pickle.load(f)
        elif isinstance(state, int):
            self.rng = np.random.default_rng(state)
        else:
            raise ValueError('Incorrect argument type')


def _calc_rot_angle(pos1, pos2) -> float:
    """
    Given coordinates of 2 adsorbates with the same center of mass, calculate the absolute value of the angle of the rotation that relates them
    """
    molecule1 = Atoms('OHH', positions=pos1)
    molecule2 = Atoms('OHH', positions=pos2)

    com1 = molecule1.get_center_of_mass()
    com2 = molecule2.get_center_of_mass()
    assert(np.allclose(com1, com2))

    rel1 = pos1 - com1
    rel2 = pos2 - com2

    ghost1 = np.cross(rel1[1], rel1[2])
    ghost1 /= np.linalg.norm(ghost1)
    ghost2 = np.cross(rel2[1], rel2[2])
    ghost2 /= np.linalg.norm(ghost2)
    rel1[0] = ghost1
    rel2[0] = ghost2

    rot = np.linalg.solve(rel1, rel2).T
    angle = np.arccos((np.trace(rot) - 1) / 2)
    return angle

def _calc_rot_matrix(angle: float, axis):
    """
    Calculate a rotation matrix given the angle and axis of rotation
    see https://stackoverflow.com/questions/6721544/circular-rotation-around-an-arbitrary-axis
    """

    axis_normalized = axis / np.linalg.norm(axis)

    q = np.array([np.cos(angle / 2)] + 3 * [np.sin(angle / 2)])
    q[1:] *= axis_normalized
    return np.array([
        [q[0]**2 + q[1]**2 - q[2]**2 - q[3]**2, 2 * (q[1] * q[2] - q[0] * q[3]), 2 * (q[1] * q[3] + q[0] * q[2])],
        [2 * (q[2] * q[1] + q[0] * q[3]), q[0]**2 - q[1]**2 + q[2]**2 - q[3]**2, 2 * (q[2] * q[3] - q[0] * q[1])],
        [2 * (q[3] * q[1] - q[0] * q[2]), 2 * (q[3] * q[2] + q[0] * q[1]), q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2]
    ])