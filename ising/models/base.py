# Basic imports
import numpy as np
from scipy.sparse.linalg import eigsh
from abc import ABC, abstractmethod

# Local imports
import ising.utils.calculation_utils as cu
import ising.utils.ising_plot_utils as ipu
from ising.utils.file_utils import projfile, sysfile, eigenfile
from ising.utils.file_utils import save_sparse_csr, load_sparse_csr


#######################################
# Model Class:
#######################################

# Default verbosity
VERBOSE = 5

# Defaults for loading, saving, and overwriting saved model files
LOAD_DEFAULT = False
SAVE_DEFAULT = False
OVERWRITE_DEFAULT = False

# Parameter determining if we find both full and subspace eigensystems
# when using symmetries for exact diagonalizaion
GET_FULL_ESYS = True


class model_1d(ABC):
    """An abstract model class designed to facilitate calculations with 1D
    spin chains and Hamiltonians with sparse matrix representations.
    """

    # ========================
    # Description, basic utils
    # ========================
    def __str__(self, verbose=True):
        # Basic description
        desc_str = self.name + ": A 1D spin {S} chain of size {L}.\n"

        # Parameters
        param_str = "Model parameters:\n"
        for key, value in self.params.items():
            param_str += '    ' + str(key) + ' : ' + str(value) + '\n'

        # Symmetries and Hamiltonian
        proj_str = "has_projectors : " + str(self.has_projectors) + '\n'
        if self.has_projectors and self.verbose > 0:
            proj_str += '    ' + str(len(self.projectors.items()))\
                        + ' symmetry sectors.\n'
            if self.verbose > 1:
                # Fix
                proj_str += '    Symmetries: '.format(self.symmetries)
                proj_str += '        ' + str([self.projectors.keys()])
        H_str = "has_H : " + str(self.has_H) + '\n'
        if self.has_H and self.verbose > 1:
            print('H:\n', self.H)

        return desc_str + '\n' + param_str + '\n' + proj_str + '\n' + H_str

    # ========================
    # File utils
    # ========================
    def projector_file(self):
        return projfile(self.L, self.S, **self.symmetries)

    def H_file(self):
        return sysfile(self.name, **self.params)

    def eigen_file(self):
        return eigenfile(self.name, **self.params)

    # ========================
    # Calculation Utils
    # ========================

    # ------------------
    # Projectors
    # ------------------
    def load_projectors(self):
        if self.verbose > 0:
            print("Finding symmetry projectors...")
        # Load projectors
        projectors = load_sparse_csr(self.projector_file())
        self.projectors = projectors
        self.has_projectors = True
        if self.verbose > 0:
            print("Complete!")

    def gen_projectors(self, save=SAVE_DEFAULT):
        if self.verbose > 0:
            print("Generating projectors...")
        savefile = None
        if save:
            savefile = self.projector_file()
        # Make projectors:
        projectors = cu.get_symm_proj(self.L, self.S, **self.symmetries,
                                      save_projfile=savefile)
        self.projectors = projectors
        self.has_projectors = True
        if self.verbose > 0:
            print("Complete!")

    def get_projectors(self, load_file=LOAD_DEFAULT,
                       overwrite=OVERWRITE_DEFAULT):
        if load_file:
            # Attempt to load from file
            try:
                self.load_projectors()
            except FileNotFoundError:
                print("Projectors not found.\n")
                self.gen_projectors(save=SAVE_DEFAULT)
        else:
            # Generate projectors, save when appropriate
            file_exists = False
            self.gen_projectors(save=overwrite or not file_exists)

    # ------------------
    # Hamiltonian
    # ------------------
    def load_H(self):
        if self.verbose > 0:
            print("Finding Hamiltonian...")
        # Load Hamiltonian
        self.H = load_sparse_csr(self.H_file())
        self.has_H = True
        if self.verbose > 0:
            print("Complete!")

    def save_H(self):
        assert self.has_H, "No Hamiltonian to save!"
        if self.verbose > 0:
            print("Saving Hamiltonian...")
        # Save Hamiltonian (full dict doesn't take up too much memory)
        system_dict = {'H': self.H,
                       'H_proj': {sector: P @ self.H @ np.conj(P.T)
                                  for sector, P in
                                  self.projectors.items()}
                       }
        save_sparse_csr(self.H_file(), **system_dict)
        if self.verbose > 0:
            print("Complete!")

    @abstractmethod
    def gen_H(self, save):
        if self.verbose > 0:
            print("Generating Hamiltonian...")
        # Make Hamiltonian:
        H = None
        self.H = H
        if save:
            # Save Hamiltonian:
            self.save_H()
        self.has_H = True
        if self.verbose > 0:
            print("Complete!")

    def get_H(self, load_file=LOAD_DEFAULT, overwrite=OVERWRITE_DEFAULT):
        if load_file:
            # Attempt to load from file
            try:
                self.load_H()
            except FileNotFoundError:
                print("Hamiltonian not found.\n")
                self.gen_H(save=SAVE_DEFAULT)
        else:
            # Generate Hamiltonian, save when appropriate
            file_exists = False
            self.gen_H(save=overwrite or not file_exists)

    # ------------------
    # Diagonalization
    # ------------------

    def load_eigensys(self, k=None, **params):
        if self.verbose > 0:
            print("Finding eigensys...")
        # Load eigensys
        self.eigensys = np.load(self.eigen_file(), allow_pickle=True)
        self.loaded_eigensys = True
        self.has_eigensys = True
        if self.verbose > 0:
            print("Complete!")

    def save_eigensys(self, k=None, **params):
        assert self.has_eigensys, "No eigensys to save!"
        if self.verbose > 0:
            print("Saving eigensys...")
        # Save eigensys
        np.savez(self.eigen_file(), **self.eigensys)
        if self.verbose > 0:
            print("Complete!")

    def gen_eigensys(self, save=SAVE_DEFAULT,
                     k=None, use_symms=True, **params):
        if self.verbose > 0:
            print("Generating Hamiltonian...")
        savefile = None
        if save:
            savefile = self.eigen_file()
        # Make eigensystem:
        if self.has_projectors and use_symms:
            # If using symmetries
            if k is not None:
                if self.verbose > 5:
                    print("    Using eigsh with symmetries")
                evals, evecs = eigsh(self.H, k=k, **params)
                self.eigensys = {'evals': evals, 'evecs': evecs}
            else:
                if self.verbose > 5:
                    print("    Using eigh with symmetries")
                self.eigensys = cu.eigh_symms(self.H, self.L, self.S,
                                              proj_dict=self.projectors,
                                              save_eigenfile=savefile,
                                              get_all=GET_FULL_ESYS)
        else:
            # If we are not using symmetries
            if k is not None:
                if self.verbose > 5:
                    print("    Using eigh without symmetries")
                evals, evecs = eigsh(self.H, k=k, **params)
                self.eigensys = {'evals': evals, 'evecs': evecs}
            else:
                if self.verbose > 5:
                    print("    Using eigh without symmetries")
                evals, evecs = np.linalg.eigh(self.H.toarray())
                self.eigensys = {'evals': evals, 'evecs': evecs}

        self.generated_eigensys = True
        self.has_eigensys = True
        if self.verbose > 0:
            print("Complete!")

    def get_eigensys(self, load_file=LOAD_DEFAULT,
                     overwrite=OVERWRITE_DEFAULT, **params):
        if load_file:
            # Attempt to load from file
            try:
                self.load_eigensys(**params)
            except FileNotFoundError:
                print("Eigensystem not found.\n")
                self.gen_eigensys(**params)
        else:
            # Generate projectors, save when appropriate
            file_exists = False
            self.gen_eigensys(save=overwrite or not file_exists,
                              **params)

    # ========================
    # Plotting utils
    # ========================

    def plot_eval_dist(self, sector=None, multiplot=None, **params):
        assert self.has_eigensys, "No eigenvalues to plot!"
        if sector is None:
            evals = self.eigensys['evals']
        elif self.loaded_eigensys:
            evals = self.eigensys['subspace evals'][()][sector]
        elif self.generated_eigensys:
            evals = self.eigensys['subspace evals'][sector]

        if multiplot is None:
            ipu.plot_evaldist(evals, **params)
        else:
            multiplot.plot(ipu.plot_evaldist(evals, **params))

    def plot_levelspace_dist(self, sector=None, multiplot=None, **params):
        assert self.has_eigensys, "No eigenvalues to plot!"
        if sector is None:
            evals = self.eigensys['evals']
        elif self.loaded_eigensys:
            evals = self.eigensys['subspace evals'][()][sector]
        elif self.generated_eigensys:
            evals = self.eigensys['subspace evals'][sector]

        if multiplot is None:
            ipu.plot_lvlspace(evals, **params)
        else:
            multiplot.plot(ipu.plot_lvlspace(evals, **params))

    def plot_eev_density(self, Op, multiplot=None, **params):
        assert self.has_eigensys, "No eigenvalues to plot!"
        if multiplot is None:
            ipu.plot_plot_eev_density(L=self.L, Op=Op,
                                      evals=self.evals, evecs=self.evecs,
                                      **params)
        else:
            multiplot.plot(ipu.plot_lvlspace(evals=self.evals, **params))

    def plot_entanglement(self, eval, multiplot=None, **params):
        assert self.has_eigensys, "No states for which to plot entropies!"
        state_index = np.where(self.evals == eval)[0]
        assert len(state_index) == 1, "Degeneracy/ambiguity in desired state!"
        state = self.evecs[state_index]

        if multiplot is None:
            ipu.plot_state_entropies(L=self.L, state=state, **params)
        else:
            multiplot.plot(ipu.plot_state_entropies(L=self.L, state=state,
                                                    **params))

    # ========================
    # Initialize
    # ========================

    def __init__(self, name, symmetries, fast_start=True,
                 **params):
        # Initializing
        self.name = name
        assert 'L' in params.keys(), "Missing a length for the spin chain."
        self.L = params['L']
        assert 'S' in params.keys(), "Missing a spin for the spin chain."
        self.S = params['S']
        self.params = params
        self.symmetries = symmetries

        # Debugging parameters
        self.verbose = VERBOSE

        # Setting up model
        self.projectors = None
        self.has_projectors = False
        self.H = None
        self.has_H = False
        self.eigensys = None
        self.has_eigensys = False
        # Some syntax differs for numpy.loaded objects and generated dicts,
        # so we keep track of how we obtained the eigensystem.
        self.loaded_eigensys = False
        self.generated_eigensys = False

        # Performing computations
        if fast_start:
            self.get_projectors()
            self.get_H()
            self.get_eigensys()
