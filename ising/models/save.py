# Basic imports
import numpy as np
import os.path

# Local imports
import ising.utils.calculation_utils as cu

# File utils
from ising.utils.file_utils import load_sparse_csr
from ising.utils.file_utils import projfile, sysfile, eigenfile
import ising.models.base as base


# ===================================
# Saving and Loading Models
# ===================================
# Code to save and load Hamiltonians and eigensystems on spin chains.

def save_projectors(L, S=1/2, **symmetries):
    # label = 'fti' if L % 2 == 0 else 'ft'
    cu.get_symm_proj(L, S, **symmetries,
                     save_projfile=projfile(L, S, **symmetries))
    return


def load_projectors(L, S, **symmetries):
    return load_sparse_csr(projfile(L, S, **symmetries))


def save_default_model(model, L):
    """Saving information associated with the exact diagonalization via
    symmetry for the model model with the given model_params.
    """
    H, model_params, symmetries = base.gen_model(model, L=L)

    assert os.path.isfile(projfile(L, S=1/2, **symmetries)),\
        "Could not find projection operators. Try running "\
        + "```save_projectors(L)```"

    # Diagonalize with symmetries, save results
    cu.eigh_symms(H, L, S=1/2,
                  save_systemfile=sysfile(model, **model_params),
                  save_eigenfile=eigenfile(model, **model_params),
                  # Optionally, load saved projectors:
                  load_projfile=projfile(**model_params, **symmetries),
                  )

    return


def load_model(model, model_params={}):
    """Loading Hamiltonian and eigensystem information associated with the
    exact diagonalization via symmetry for the model model with the given
    model_params.

    To get the full Hamiltonian and eigenvalues, you may use:
    ```
    # Loading model info
    system_dict, eigen_dict = load_model(model, model_params)

    # Defining Hamiltonian and eigensystem
    H = system_dict['H']
    evals, evecs = eigen_dict['evals'], eigen_dict['evecs']
    ```
    """
    # Fetching information about the system
    system_dict = load_sparse_csr(sysfile(model, **model_params))
    eigen_dict = np.load(eigenfile(model, **model_params), allow_pickle=True)

    return system_dict, eigen_dict
