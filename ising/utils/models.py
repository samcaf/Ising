# Basic imports
import numpy as np
import os.path

# Local imports
import ising.utils.operator_utils as ou
import ising.utils.calculation_utils as cu

# File utils
from ising.utils.file_utils import load_sparse_csr
from ising.utils.file_utils import valid_models, projfile, sysfile, eigenfile

# ===================================
# Models
# ===================================


# ------------------------------------
# Mixed Field Ising model
# ------------------------------------

def gen_mixedfieldising(L, J, hz, hx, bc='pbc', **params):
    """Generates a Hamiltonian for the mixed field ising model.
    A non-zero hz leads to to non-integrable dynamics.
    """
    # Pauli Matrices
    S = 1/2
    s0, sx, sy, sz = ou.gen_s0sxsysz(L=1, S=S)
    s0 = s0[0]
    sx = sx[0]
    sy = sy[0]
    sz = sz[0]

    # Interactions
    Hxx = J * ou.gt(ou.gen_tnsk(ou.sk(sx, sx), L, S, n=2, bc=bc))
    Hx = hx * ou.gt(ou.gen_tnsk(sx, L, S, n=1, bc=bc))
    Hz = hz * ou.gt(ou.gen_tnsk(sz, L, S, n=1, bc=bc))

    # Full Hamiltonian
    H = Hxx + Hx + Hz
    return H


# ------------------------------------
# XXZ model
# ------------------------------------

def gen_xxz_nn(L, a, b, bc='pbc', **params):
    """Generates a Hamiltonian for the xxz model with nearest neighbor
    interactions."""
    s0, sx, sy, sz = ou.gen_s0sxsysz(L)
    # Pauli Matrices
    S = 1/2
    s0, sx, sy, sz = ou.gen_s0sxsysz(L=1, S=S)
    s0 = s0[0]
    sx = sx[0]
    sy = sy[0]
    sz = sz[0]

    # Interactions
    Hxx = a * ou.gt(ou.gen_tnsk(ou.sk(sx, sx), L, S, n=2, bc=bc))
    Hyy = a * ou.gt(ou.gen_tnsk(ou.sk(sy, sy), L, S, n=2, bc=bc))
    Hzz = b * ou.gt(ou.gen_tnsk(ou.sk(sz, sz), L, S, n=2, bc=bc))

    # Full Hamiltonian
    H = Hxx + Hyy + Hzz
    return H


def gen_xxz_nnn(L, j2, bc='pbc', **params):
    """Generates a next-to-nearest neighbor interaction for the xxz model.
    This additional term in the Hamiltonian leads to non-integrable dynamics.
    """
    # Pauli Matrices
    S = 1/2
    s0, sx, sy, sz = ou.gen_s0sxsysz(L=1, S=S)
    s0 = s0[0]
    sx = sx[0]
    sy = sy[0]
    sz = sz[0]

    # Interactions
    Hxx_nnn = j2 * ou.gt(ou.gen_tnsk(ou.sk(ou.sk(sx, np.eye(int(2*S+1))), sx),
                                     L, S, n=3, bc=bc))
    return Hxx_nnn


# ------------------------------------
# ZXXXXZZ model
# ------------------------------------

def gen_zxxxxzz(L, J, hz, hzz, hxxxx, bc='pbc', **params):
    """Generates a Hamiltonian for a model with nearest neighbor x
    interactions, a z-directional magnetic field, an x-x-x-x interaction,
    and a z-z nearest neighbor interaction.
    """
    S = 1/2
    s0, sx, sy, sz = ou.gen_s0sxsysz(L=1, S=S)
    s0 = s0[0]
    sx = sx[0]
    sy = sy[0]
    sz = sz[0]

    Hxx = J * ou.gt(ou.gen_tnsk(ou.sk(sx, sx), L, S, n=2, bc=bc))
    Hz = hz*ou.gt(ou.gen_tnsk(sz, L, S, n=1, bc=bc))

    # Non-integrable pieces
    Hxxxx = hxxxx * ou.gt(ou.gen_tnsk(ou.sk(ou.sk(ou.sk(sx, sx), sx), sx),
                                      L, S, n=4, bc=bc))
    Hzz = hzz * ou.gt(ou.gen_tnsk(ou.sk(sz, sz), L, S, n=2, bc=bc))

    H = Hxx + Hz + Hxxxx + Hzz

    return H


# ------------------------------------
# Default Models
# ------------------------------------

def default_params(model, L, integrable=False):
    """Setting up default parameters for each of the models above."""
    model_params = None
    if model == 'MFIM':
        model_params = {'L': L,
                        'S': 1/2,
                        'J': 1,
                        'hx': (np.sqrt(5)+1)/4.,
                        'hz': (np.sqrt(5)+5)/8. if not integrable else 0}
        symmetries = {'spin_flip': False,
                      'translation': True,
                      'inversion': True,
                      'u1': False}
    if model == 'XXZ':
        model_params = {'L': L,
                        'S': 1/2,
                        'a': 1,
                        'b': 1.05,
                        'j2': .3 if not integrable else 0}
        symmetries = {'spin_flip': True,
                      'translation': True,
                      'inversion': True,
                      'u1': False}
    if model == 'ZXXXXZZ':
        model_params = {'L': L,
                        'S': 1/2,
                        'J': 1,
                        'hz': .5,
                        'hzz': .3,
                        'hxxxx': .3 if not integrable else 0,
                        }
        symmetries = {'spin_flip': True,
                      'translation': True,
                      'inversion': True,
                      'u1': False}

    return model_params, symmetries


def gen_model(model, L, integrable=False):
    assert model in valid_models, "Invalid model."

    model_params, symmetries = default_params(model, L=L,
                                              integrable=integrable)
    if model == 'MFIM':
        H = gen_mixedfieldising(**model_params)
    if model == 'XXZ':
        H = gen_xxz_nn(**model_params) + gen_xxz_nnn(**model_params)
    if model == 'ZXXXXZZ':
        H = gen_zxxxxzz(**model_params)

    return H, model_params, symmetries


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
    H, model_params, symmetries = gen_model(model, L=L)

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
