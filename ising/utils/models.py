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

def gen_mixedfieldising(L, J, hz, hx, bc='obc', **params):
    """Generates a Hamiltonian for the mixed field ising model.
    A non-zero hz leads to to non-integrable dynamics.
    """
    s0, sx, sy, sz = ou.gen_s0sxsysz(L)
    H = J*ou.gen_interaction_kdist(sz, k=1, bc=bc)\
        + hx*ou.gen_op_total(sx) + hz*ou.gen_op_total(sz)

    return H


# ------------------------------------
# XXZ model
# ------------------------------------

def gen_xxz_nn(L, a, b, bc='obc', **params):
    """Generates a Hamiltonian for the xxz model with nearest neighbor
    interactions."""
    s0, sx, sy, sz = ou.gen_s0sxsysz(L)
    H = a*ou.gen_interaction_kdist(sx, k=1, bc=bc)\
        + a*ou.gen_interaction_kdist(sy, k=1, bc=bc)\
        + b*ou.gen_interaction_kdist(sz, k=1, bc=bc)
    return H


def gen_xxz_nnn(L, j2, bc='obc', **params):
    """Generates a next-to-nearest neighbor interaction for the xxz model.
    This additional term in the Hamiltonian leads to non-integrable dynamics.
    """
    s0, sx, sy, sz = ou.gen_s0sxsysz(L)
    H1 = j2*ou.gen_interaction_kdist(sz, k=2, bc=bc)
    return H1


# ------------------------------------
# ZXXXXZZ model
# ------------------------------------

def gen_zxxxxzz(L):
    """Generates a Hamiltonian for the xxz model with nearest neighbor
    interactions."""
    S = 1/2
    h = .5
    hxxxx = .3
    hzz = .2

    s0, sx, sy, sz = ou.gen_s0sxsysz(L=1, S=S)
    s0 = s0[0]
    sx = sx[0]
    sy = sy[0]
    sz = sz[0]

    Hxx = ou.gt(ou.gen_tnsk(ou.sk(sx, sx), L, S, 2, 0, 'pbc'))
    Hz = h*ou.gt(ou.gen_tnsk(sz, L, S, 1, 0, 'pbc'))

    # Non-integrable pieces
    Hxxxx = hxxxx*ou.gt(ou.gen_tnsk(ou.sk(ou.sk(ou.sk(sx, sx), sx), sx), L, S,
                        4, 0, 'pbc'))
    Hzz = hzz*ou.gt(ou.gen_tnsk(ou.sk(sz, sz), L, S, 2, 0, 'pbc'))

    H = Hxx + Hz + Hxxxx + Hzz

    # DEBUG:
    # Get Ising even states
    inds = cu.s_val_inds(L, 1)
    # np.where(ou.sz_string(L, diag=True) == 1)[0]

    # momentum 0, even under inversion
    P = cu.k_inv_proj(inds, L, S=1/2, k=0, inv_val=1)
    P = cu.k_inv_proj(inds, L, S=1/2, k=0, inv_val=None)

    # Checking that it is a symmetry of the Hamiltonian
    squareproj = np.conj(P.T) @ P
    t = np.max(np.abs((squareproj @ H - H @ squareproj)))

    print(t)
    raise AssertionError

    return H


# ------------------------------------
# Default Models
# ------------------------------------

def gen_model(model, model_params, integrable=False):
    assert model in valid_models, "Invalid model."
    assert 'L' in model_params.keys(), "Need length to generate model."

    if model == 'MFIM':
        if model_params.keys() == {'L'}:
            model_params = {'L': model_params['L'],
                            'S': 1/2,
                            'J': 1,
                            'hx': (np.sqrt(5)+1)/4.,
                            'hz': 0 if integrable else (np.sqrt(5)+5)/8.}
        H = gen_mixedfieldising(**model_params)

    if model == 'XXZ':
        if model_params.keys() == {'L'}:
            model_params = {'L': model_params['L'],
                            'S': 1/2,
                            'a': 1,
                            'b': 1.05,
                            'j2': 0 if integrable else .3}
        H = gen_xxz_nn(**model_params) + gen_xxz_nnn(**model_params)

    if model == 'ZXXXXZZ':
        H = gen_zxxxxzz(model_params['L'])
        model_params = {'L': model_params['L']}

    return H, model_params


# ===================================
# Saving and Loading Models
# ===================================
# Code to save and load Hamiltonians and eigensystems on spin chains.

def save_projectors(L, S=1/2):
    # label = 'fti' if L % 2 == 0 else 'ft'
    cu.get_symm_proj(L, S, save_projfile=projfile(L, S))
    return


def load_projectors(L, S=1/2):
    return load_sparse_csr(projfile(L, S))


def save_model(model, model_params):
    """Saving information associated with the exact diagonalization via
    symmetry for the model model with the given model_params.
    """
    H, model_params = gen_model(model, model_params)
    L = model_params['L']

    assert os.path.isfile(projfile(L, S=1/2)),\
        "Could not find projection operators. Try running "\
        + "```save_projectors(L)```"

    # Diagonalize with symmetries, save results
    cu.eigh_symms(H, L, S=1/2,
                  save_systemfile=sysfile(model, **model_params),
                  save_eigenfile=eigenfile(model, **model_params),
                  # Optionally, load saved projectors:
                  load_projfile=projfile(**model_params),
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
