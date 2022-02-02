# Basic imports
import numpy as np

# Local imports
import ising.utils.operator_utils as ou

# ===================================
# Models
# ===================================
ising_types = ['MFIM', 'XXZ', 'ZXXXXZZ']


# ------------------------------------
# Mixed Field Ising model
# ------------------------------------

def gen_mixedfieldising(L, J, hz, hx, bc='pbc', **params):
    """Generates a Hamiltonian for the mixed field ising model.
    A non-zero hz leads to to non-integrable dynamics.
    """
    # Pauli Matrices
    S = 1/2
    _, sx, _, sz = ou.gen_s0sxsysz(L=1, S=S)
    sx = sx[0]
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

def gen_xxz(L, a, b, bc='pbc', **params):
    """Generates a Hamiltonian for the xxz model with nearest neighbor
    interactions."""
    # Pauli Matrices
    S = 1/2
    _, sx, sy, sz = ou.gen_s0sxsysz(L=1, S=S)
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
    _, sx, _, _ = ou.gen_s0sxsysz(L=1, S=S)
    sx = sx[0]

    # Interactions
    # (Of the form [sx*1*sx])
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

def default_params(name, L, integrable=False):
    """Setting up default parameters for each of the models above."""
    model_params = None
    if name == 'MFIM':
        model_params = {'L': L,
                        'S': 1/2,
                        'J': 1,
                        'hx': (np.sqrt(5)+1)/4.,
                        'hz': (np.sqrt(5)+5)/8. if not integrable else 0}
        symmetries = {'spin_flip': False,
                      'translation': True,
                      'inversion': True,
                      'u1': False}
    if name == 'XXZ':
        model_params = {'L': L,
                        'S': 1/2,
                        'a': 1,
                        'b': 1.05,
                        'j2': .3 if not integrable else 0}
        symmetries = {'spin_flip': True,
                      'translation': True,
                      'inversion': True,
                      'u1': False}
    if name == 'ZXXXXZZ':
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


def gen_model(name, L, model_params=None, integrable=False):
    assert name in ising_types, "Invalid model."

    def_params, symmetries = default_params(name, L=L,
                                            integrable=integrable)

    if model_params is None:
        model_params == def_params

    if name == 'MFIM':
        H = gen_mixedfieldising(**model_params)
    if name == 'XXZ':
        H = gen_xxz(**model_params) + gen_xxz_nnn(**model_params)
    if name == 'ZXXXXZZ':
        H = gen_zxxxxzz(**model_params)

    return H, model_params, symmetries
