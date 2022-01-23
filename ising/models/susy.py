# Basic imports
import numpy as np
import scipy.sparse as sparse

# Local imports
import ising.utils.operator_utils as ou


# ===================================
# Lattice Supersymmetry Utilities
# ===================================

# Supercharges

def Qdag(L, model='M', **args):
    """Supercharge Q^† for a 1D supersymmetric lattice system."""
    if model == 'M':
        return Qdag_M(L, **args)
    else:
        print("Invalid model.")


def Q(L, model='M', **args):
    """Supercharge Q for a 1D supersymmetric lattice system."""
    return sparse.csr_matrix.conjugate(Qdag(L, model, **args)).T


# Hamiltonians

def H_susy(L, model='M', **args):
    """Hamilttonian for a 1D supersymmetric lattice system."""
    H = ou.acomm(Qdag(L, model, **args), Q(L, model, **args), toarray=False)
    return H


# ===================================
# Supercharge Utilities
# ===================================

def Qdag_M(L, width=1, k=0, bc='pbc', ys=None):
    """Supercharge for the lattice models M_l of
    https://arxiv.org/pdf/cond-mat/0307338.pdf

    Uses projectors of the form
        P_i = 1 - c_i^† c_i
    which projects onto states in which the lattice site i is empty.
    In this code, we imagine the state with spin S_z = +1/2 is
    occupied by a fermion, and that with S_z = -1/2 is empty.

    Our definition here is M_l, and using k to describe the momentum of
    the supercharges, as in ou.tnsk.
    """
    # Defining creation and annihilation operators
    cdag = sparse.diags(([1]), (1), format='csr')
    c = sparse.diags(([1]), (-1), format='csr')

    # Canonical commutation relations
    assert(np.array_equal(ou.acomm(cdag, c), np.eye(2)))
    assert(np.array_equal(ou.acomm(c, c), np.zeros((2, 2))))
    assert(np.array_equal(ou.acomm(cdag, cdag), np.zeros((2, 2))))

    # Defining projector onto "empty" lattice sites
    P = sparse.diags(([0, 1]), (0), format='csr')

    if width >= 1:
        # Defining a fat fermion creation operator d^†
        ddag = ou.sk(P, ou.sk(cdag, P))
        # Defining associated supercharge by summing
        Qdag = ou.gt(ou.gen_tnsk(ddag, L, S=1/2, n=3, k=k))

    if width >= 2:
        print("Invalid width.")
        pass
        # Qdag_list = [Qdag]
        # return
    # if width >= 2:
    #     pass
    #     Qdag = ou.gen_op_total(Qdag_list * ys)

    # Supersymmetry commutation relations
    assert(np.array_equal(Qdag@Qdag.toarray(), np.zeros((2**L, 2**L))))

    return Qdag


def Q_M(L, **args):
    return sparse.csr_matrix.conjugate(Qdag_M(L, **args))
