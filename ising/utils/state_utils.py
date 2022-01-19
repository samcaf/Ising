# Basic Imports
import numpy as np

########################################
# 1D Spin Chain General Utilities
########################################
# A python toolbox for states on one dimensional quantum spin chains.


# ================================
# Basic Utilities
# ================================


def normalize_vec(state):
    """Normalizes the vector input state.

    Parameters
    ----------
    state : array
        An array  describing the state to be normalized.

    Returns
    -------
    type
        The normalized version of the input state.
    """
    return(state/np.sqrt(np.conj(state)@state))


# ----------------------
# Bitstring Utils
# ----------------------

def btranslate_bitstring(b, d):
    """Translates the bitstring b by d positions."""
    b1 = b[0:d]
    b2 = b[d:]
    return b2 + b1


def bget_cycle_length(b, base):
    """Computes the number of translations on a bitstring representing a spin
    chain of site dimension (base) allowed before bitstring b comes back to
    itself.
    """
    n = int(b, base)
    L = len(b)
    for i in range(1, L):
        if n == int(btranslate_bitstring(b, i), base):
            return i
    return L


# ================================
# State Utilities
# ================================


def bget_T_representatives(L, base):
    """Get one representative from each set of basis states on a spin chain of
    site dimension (base) that are equivalent up to the action of the
    translation operator.

    Parameters
    ----------
    rep : string
        Bit string representation of a state, at which the output state
        will be proportional to 1.
    k : float
        Momentum of the translation eigenstate.
    L : int
        An integer describing the length of the spin chain.
    cycle : int
        Number of translations required to get from the state to itself.
    base : int
        Base of the input/output strings, given by the dimension of the
        Hilbert space at each site.

    Returns
    -------
    list of strings
        A list of strings associated with representatives of equivalence
        classes of basis states which are equivalent up to translation.

    """
    reps = []
    for n in range(base**L):
        flag = False
        b = np.base_repr(n, base).zfill(L)
        for d in range(1, L):
            if btranslate_bitstring(b, d) in reps:
                flag = True
        if not flag:
            reps.append(b)
    return reps


def bget_T_eigenstate(rep, k, L, cycle, base):
    """Getting eigenstates of the translation operator with momentum k
    on a spin chain with site dimension (base).

    Parameters
    ----------
    rep : string
        Bit string representation of a state, at which the output state
        will be proportional to 1.
    k : float
        Momentum of the translation eigenstate.
    L : int
        An integer describing the length of the spin chain.
    cycle : int
        Number of translations required to get from the state to itself.
    base : int
        Base of the input/output strings, given by the dimension of the
        Hilbert space at each site.

    Returns
    -------
    array
        Array associated with the translation eigenstate with momentum (k)
        obtained by translating the orignal state (rep).

    """
    # Starting with a state at a site determined by (rep):
    state = np.zeros(base**L)
    state[int(rep, base)] = 1

    for i in range(1, cycle):
        # Setting up a state which is translated by i relative to (state):
        trstate = np.zeros(base**L)
        trstate[int(btranslate_bitstring(rep, i), base)] = 1

        # Adding a translated piece with an additional phase to the state
        state = state + np.exp(2 * np.pi * 1j * k * i/L) * trstate

    # Normalizing and returning the state
    state = state/np.sqrt(cycle)

    return state


def gen_state_bloch(thetaList, phiList):
    """Generates a set of states on the Bloch sphere with azimuthal angles in
    thetaList and polar angles in phiList
    """
    L = len(thetaList)
    psi = np.kron([np.cos(thetaList[0]/2.),
                   np.exp(1j*phiList[0])*np.sin(thetaList[0]/2.)],
                  [np.cos(thetaList[1]/2.),
                   np.exp(1j*phiList[1])*np.sin(thetaList[1]/2.)])
    for i in range(2, L):
        psi = np.kron(psi, [np.cos(thetaList[i]/2.),
                            np.exp(1j*phiList[i])*np.sin(thetaList[i]/2.)])
    return psi
