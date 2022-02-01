# Basic imports
import numpy as np


def entropy_formula(L, ell, c, cprime):
    """A formula for the entanglement entropy of the ground state of
    a 1D CFT.
    Useful for understanding models near criticality.

    See, for example,
    https://arxiv.org/pdf/0905.4013.pdf
    
    Parameters
    ----------
    L : int
        An integer describing the length of the spin chain.
    ell : int
        An integer describing the size of the subsystem of interest,
        over whose complement we trace to find a reduced density matrix
        and thus an entanglement entropy.
    c : float
        Central charge of the CFT.
    cprime : float
        A constant which emerges in the replica calculation for
        entantlement entropy. I don't have any physical intuition
        for this one.

    Returns
    -------
    float
        The entanglement entropy of the 

    """
    return (c/3) * np.log((L/np.pi) * np.sin(np.pi*ell/L))
