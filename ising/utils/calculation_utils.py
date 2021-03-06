# Basic Imports
import numpy as np

# Linear Algebra Imports
import scipy.sparse as sparse

# Import for curve fitting
from scipy.optimize import curve_fit

# Local imports
import ising.utils.state_utils as su
import ising.utils.operator_utils as ou
from ising.utils.file_utils import save_sparse_csr, load_sparse_csr

########################################
# 1D Spin Chain Calculation Utilities
########################################

DEBUG = False


# ================================
# Time evolution
# ================================

def time_evolve(state, evals, evecs, Ts):
    """Time evolves a state in a system with eigenvalues evals and eigenvectors
    evecs for times in the list Ts.

    Parameters
    ----------
    state : np.array
        Computational basis representation of the input state.
    evals : list
        Eigenvalues of the Hamiltonian used for time evolution.
    evecs : list
        Corresponding eigenstates of the Hamiltonian used for time evolution.
    Ts : list
        List of times for which the state (state) is time evolved.

    Returns
    -------
    list
        List of time evolved states, evolved for times in Ts.
    """
    states = [state]
    for t in Ts:
        eigenbasis_st = np.conj(evecs.T) @ state
        st_t = evecs @ (np.exp(-1j*t*evals)*(eigenbasis_st))
        states.append(st_t)
    return states


def op_ev_state(state, evals, evecs, Op, Ts=[0]):
    """Operator expectation values of the operator (Op) in the state (state)
    at times in Ts.

    Parameters
    ----------
    state : np.array
        Computational basis representation of the input state.
    evals : list
        Eigenvalues of the Hamiltonian used for time evolution.
    evecs : list
        Corresponding eigenstates of the Hamiltonian used for time evolution.
    Op : matrix
        Operator in whose eigenvalues we are interested.
    Ts : list
        List of times for which the state (state) is time evolved.

    Returns
    -------
    list
        List of time evolved operator expectation values, evolved for times in
        Ts.
    """
    states = time_evolve(state, evals, evecs, Ts)
    return np.sum(np.conj(states)*(Op.dot(states)), axis=0)


# ================================
# Eigenstate Thermalization (ETH)
# ================================

def op_ev(Op, vecs):
    """Returns the expectation values of the operator Op in the states vecs."""
    return np.sum(np.conj(vecs)*(Op.dot(vecs)), axis=0)


def op_eev_fluct(L, Op, evals, evecs, deltaE,
                 spectrum_center=1/2, spectrum_width=20):
    """
    Uses the eigenvalues evals and eigenvectors evecs of a Hamiltonian on a
    spin chain of length L. Finds the microcanonical expectation value of an
    operator Op within an energy window deltaE.

    Returns the operator eigenstate expectation values (eevs), microcanonical
    expectation values (mcevs), and fluctuations of eevs around the mcevs,
    all within an energy window of (spectrum_width) percent around
    (spectrum_center).

    As suggested in 1308.2862, as a default we consider only the middle 20%
    of the spectrum.
    """
    # Finding eigenstate expectation values
    op_eev = op_ev(Op, evecs)

    # Defining the range of energies used to find fluctuations
    index_min = int((2**L)//(1/spectrum_center) - (2**L)//(spectrum_width/2))
    index_max = int((2**L)//(1/spectrum_center) + (2**L)//(spectrum_width/2))

    # Finding microcanonical operator expectation values
    op_ev_mc = []

    # Finding microcanonical expectation values for a set of energies
    for n in range(index_min, index_max+1):
        index_left = np.argmax(evals >= evals[n] - deltaE)
        index_right = np.argmax(evals > evals[n] + deltaE)
        op_ev_mc.append(np.mean(op_eev[index_left:index_right]))

    # Finding fluctuations of the operator eigenstate expectation values around
    # the microcanonical expectation value
    sigmaOp = (op_eev[index_min:index_max+1] - op_ev_mc)**2

    return op_eev[index_min:index_max+1], op_ev_mc, sigmaOp


# ================================
# Symmetry Projectors
# ================================

def s_val_inds(L, s_val):
    """Assumes a lattice of length L of spin 1/2 particles.
    Returns indices associated with computational basis (z basis) states whose
    eigenvalue under a string of sigma_z operators is s_val.

    If s_val is None, returns indices associated with all z basis states.
    """
    if s_val in [1, -1]:
        return np.where(ou.sz_string(L, diag=True) == s_val)[0]
    elif s_val is None:
        return np.arange(2**L)
    else:
        raise AssertionError("Invalid eigenvalue s_val for sigma_z string.")


def k_proj(ind, L, S, k=0):
    """Using only states tabulated by ind, produce a projector onto the states
    of momentum k. Previously makek.

    More precisely, takes in a list of ints, encoding pure states on a spin S
    chain of length L, and returns a matrix which projects onto the associated
    momentum eigenstates with momentum k.

    Note:
    Only returns states which can have momentum k and retain their periodicity:
    Consider a state with periodicity r and perform the usual procedure to find
    momentum eigenstates. Add equivalent states translated by n, each with
    additional phases e^{2pi*n*i*k/L}. We would like the periodicity of the
    resulting state to also be r, requiring that r*k/L is an integer.
    Enforced with a tolerance parameter tol.

    Parameters
    ----------
    ind : list
        List of ints which encode states
    L : int
        An integer describing the length of the spin chain.
    S : float
        An integer or half-integer defining the spin of the operators.
    k : integer
        Momentum of the translation eigenstate.

    Returns
    -------
    scipy.sparse.csr_matrix
        A projector onto the eigenstates of translation with momentum k
        associated with the pure states represented by the elements of ind.

    """
    tol = .1/L
    rep_dim = int(np.round(2*S+1, 1))

    tval = []
    tjind = []
    tiind = []
    counter = 0

    for n in ind:
        # Looping over relevant states
        # Beginning with transformation into binary string
        b = np.base_repr(n, base=rep_dim).zfill(L)

        # List whose members are all states in a translation class
        trans_ind = [n]

        for d in range(1, L+1):
            # Getting representatives of translated states, translating by d
            this_ind = int(su.btranslate_bitstring(b, d), rep_dim)
            trans_ind.append(this_ind)
            if this_ind < n:
                # Ensuring we don't get repeat translation classes.
                # In particular, n has to be smaller than all of its
                # translations, and there is a unique such representative of
                # any translation class.
                break
            elif this_ind == n:
                phase = 2 * np.pi * k / L
                # Phase associated single site translation for momentum k
                # Making sure phase * d is a multiple of 2pi (see tval below):
                if np.abs(np.round(k*d/L) - k*d/L) < tol:
                    # Skip last element of list, to not repeat original state:
                    tjind.append(trans_ind[:-1])

                    # Construction of sparse matrix requires
                    # i and j indices of matrix and matrix elements.
                    # i value is row, and this appears in tiind:
                    tiind.append([counter]*(d))
                    # j value is column and is in tjind:
                    counter = counter + 1
                    # Value of projection matrix at given i and j:
                    tval.append((np.exp(1j * phase) ** np.arange(d))
                                / np.sqrt(d))
                    # The form of these elements is why we require that
                    # phase * d is a multiple of 2pi.
                break

    # If there are no states to project onto:
    if counter == 0:
        return(None)

    # Return the projection matrix onto relevant states of momentum k
    return sparse.csr_matrix((np.concatenate(tval),
                             (np.concatenate(tiind), np.concatenate(tjind))),
                             shape=(counter, rep_dim**L))


def k_inv_proj(ind, L, S, k=0, inv_val=1):
    """Using only states tabulated by ind, produce a projector onto the states
    of momentum k and inversion eval inv. Previously makekinv.

    More precisely, takes in a list of ints, encoding pure states on a spin S
    chain of length L, and returns a matrix which projects onto the associated
    momentum eigenstates with momentum k and inversion eigenvalue inv_val.

    Requires L to be even, and k = 0 or L/2. This is because only k = 0 and
    k = L/2 have additional inversion symmetry that commutes with translation
    symmetry.
    To see this, calculate the commutator between inversion and translation
    in different k sectors:
        <k'| T * I - I * T |k>.
    This is zero only if the translation eigenvalue is real, so that k = 0 or
    k = pi.

    Parameters
    ----------
    ind : list
        List of ints which encode states
    L : int
        An integer describing the length of the spin chain.
    S : float
        An integer or half-integer defining the spin of the operators.
    k : integer
        Momentum of the translation eigenstate.
    inv_val : integer
        Inversion eigenvalue of

    Returns
    -------
    scipy.sparse.csr_matrix
        A projector onto the eigenstates of translation with momentum k
        associated with the pure states represented by the elements of ind.
    """
    if inv_val is None:
        return k_proj(ind, L, S, k=k)

    assert k == 0 or k == L/2, \
        "Inversion symmetry only commutes with translation symmetry "\
        + "in the k=0 and k=pi sectors."

    tol = .1/L
    rep_dim = int(np.round(2*S+1, 1))
    tval = []
    tjind = []
    tiind = []
    counter = 0

    for n in ind:
        # Looping over relevant states.
        # Beginning with transformation into rep_dim-ary string
        # (e.g. binary string for spin 1/2 chain).
        b = np.base_repr(n, base=rep_dim).zfill(L)

        # List whose members are all states in a translation class.
        trans_ind = [n]

        for d in range(1, L+1):
            # Getting representatives of translated states, translating by d
            this_ind = int(su.btranslate_bitstring(b, d), rep_dim)
            trans_ind.append(this_ind)
            if this_ind < n:
                # Ensuring we don't get repeat translation classes.
                # In particular, n has to be smaller than all of its
                # translations, and there is a unique such representative of
                # any translation class.
                break
            elif this_ind == n:
                phase = 2 * np.pi * k / L
                # Phase associated single site translation for momentum k
                # Making sure phase * d is a multiple of 2pi (see tval below):
                if np.abs(np.round(k*d/L) - k*d/L) < tol:
                    # Setting up list for translation class states
                    # on inverted lattice:
                    inv_ind = []
                    for j in trans_ind[:-1]:
                        # Inverting states and adding them to inv_ind
                        inv_string = np.base_repr(j, rep_dim).zfill(L)[::-1]
                        inv_ind.append(int(inv_string, rep_dim))
                    # If the translation class is not invariant under
                    # inversion, create appropriate translation + inversion
                    # eigenstates.
                    if not set(inv_ind).issubset(trans_ind[:-1]):
                        # Since there will be another translation class with
                        # the same end projector associated with inv_ind,
                        # we arbitrarily pick max(ind) > max(inv_ind) here.
                        if max(trans_ind) > max(inv_ind):
                            if inv_val == 1:
                                tjind.append(trans_ind[:-1])
                                tjind.append(inv_ind)
                                tval.append(
                                    np.exp(1j * phase)
                                    ** np.concatenate([np.arange(d),
                                                       np.arange(d)])
                                    / np.sqrt(2*d))
                                tiind.append([counter]*(2*d))
                                counter = counter + 1
                            if inv_val == -1:
                                tjind.append(trans_ind[:-1])
                                tjind.append(inv_ind)
                                tval.append(
                                    np.concatenate([
                                        # Original states
                                        np.exp(1j * phase) ** np.arange(d),
                                        # Inverted states with extra -1
                                        -np.exp(1j * phase) ** np.arange(d)])
                                    / np.sqrt(2*d))
                                tiind.append([counter]*(2*d))
                                counter = counter + 1
                    # Otherwise, the translation class is invariant (t.c.i.):
                    else:
                        if k == 0:
                            # If k = 0 and t.c.i., then inv has to be 1:
                            if inv_val == 1:
                                tjind.append(trans_ind[:-1])
                                tval.append(
                                        (np.exp(1j * phase) ** np.arange(d))
                                        / np.sqrt(d))
                                tiind.append([counter]*(d))
                                counter = counter + 1
                        if k == L/2:
                            # Finding the site in the translation class which
                            # is the same as the first site in its inverse:
                            first_inv_ind = np.where(
                                np.array(trans_ind[:-1]) == inv_ind[0])[0]

                            if inv_val == 1:
                                if (first_inv_ind[0] % 2) == 0:
                                    # If the first inverse is on an even site,
                                    # the associated phase at this inverted
                                    # site when k=L/2 is
                                    # exp(2*pi*site/2) = 1 = inversion eval.
                                    tjind.append(trans_ind[:-1])
                                    tval.append(
                                        (np.exp(1j * phase) ** np.arange(d))
                                        / np.sqrt(d))
                                    tiind.append([counter]*(d))
                                    counter = counter + 1
                            if inv_val == -1:
                                if (first_inv_ind[0] % 2) == 1:
                                    # If the first inverse is on an odd site,
                                    # the associated phase at this inverted
                                    # site when k=L/2 is
                                    # exp(2*pi*site/2) = -1 = inversion eval.
                                    tjind.append(trans_ind[:-1])
                                    tval.append(
                                        (np.exp(1j * phase) ** np.arange(d))
                                        / np.sqrt(d))
                                    tiind.append([counter]*(d))
                                    counter = counter + 1
                break

    # If there are no states to project onto:
    if counter == 0:
        return(None)

    # Return the projection matrix onto the correct momentum and inversion eval
    return sparse.csr_matrix((np.concatenate(tval),
                             (np.concatenate(tiind), np.concatenate(tjind))),
                             shape=(counter, rep_dim**L))


def get_symm_proj(L, S,
                  spin_flip=True,
                  translation=True,
                  inversion=False,
                  u1=False,
                  save_projfile=None):
    """Gets projectors associated with translation, inversion (when applicable),
    and spin flip symmetries on a lattice spin model with L sites and spin S.
    Saves the projectors to projfile if it is specified.

    Parameters
    ----------
    L : int
        An integer describing the length of the spin chain.
    S : float
        An integer or half-integer defining the spin of the operators.
    spin_flip : bool
        Bool determining whether or not to store spin flip projectors.
    translation : bool
        Bool determining whether or not to store translation projectors.
    inversion : bool
        Bool determining whether or not to store inversion projectors when
        appropriate.
    u1 : bool
        Bool determining whether or not to store U(1) projectors associated
        with total z spin. Currently not implemented.
    profile : str
        Optional file to which the projectors are stored if specified.

    Returns
    -------
    dict
        A dictionary whose entries are scipy.sparse.csr_matrix symmetry
        projectors, and whose keys are lists containing the eigenvalues of the
        associated symmetries.
    """
    if L % 2 == 1 and inversion is True:
        print("Cannot invert odd-length spin chain.")
        inversion = False

    # Encoding eigenvalues of symmetries
    spin_flip_vals = [1, -1] if spin_flip else [None]
    k_vals = np.arange(L) if translation else [None]
    inv_vals = [1, -1] if inversion else [None]
    # u1_vals = None

    # Preparing for storage of projectors in a dictionary
    proj_dict = {}

    for s_val in spin_flip_vals:
        # Iterating over spin flip eigenvalues
        s_inds = s_val_inds(L, s_val)
        for k_val in k_vals:
            # Iterating over momenta
            if k_val is None:
                raise AssertionError("Have not yet coded s_val projectors.")
            elif L % 2 == 0 and (k_val == 0 or k_val == L/2):
                for inv_val in inv_vals:
                    # Iterating over inversion eigenvalues
                    sector = (s_val, k_val) if inv_val is None\
                        else (s_val, k_val, inv_val)
                    proj = k_inv_proj(s_inds, L, S, k=k_val, inv_val=inv_val)
                    proj_dict[str(sector)] = proj
            else:
                sector = (s_val, k_val)
                proj = k_proj(s_inds, L, S, k=k_val)
                proj_dict[str(sector)] = proj

    if DEBUG:
        # Takes a lot of memory due to dense matrix intermediates.
        # Ensuring that the sum of projectors behaves like the identity
        sum_proj = sum([np.conj(p.T) @ p.toarray()
                        for p in proj_dict.values()])
        delta_sum_proj = sum_proj - np.eye(int(2*S+1)**L)
        try:
            assert np.max(np.abs(delta_sum_proj)) < 1e-10
        except AssertionError:
            print("Projectors for length-"+str(L)+"-chain do not add up "
                  + "to the identity.")
            print("The maximum matrix element of |1 - sum_i P_i| is "
                  + str(np.max(np.abs(delta_sum_proj))))
            # return

    if save_projfile is not None:
        save_sparse_csr(save_projfile, **proj_dict)

    return proj_dict


# ================================
# Subspace Diagonalization
# ================================

def diagonalize_subspaces(H, proj_dict,
                          L, S,
                          sectors='all',
                          verbose=0):
    """Diagonalizes a matrix H in the subspaces of Hilbert space defined by
    projectors. The projectors are entries of a dictionary proj_dict,
    corresponding to a list of keys in 'keys'.

    Designed for a 1D lattice spin system of length L with and site spin S.
    """
    # Preparing to store subspace eigensystems in dictionaries
    eval_dict = {}
    evec_dict = {}
    if sectors == 'all':
        sectors = proj_dict.keys()

    # Diagonalize in the desired subspaces/symmetry sectors
    for sector in sectors:
        # Projector and square projector onto subspace
        P = proj_dict[sector]
        P2 = np.conj(P.T) @ P

        non_comm = np.max(np.abs((P2 @ H - H @ P2)))
        try:
            assert non_comm < 1e-10
        except AssertionError:
            print("For system size " + str(L) + ", projector P onto the "
                  + "symmetry sector " + str(sector)
                  + " does not commute with H.")
            print("The maximum matrix element of |[H, P]| is "+str(non_comm))
            # return

        # Hamiltonian in subspace
        H_proj = P @ H @ np.conj(P.T)
        evals, evecs = np.linalg.eigh(H_proj.toarray())

        # DEBUG:
        # Add methods for including fewer eigenvalues/vectors

        # Storing the eigensystem
        eval_dict[sector] = evals
        evec_dict[sector] = evecs

    return eval_dict, evec_dict


def eigh_symms(H, L, S,
               proj_dict=None,
               load_projfile=None,
               save_systemfile=None,
               save_eigenfile=None,
               get_all=False):
    """Diagonalize the operator H by dividing into symmetry sectors and
    finding the eigensystem of each.
    Saves the subspace Hamiltonians and eigensystems in dictionaries if
    files are specified.

    Designed for a 1D lattice spin system of length L with and site spin S.
    """
    # Finding symmetry projectors and diagonalizing H in symmetry sectors
    if proj_dict is None:
        if load_projfile is None:
            proj_dict = get_symm_proj(L, S)
        else:
            proj_dict = load_sparse_csr(load_projfile)
    sub_evals, sub_evecs = diagonalize_subspaces(H, proj_dict, L, S)

    # Adding additional options for playing with memory requirements
    # for eigensystems
    if get_all:
        # Finding the full eigensystem
        # Concatenating results for all symmetry sectors
        all_evals = []
        all_evecs = []
        for i, sector in enumerate(sub_evals.keys()):
            all_evals = np.concatenate((all_evals, sub_evals[sector]))

            # Putting the eigenvectors into the full Hilbert space
            P_hc = np.conj(proj_dict[sector].T)
            sector_evecs = np.array([P_hc @ evec
                                     for evec in sub_evecs[sector]])
            if i == 0:
                all_evecs = sector_evecs
            else:
                all_evecs = np.concatenate((all_evecs, sector_evecs))
        eigen_dict = {'evals': all_evals, 'evecs': all_evecs,
                      'subspace evals': sub_evals, 'subspace evecs': sub_evecs}
        # Alternatively, could use
        # eigen_dict = {'evals': all_evals,
        #               'evecs': all_evecs,
        #               'subspace evals': sub_evals,
        #               'subspace evecs': sub_evecs
        #               }
        # to additionally store subspace information.
    else:
        # Finding the sub-eigensystems, of smaller dimensionality
        eigen_dict = {'subspace evals': sub_evals,
                      'subspace evecs': sub_evecs}

    # System dictionary doesn't take up too much memory
    system_dict = {'H': H,
                   'H_proj': {sector: P @ H @ np.conj(P.T)
                              for sector, P in
                              proj_dict.items()}
                   }

    # Saving if save files are specified:
    if save_systemfile is not None:
        save_sparse_csr(save_systemfile, **system_dict)
    if save_eigenfile is not None:
        np.savez(save_eigenfile, **eigen_dict)

    return eigen_dict


def esys_from_sub_dicts(proj_dict, eigen_dict, usetype=np.dtype('c8')):
    """A method to retrieve a full eigensystem from orthogonal
    sub-eigensystems.

    Takes in dictionaries encoding the projectors into a set of subspaces
    and the eigensystems of the subspaces, respectively.

    Returns lists containing the eigenvalues and eigenvectors of
    the full system, respectively.

    By default, uses numpy's complex64 dtype to avoid memory errors from
    the outset. If you pass np.dtype('c16') to usetype, this will try to
    use the complex128 dtype, but tries to use complex64 instead if there
    are memory errors.
    """
    # Getting sub-eigensystems
    sub_evals = eigen_dict['subspace evals'][()]
    sub_evecs = eigen_dict['subspace evecs'][()]

    # Concatenating results for all subspaces
    all_evals = []
    all_evecs = []
    for i, sector in enumerate(sub_evals.keys()):
        all_evals = np.concatenate((all_evals, sub_evals[sector]))

        # Putting the eigenvectors into the full Hilbert space
        P_hc = np.conj(proj_dict[sector].T)

        try:
            # Try to get the projected eigenvectors
            sector_evecs = np.array([P_hc@v.astype(usetype)
                                     for v in sub_evecs[sector]])
        except np.core._exceptions.MemoryError as e:
            # If we run into memory errors
            if usetype is np.dtype('c16'):
                # If we are trying to use a larger complex datatype
                sector_evecs = np.array([P_hc@v.astype(np.dtype('c8'))
                                         for v in sub_evecs[sector]])
            else:
                # Otherwise, we've done everything we can already.
                # Raise the error.
                raise e

        if i == 0:
            all_evecs = sector_evecs
        else:
            all_evecs = np.concatenate((all_evecs, sector_evecs))

    return all_evals, all_evecs


def esys_from_sub_files(load_projfile, load_eigenfile):
    """A method to retrieve a full eigensystem from orthogonal
    sub-eigensystems.

    Takes in files storing the projectors into a set of subspaces and
    the eigensystems of the subspaces, respectively.

    Returns lists containing the eigenvalues and eigenvectors of
    the full system, respectively.
    """
    proj_dict = load_sparse_csr(load_projfile)
    eigen_dict = np.load(load_eigenfile, allow_pickle=True)
    return esys_from_sub_dicts(proj_dict, eigen_dict)


# ================================
# Entanglement
# ================================

def entanglement_entropy(state, cut_len, rep_dim=2):
    """Returns the entanglement entropy of a state on a spin chain with local
    dimension dim, across the cut at site cut_len.


    Parameters
    ----------
    state : array
        Array represting the state in the tensor product basis.
    cut_len : int
        Location of the cut across which entanglement entropy is calculated.
    rep_dim : int
        Dimension of the Hilbert space at each lattice site.

    Returns
    -------
    float
        Entanglement entropy across the cut.

    """
    # Get the size of the spin chain
    L = int(np.round(np.log(len(state))/np.log(rep_dim)))

    # Divide the state into cut_len site slices, introducing a matrix C_ij
    Cij = np.reshape(state, (rep_dim**cut_len, rep_dim**(L-cut_len)))

    # Finding the singular values of C_ij (and thus the reduced density matrix)
    S = np.linalg.svd(Cij, full_matrices=0, compute_uv=0)
    S = np.abs(S)
    S = S[S > (10**-15)]

    # Returning the entanglement entropy
    return - np.sum((S**2)*np.log(S**2))

# ---------------------------------------------
# Entanglement Entropy Calculation Details:
# ---------------------------------------------
# The calculation first divides the state into the state on the first
# L - x_cut sites and the final x_cut sites; the format of which is an
# (x_cut) x (L - x_cut) matrix,
#     C = {C_ij}.
#
# The state can be divided into L-cut_len sites and cut_len sites as
# |state> = sum_{k_i = 1}^dim |k_1> ... |k_{L - cut_len}> |psi^{k_1 ...s}>,
# where the final state is on the last cut_len sites.
# then the first row of C_ij is
#     C_0j = |psi^{000...00}>_j,
# The second is
#     C_1j = |psi^{000...01}>_j,
# and so on.
#
# The reduced density matrix for the last x_cut sites is then
# rho = sum_{k_i} |psi^{k}><psi^{k}|,
# and thus has matrix elements
#     rho_ij = sum_{k_i} (|psi^{k}>)_i (|psi^{k}>)_j = (C^T)_ik C_kj,
# since (C^T)_ik = |psi^{k}>_i.
#
# Next, the singular values s of C are the square roots of the eigenvalues
# of C^T C, so the singular values of C give the eigenvalues of rho.
# With these, the calculation of the entanglement entropy is simple:
#     S_ent = - sum_s s^2 log(s^2).


# ---------------------
# CFT Entanglement
# ---------------------

def cft_entropy(L, cut_len, c, cprime):
    """A formula for the entanglement entropy of the ground state of
    a 1D CFT.
    Useful for understanding models near criticality.

    See, for example,
    https://arxiv.org/pdf/0905.4013.pdf

    Parameters
    ----------
    L : int
        An integer describing the length of the spin chain.
    cut_len : int
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
    return (c/3) * np.log((L/np.pi) * np.sin(np.pi*cut_len/L))


def match_cft_entropy(L, cut_lens, entropies):
    """Matches a list of cut lengths and associated entropies
    to the CFT function for entanglement entropy, using
    scipy.optimize.curve_fit.

    Outputs the associated constants (central charge and cprime)
    for the matched CFT.
    """
    def model_cft_entropy(cut_len, c, cprime):
        return cft_entropy(L, cut_len, c, cprime)

    params, _ = curve_fit(model_cft_entropy, cut_lens, entropies)

    c, cprime = params

    return c, cprime
