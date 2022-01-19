# Basic Imports
import numpy as np
import scipy.sparse as sparse
import os.path

# Imports for central conneccted operators
import itertools

# Imports for debugging
from pprint import pprint

########################################
# 1D Spin Chain Operator Utilities
########################################
# A python toolbox for operators and symmetries on one dimensional quantum
# spin chains.


# ================================
# Operator Utilities
# ================================

# ----------------------
# Basic Operator Utils
# ----------------------


def sk(A, B, format='csr'):
    """A quick command for the sparse tensor product
    (Kronecker product) of the matrices A and B.

    Parameters
    ----------
    A, B : sparse array
        Sparse input arrays.
    format : string
        Format of the result.

    Returns
    -------
    array
        Tensor (Kronecker) product of the sparse input arrays.

    """
    return(sparse.kron(A, B, format))


def gen_op_total(op_list):
    """Sums the operators in op_list.

    Parameters
    ----------
    op_list : array
        Array of operators.

    Returns
    -------
    operator
        Summed operator whose type matches the type of the elements of op_list.

    """
    L = len(op_list)
    tot = op_list[0]
    for i in range(1, L):
        tot = tot + op_list[i]
    return tot


# Shorthand for the above function
gt = gen_op_total


def gen_op_prod(op_list):
    """Takes the matrix product (not the tensor product) of the operators in
    op_list.

    Parameters
    ----------
    op_list : array
        Array of operators.

    Returns
    -------
    operator
        Product operator whose type matches the type of the elements of
        op_list.

    """
    L = len(op_list)
    P = op_list[0]
    for i in range(1, L):
        P = P*op_list[i]
    return P


# ----------------------
# Spin Operators
# ----------------------

def gen_slsr(L, S):
    """Gives the lowering and raising operators of the spin S representation of
    SU(2) on each site in the tensor product basis, on a spin chain of length
    L.

    For the operators at each site only, see, for example,
    Griffiths Quantum Mechanics, 2nd Ed, Problem 4.53:
    https://notendur.hi.is/mbh6/html/_downloads/introqm.pdf#page=207

    Parameters
    ----------
    L : int
        An integer describing the length of the spin chain.
    S : float
        An integer or half-integer defining the spin of the operators.

    Returns
    -------
    2 lists of scipy.sparse.csr_matrix objects
        Lowering/raising operators at each site in the tensor product basis.

    """
    # Number of rows/columns in the spin S representation of SU(2):
    rep_dim = int(round((2*S+1), 1))

    # Entries along the diagonal of S_z
    inc = np.arange(0, 2*S+1, 1)

    # Proportional to off diagonal elements of S_x
    srd = np.sqrt(inc[1:]*np.flip(inc[1:]))

    # Single site raising/lowering operators, with only off-diagonal elements
    sr = sparse.diags((srd), (1), format='csr')
    sl = sparse.diags((srd), (-1), format='csr')

    # Initializing lists for operators at each site
    sr_list = []
    sl_list = []

    # In the following, we take the tensor product
    # (id * id * ... * id * S_{r/l}^{site i} * id * ... * id)
    # to produce the raising/lowering operator at site i in the tensor product
    # basis
    for i_site in range(L):
        if i_site == 0:
            sr_i = sr
            sl_i = sl

        else:
            sr_i = sparse.csr_matrix(np.eye(rep_dim))
            sl_i = sparse.csr_matrix(np.eye(rep_dim))

        for j_site in range(1, L):
            if j_site == i_site:
                sr_i = sparse.kron(sr_i, sr, 'csr')
                sl_i = sparse.kron(sl_i, sl, 'csr')
            else:
                sr_i = sparse.kron(sr_i, np.eye(rep_dim), 'csr')
                sl_i = sparse.kron(sl_i, np.eye(rep_dim), 'csr')
        sr_list.append(sr_i)
        sl_list.append(sl_i)

    return sl_list, sr_list


def gen_s0sxsysz(L, S=1/2):
    """Gives the identity, and spin matrices sx, sy, sz of spin S on each site
    of a chain of length L; useful for computations involving set of spin S
    states on each site.

    Note that this returns the relevant local operators on the Hilbert space
    of the full spin chain, in the tensor product representation.

    For the operators at each site only, see, for example,
    Griffiths Quantum Mechanics, 2nd Ed, Problem 4.53:
    https://notendur.hi.is/mbh6/html/_downloads/introqm.pdf#page=207

    Parameters
    ----------
    L : int
        An integer describing the length of the spin chain.
    S : float
        An integer or half-integer defining the spin of the operators.

    Returns
    -------
    4 lists of scipy.sparse.csr_matrix objects
        Identity, along with S_x S_y and S_z of spin S, on the spin chain.

    """
    # Number of rows/columns in the spin S representation of SU(2):
    rep_dim = int(round((2*S+1), 1))

    # Entries along the diagonal of S_z
    inc = np.arange(0, 2*S+1, 1)
    szd = S - inc

    # Off diagonal elements of S_x
    sxd = 1/2 * np.sqrt(inc[1:]*np.flip(inc[1:]))

    # S_x and S_y matrices of spin S, with only off-diagonal elements
    sx = sparse.diags((sxd, sxd), (1, -1), format='csr')
    sy = sparse.diags((-1J*sxd, 1J*sxd), (1, -1), format='csr')
    sz = sparse.diags(szd, 0, format='csr')

    # Initializing lists for operators at each site
    s0_list = []
    sx_list = []
    sy_list = []
    sz_list = []

    id = sparse.identity(rep_dim**L, format='csr')

    # In the following, we take the tensor product
    # (id * id * ... * id * S_a^{site i} * id * ... * id)
    # to produce the S_a operator at site i in the tensor product basis

    for i_site in range(L):
        if i_site == 0:
            X = sx
            Y = sy
            Z = sz
        else:
            X = sparse.csr_matrix(np.eye(rep_dim))
            Y = sparse.csr_matrix(np.eye(rep_dim))
            Z = sparse.csr_matrix(np.eye(rep_dim))

        for j_site in range(1, L):
            if j_site == i_site:
                X = sparse.kron(X, sx, 'csr')
                Y = sparse.kron(Y, sy, 'csr')
                Z = sparse.kron(Z, sz, 'csr')
            else:
                X = sparse.kron(X, np.eye(rep_dim), 'csr')
                Y = sparse.kron(Y, np.eye(rep_dim), 'csr')
                Z = sparse.kron(Z, np.eye(rep_dim), 'csr')

        sx_list.append(X)
        sy_list.append(Y)
        sz_list.append(Z)
        s0_list.append(id)

    return s0_list, sx_list, sy_list, sz_list


# ----------------------
# Symmetry Operators
# ----------------------


def gen_tr(L, S):
    """Generates a translation operator for a spin S chain of length L.

    Parameters
    ----------
    L : int
        An integer describing the length of the spin chain.
    S : float
        An integer or half-integer defining the spin of the operators.

    Returns
    -------
    scipy.sparse.csr_matrix
        A matrix which implements a translation by a single site in the tensor
        product basis of the spin chain.

    """
    rep_dim = int(np.round((2*S+1), 1))
    full_dim = rep_dim**L

    # A vector used to describe permuations of (full_dim)/full Hilbert space.
    mb = ['0']*(full_dim)

    for i in range(full_dim):
        # Represents i as a base-rep_dim string of length L:
        a = (np.base_repr(i, rep_dim).zfill(L))
        # Note that zfill adds zeros to the front of the string

        # Cylically permuting the string, which will be an index of an array.
        # The numbers perform a cyclic permutation on range(full_dim), in which
        # below the first (rep_dim) numbers of (full_dim) get moved to the last
        # (rep_dim) numbers
        a = a[-1] + a[:-1]

        # Produces an integer from the permuted string of base rep_dim
        mb[i] = int(a, rep_dim)

    # Setting up a sparse translation in the tensor product basis
    # Recall that the first argument of sparse.csr_matrix,
    # (data, (row_ind, col_ind)),
    # satisfies
    # a[row_ind[k], col_ind[k]] = data[k].
    return sparse.csr_matrix(([1]*(full_dim), (mb, range(full_dim))),
                             shape=(full_dim, full_dim))
    # This acts on the last (rep_dim) elements of a vector to the front,
    # and performs a similar cyclic permutation on the remaining elements


def gen_invtr(L, S):
    """Generates a translation operator for a spin S chain of length L.
    The direction of the permutation is opposite that produced by gen_tr(L, S).

    For comments, see gen_tr(L, S).
    """
    rep_dim = int(np.round((2*S+1), 1))
    full_dim = rep_dim**L
    mb = ['0']*(full_dim)
    for i in range(full_dim):
        a = (np.base_repr(i, rep_dim).zfill(L))
        a = a[1:] + a[0]
        mb[i] = int(a, rep_dim)
    return sparse.csr_matrix(([1]*(full_dim), (mb, range(full_dim))),
                             shape=(full_dim, full_dim))


def gen_invop(L, S):
    """Generates an operator which inverts a spin chain the lattice through its
    center, switching the first site to the final site, the second site with
    the penultimate site, etc.

    For further comments, see gen_tr(L, S).
    """
    rep_dim = int(np.round(2*S+1, 1))
    mb = ['0']*(rep_dim**L)
    for i in range(rep_dim**L):
        # Inverting the string associating with a state, thus sending it
        # to the associated state on the inverted lattice.
        mb[i] = int((np.base_repr(i, rep_dim).zfill(L))[::-1], rep_dim)
    return sparse.csr_matrix(([1]*(rep_dim**L), (range(rep_dim**L), mb)),
                             shape=(rep_dim**L, rep_dim**L))


def gen_diagprojector(symvec, symval):
    """Takes in a set of eigenvalues, symvec, associated with a symmetry
    operator, and the eigenvalue, symval, onto which we would like to project.
    Returns a operator in the symmetry eigenbasis which projects onto the given
    symmetry value, symval.

    Parameters
    ----------
    symvec : np.array
        Array of symmetry values/eigenvalues of a symmetry operator.

    symval : float
        The symmetry eigenvalue onto which we would like to project.

    Returns
    -------
    type
        Projector onto the symmetry sector, in the symmetry eigenbasis.

    """
    # Must use np.array for np.where
    symvec = np.array(symvec)

    # Finding which and how many symmetry values match the desired value
    ind = np.where(symvec == float(symval))
    dim = np.size(ind)

    # Initializing a (symvec)x(dim) matrix with all zeros, to project the space
    # of size symvec onto a subset of size (dim)
    P = sparse.lil_matrix((dim, len(symvec)))

    # Preparing the projector onto symval
    for j in range(dim):
        P[j, ind[0][j]] = 1.0

    # Return the projector, in the symmetry eigenbasis
    return P


def sz_string(L, diag=False):
    """Assumes a lattice of length L of spin 1/2 particles.
    Generates a matrix representation for the a string of sigma_zs in
    the computational (z) basis, if diag is False. Otherwise, returns a list
    containing the diagonal elements of the matrix. Previously gen_F_bf.

    In the case where the model has only an xx interaction for its quadratic
    part, this coincides with the operator that implements the Ising
    'spin flip' operation.
    """
    diagonal = np.zeros(2**L)

    for i in range(2**L):
        i_bin = np.base_repr(i, base=2).zfill(L)
        value = (-1)**sum([int(s) for s in i_bin])
        diagonal[i] += value

    sz_string = sparse.diags(diagonal, format='csr')

    if diag:
        return(diagonal)

    return(sz_string)


def spinflip1(L):
    """Generates a matrix operator in the tensor product basis that flips every
    spin on a spin 1 spin chain of length L.

    For further comments, see gen_tr(L, S).
    """
    rep_dim = int(np.round(2*1+1, 1))
    mb = ['0']*(rep_dim**L)

    for i in range(rep_dim**L):
        # Generating a string for the permutation.
        perm_string = np.base_repr(i, rep_dim).zfill(L)
        # Setting 2s to 0s in the string (m=1 --> m=-1);
        # 4 does not appear in the orignal string, and is a temp value
        perm_string = perm_string.replace('0', '4').replace('2', '0')
        # Setting (previous) 0s to 2s in the string (m=-1 --> m=1)
        perm_string = perm_string.replace('4', '2')

        # The index to which i will be permuted through the spin flip
        mb[i] = int(perm_string, rep_dim)

    # Return the spin flip operator
    return sparse.csr_matrix(([1]*(rep_dim**L), (range(rep_dim**L), mb)),
                             shape=(rep_dim**L, rep_dim**L))


# ----------------------
# Interaction Utils
# ----------------------

def gen_interaction_kdist(op_list, op_list_2=[], k=1, bc='obc'):
    """Generates an operator which sums products of operators
    whose indices in a list differ by k.
    For example, if op_list_2 is undefined, one term in the total operator is
    op_list[0]*op_list[k]

    Useful in generating interaction terms in Hamiltonains with
    next^{k-1}-nearest neighbor interactions.

    Parameters
    ----------
    op_list : array
        A list of operators, generally one at each site of a spin chain.
    op_list_2 : array (optional)
        A list of operators, generally one at each site of a spin chain.
        The operators in op_list will be multiplied by operators in
        op_list_2. If not given, op_list_2 is set to op_list.
    k : int
        Difference in the index between operators in op_list and op_list_2
        which are multiplied together.
    bc : string
        Description of the boundary conditions used in generating the overall
        interaction. Can be 'obc' (open boundary conditions) or 'pbc' (periodic
        boundary conditions)

    Returns
    -------
    operator
        Overall next^{k-1}-nearest neighbor interaction Hamiltonian produced
        with op_list and op_list_2.

    """
    L = len(op_list)

    if op_list_2 == []:
        op_list_2 = op_list
    H = sparse.csr_matrix(op_list[0].shape)
    Lmax = L if bc == 'pbc' else L-k
    for i in range(Lmax):
        H = H + op_list[i]*op_list_2[np.mod(i+k, L)]
    return H


def gen_Hbond_Hfield_xint(L, bc='obc'):
    """Generates Hamiltonians for a length L spin 1/2 chain corresponding to
    both local x interactions and a magnetic field in the z direction,
    respectively.
    """
    s0, x, y, z = gen_s0sxsysz(L)
    Hfield = gen_op_total(z)
    Hbond = gen_interaction_kdist(x, k=1, bc=bc)
    return Hbond, Hfield


def gen_Hbond_Hfield_zint(L, bc='obc'):
    """Generates Hamiltonians for a length L spin 1/2 chain corresponding to
    both local z interactions and a magnetic field in the x direction,
    respectively.
    """
    s0, x, y, z = gen_s0sxsysz(L)
    Hfield = gen_op_total(x)
    Hbond = gen_interaction_kdist(z, k=1, bc=bc)
    return Hbond, Hfield


def gen_tnsk(mat, L, S, n, k=0, bc='pbc'):
    """Translates the n-site operator mat on a spin S chain of length L, by a
    single site at a time. Produces a list of all possible translated
    operators.

    If the boundary conditions are open, the operator is translated only up to
    the boundary. If the boundary conditions are periodic, the operator is
    translated to every site of the spin chain.

    The timesink of this function is in the translation.

    Parameters
    ----------
    mat : operator
        The operator to be translated to several sites.
        Acts on n-sites.
    L : int
        An integer describing the length of the spin chain.
    S : float
        An integer or half-integer defining the spin of the operators.
    n : type
        Number of sites on which the matrix mat acts.
    k : type
        The phase given to mat for every translation.
    bc : string
        Description of the boundary conditions used in generating the overall
        interaction. Can be 'obc' (open boundary conditions) or 'pbc' (periodic
        boundary conditions)

    Returns
    -------
    list
        A list of operators which each act like mat on a translated set of n
        sites.

    """
    rep_dim = int(np.round(2*S+1, 1))

    # Initializing list of translated operators.
    m_list = []

    for i in range(L-n+1):
        # Add an operator of the form 1*1*...*1*mat*1*...*1 to the list,
        # up to an additional phase, all the way to the boundary of the chain.
        m_list.append(np.exp(1j*k*i)
                      * sk(
                           sk(sparse.eye(rep_dim**(i)), mat),
                           sparse.eye(rep_dim**(L-n-i))
                           )
                      )

    if bc == 'pbc' and n != 1:
        # If we have periodic boundary conditions (we can perform translation)
        # and if the operator does not already act on a single site (no
        # translation needed):
        P = gen_tr(L, S)
        Pinv = gen_invtr(L, S)
        for i in range(n-1):
            # Append all translated operators which are translated past the
            # boundary.
            m_list.append((np.exp(1j*k))*P@m_list[L-n+i]@Pinv)

    return m_list


def gen_sztot(L, S):
    """Returns the total S_z operator on a spin chain of length L."""
    # Find S_z at the first site.
    first_sz = gen_s0sxsysz(1, S)[3][0]
    # Generate a list of translated S_z operators, with no extra phase.
    # Notice that since S_z acts on a single site, the boundary conditions are
    # unimportant.
    tr_sz_list = gen_tnsk(first_sz, L, S, n=1, k=0, bc='obc')

    # Return a translation invariant sum.
    return gt(tr_sz_list)


# ===================================
# Spin 1/2 Operators at sites:
# ===================================
# Directions for pauli matrices
dirs = ['x', 'y', 'z']


def op_at_sites(pauli_dirs, sites, sigma_list, verbose=0):
    """Returns a product of pauli operators in the directions set by pauli_dirs
    at lattice sites set by sites:
        op_list = [set of sigma_i at the given sites,
                   if the pauli direction at the site is given as i]

    Uses an input sigma_list in order to find the relevant pauli operators
    without computing them each time.

    Recall that for a particular L, these can be obtained by
    sigma_list = gen_s0sxsysz(L)[1:]
    which returns a list of the form
    sigma_list = [sx_list, sy_list, sz_list]
    """
    if verbose > 1:
        print("    sites: " + str(sites))
        if verbose > 2:
            print("    directions: " + str(dirs))
        print("    directions for Pauli operators" + str(pauli_dirs))
        if verbose > 3:
            print("    sigma_list:\n    " + str(sigma_list))
        print("\n")
    op_list = [sigma_list[i][int(site)]
               for isite, site in enumerate(sites)
               for i in range(len(dirs))
               if pauli_dirs[isite] == dirs[i]]
    return gen_op_prod(op_list)


def connected_central_ops(L, sigma_list, verbose=0, max_size=None):
    """Returns operators which are at the center of a lattice,
    without `gaps' of the identity, organized by their size.
    """
    # Setting minimum and maximum operator size
    if L % 2 == 0:
        size = 2
    else:
        size = 1

    if max_size is None:
        max_size = L

    # Setting up a dictionary to hold operators
    op_dict = dict()

    while size <= max_size:
        # consider all strings of the given size
        op_labels = [''.join(i) for i in itertools.product(dirs, repeat=size)]
        sites = np.arange((L-size)/2, (L+size)/2, 1)

        # Generate operators associated with these strings
        ops_this_size = [op_at_sites(list(p_dirs), sites, sigma_list)
                         for p_dirs in op_labels]

        if verbose > 0:
            print("    -----------------------\n")
            print("    Operator size: " + str(size))
            if verbose > 1:
                print("    Operator labels: " + str(op_labels))
            if verbose > 4:
                print("\n    Operators:")
                [print("    %i)\n" % (i+1) + str(Oi.toarray()) + "\n")
                 for i, Oi in enumerate(ops_this_size)]
            print("    -----------------------\n")

        # Store the operators into a dictionary
        for iop, op in enumerate(ops_this_size):
            op_dict.setdefault(size, {})[op_labels[iop]] = op

        if verbose > 3:
            # Printing operators
            pprint.pprint(op_dict, width=40)

        size += 2

    # Returns a dictionary with two keys:
    #   * the first key is the size of a set of central operators;
    #   * the second key is a label, in the form of a string, for the given
    #     central operator
    return op_dict


# ----------------------
# Saving and Loading
# ----------------------

def op_file(Ls, num):
    return 'pauli_central_ops_'+str(Ls[0])+'-'+str(Ls[-1])\
           + '_'+str(num)+'.npz'


def save_connected_central_ops(Ls, sigma_lists):
    """Saves the connected central operators associated with a list of Ls.

    sigma_lists can be obtained by
    sigma_lists = [gen_s0sxsysz(L)[1:] for L in Ls]
    """
    labelnum = 0

    outfile = op_file(Ls, labelnum)
    while os.path.isfile(outfile):
        labelnum += 1
        outfile = op_file(labelnum)

    np.savez(outfile, **{str(L): connected_central_ops(Ls[i], sigma_lists[i])
                         for i, L in enumerate(Ls)})


def load_connected_central_ops(Ls, num=0):
    """Loads the connected central operators associated with a list of Ls."""
    infile = op_file(Ls, num)
    return np.load(infile, allow_pickle=True)
