import numpy as np
from scipy.sparse import csr_matrix, issparse
import re


# ---------------
# Storage Files
# ---------------
# Local directory for storage
isingPath = '/home/gridsan/sfard/Projects/Ising'
storagePath = isingPath+'/operators/'
figPath = isingPath+'/figures/'
figBasicPath = isingPath+'/figures/basic/'
figLargeOpsPath = isingPath+'/figures/largeops/'

valid_models = ['MFIM', 'XXZ', 'ZXXXXZZ', 'SUSY']


def check_valid_models(model):
    assert model in valid_models, "Invalid model."


def projfile(L, S, spin_flip, translation, inversion, u1,
             **params):
    """Standardized filename to store projection operators."""
    # Extra label setup, encoding symmetries
    label = ''
    if spin_flip:
        label += 'f'
    if translation:
        label += 't'
    if inversion:
        label += 'i'
    if u1:
        label += 'u'

    # File setup
    file = 'projectors_'+label+'_L{}_S{}'.format(L, S)+'.npz'
    # Simplifying the filename for default spin 1/2
    if S == 1/2:
        file = 'projectors_'+label+'_L{}'.format(L)+'.npz'
    return storagePath+file


def sysfile(model, **params):
    # Standardized filename to store Hamiltonian and subspace Hamiltonians.
    check_valid_models(model)
    # All models we are considering now are spin 1/2
    param_info = ['_'+f+str(params[f]) for f in params.keys() if f != 'S']
    param_info = ''.join(map(str, param_info))
    return storagePath+model + '_sysfile' + param_info + '.npz'


def eigenfile(model, **params):
    # Standardized filename to store eigensystem and subspace eigensystems.
    check_valid_models(model)
    # All models we are considering now are spin 1/2
    param_info = ['_'+f+str(params[f]) for f in params.keys() if f != 'S']
    param_info = ''.join(map(str, param_info))
    return storagePath+model + '_eigenfile' + param_info + '.npz'


# ---------------
# Sparse Matrices
# ---------------
# Utils for saving and loading sparse matrices.
# See https://stackoverflow.com/a/55524189

def save_sparse_csr(filename, **kwargs):
    arg_dict = dict()
    for key, value in kwargs.items():
        if issparse(value):
            value = value.tocsr()
            arg_dict[key+'_data'] = value.data
            arg_dict[key+'_indices'] = value.indices
            arg_dict[key+'_indptr'] = value.indptr
            arg_dict[key+'_shape'] = value.shape
        else:
            arg_dict[key] = value

    np.savez(filename, **arg_dict)


def load_sparse_csr(filename):
    loader = np.load(filename, allow_pickle=True)
    new_d = dict()
    finished_sparse_list = []
    sparse_postfix = ['_data', '_indices', '_indptr', '_shape']

    for key, value in loader.items():
        IS_SPARSE = False
        for postfix in sparse_postfix:
            if key.endswith(postfix):
                IS_SPARSE = True
                key_orig = re.match('(.*)'+postfix, key).group(1)
                if key_orig not in finished_sparse_list:
                    value_original = csr_matrix((loader[key_orig+'_data'],
                                                 loader[key_orig+'_indices'],
                                                 loader[key_orig+'_indptr']),
                                                shape=loader[key_orig+'_shape']
                                                )
                    new_d[key_orig] = value_original.tolil()
                    finished_sparse_list.append(key_orig)
                break

        if not IS_SPARSE:
            new_d[key] = value

    return new_d
