# Basic imports
import numpy as np

VERBOSE = 4

# ===================================
# Parameters for models
# ===================================
INTEGRABLE = False
Ls = [7, 8, 9]
MODELS = ['ZXXXXZZ']


def default_params(model, L):
    model_params = None
    if model == 'MFIM':
        model_params = {'L': L,
                        'S': 1/2,
                        'J': 1,
                        'hx': (np.sqrt(5)+1)/4.,
                        'hz': 0 if INTEGRABLE else (np.sqrt(5)+5)/8.}
    if model == 'XXZ':
        model_params = {'L': L,
                        'S': 1/2,
                        'a': 1,
                        'b': 1.05,
                        'j2': 0 if INTEGRABLE else .3}
    if model == 'ZXXXXZZ':
        model_params = {'L': L}

    return model_params
