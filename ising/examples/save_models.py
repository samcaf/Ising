# Basic imports
import os.path

# Local importss
import ising.utils.models as models
from ising.examples.params import Ls, MODELS, default_params
from ising.utils.file_utils import projfile, eigenfile

# ===================================
# Saving a set of model information
# ===================================
# Ensuring that we have symmetry projectors
for L in Ls:
    if not os.path.isfile(projfile(L, S=1/2)):
        models.save_projectors(L)

for model in MODELS:
    for L in Ls:
        if not os.path.isfile(eigenfile(model, **default_params(model, L))):
            model_params = {'L': L}
            models.save_model(model, model_params)
