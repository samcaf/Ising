# Basic imports
import os.path

# Local imports
import ising.models.base as models
from ising.models.save import *
from ising.utils.file_utils import projfile, eigenfile
from examples.thermalization.params import Ls, MODELS, VERBOSE

# ===================================
# Saving a set of model information
# ===================================
# Ensuring that we have symmetry projectors
for model in MODELS:
    for L in Ls:
        print("##################################\n"
              + "Making model: "+str(model) + "; L="+str(L)
              + "\n##################################", flush=True)
        # Setting up model
        model_params, symmetries = models.default_params(model, L=L,
                                                         integrable=False)
        if VERBOSE > 1:
            print("\n  # ===========================\n"
                  + "  Symmetries:"
                  + "\n  # ===========================", flush=True)
            print("  "+str(symmetries)+"\n", flush=True)

        # Getting projectors onto symmetry sectors
        if not os.path.isfile(projfile(L, S=1/2, **symmetries)):
            save_projectors(L, **symmetries)
        else:
            print("Projection file found.", flush=True)

        # Getting diagonalized Hamiltonians
        if not os.path.isfile(eigenfile(model, **model_params)):
            save_default_model(model, L)
        else:
            print("Model file found.", flush=True)
