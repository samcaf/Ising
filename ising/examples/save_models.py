# Basic imports
import os.path

# Local imports
import ising.utils.models as models
from ising.examples.params import Ls, MODELS, VERBOSE
from ising.utils.file_utils import projfile, eigenfile

# ===================================
# Saving a set of model information
# ===================================
# Ensuring that we have symmetry projectors
for model in MODELS:
    for L in Ls:
        print("##################################\n"
              + "Making model: "+str(model) + "; L="+str(L)
              + "\n##################################")
        # Setting up model
        model_params, symmetries = models.default_params(model, L=L,
                                                         integrable=False)
        if VERBOSE > 1:
            print("  # ===========================\n"
                  + "  Symmetries:"
                  + "\n  # ===========================")
            print("  "+str(symmetries)+"\n\n")

        # Getting projectors onto symmetry sectors
        if not os.path.isfile(projfile(L, S=1/2, **symmetries)):
            models.save_projectors(L, **symmetries)
        else:
            print("Projection file found.")

        # Getting diagonalized Hamiltonians
        if not os.path.isfile(eigenfile(model, **model_params)):
            models.save_default_model(model, L)
        else:
            print("Model file found.")
