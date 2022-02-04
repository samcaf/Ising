# Model imports
from ising.models.base import model_1d
from ising.models.ising_type_utils import default_params, gen_model


#######################################
# Ising-Type Models:
#######################################

class ising_model(model_1d):
    """A model class designed to facilitate calculations with 1D
    spin chains and Hamiltonians with sparse matrix representations.
    # Edit
    """
    # Model Hamiltonian
    def gen_H(self, save=True):
        if self.verbose > 0:
            print("Generating Hamiltonian...")
        # Make Hamiltonian:
        H, _, _ = gen_model(name=self.name, L=self.L, model_params=self.params,
                            integrable=self.integrable)
        self.H = H
        self.has_H = True
        if save:
            # Save Hamiltonian:
            self.save_H()
        if self.verbose > 0:
            print("Complete!")

    # Initialization
    def __init__(self, name, symmetries=None, default=True, integrable=False,
                 **params):
        self.integrable = integrable
        if default:
            params, symmetries = default_params(name, params['L'],
                                                integrable=integrable)
        elif symmetries is None:
            _, symmetries = default_params(name, params['L'],
                                           integrable=integrable)
        super().__init__(name=name, symmetries=symmetries, **params)

#######################################
# SUSY Models:
#######################################

#######################################
# Disordered Models:
#######################################
