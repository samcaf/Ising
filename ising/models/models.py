# Model imports
from ising.models.base import model
from ising.models.ising_type_utils import gen_model


#######################################
# Ising-Type Models:
#######################################

class ising_type(model):
    """A model class designed to facilitate calculations with 1D
    spin chains and Hamiltonians with sparse matrix representations.
    # Edit
    """
    # Model Hamiltonian
    def gen_H(self, save=True):
        if self.verbose > 0:
            print("Generating Hamiltonian...")
        # Make Hamiltonian:
        H, _, _ = gen_model(self.name, self.L, **self.params)
        self.H = H
        if save:
            # Save Hamiltonian:
            self.save_H()
        if self.verbose > 0:
            print("Complete!")

#######################################
# SUSY Models:
#######################################

#######################################
# Disordered Models:
#######################################
