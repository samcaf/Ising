# Basic imports
import numpy as np

# Local imports
import ising.utils.calculation_utils as cu
import ising.utils.models as models
from ising.examples.params import MODELS

Ls = [8, 9, 10, 11]

for L in Ls:
    pass
    # Make sure projectors sum to 1

for L in Ls:
    for model in MODELS:
        print("##################################\n"
              + "MODEL: "+str(model) + ", L="+str(L)
              + "\n##################################")
        H, _, symmetries = models.gen_model(model, L)
        proj_list = []

        flip_vals = [1, -1] if symmetries['spin_flip'] else [None]
        k_vals = np.arange(L) if symmetries['translation'] else [None]
        inv_vals = [1, -1] if symmetries['inversion'] else [None]

        for flip_val in flip_vals:
            print("  # ===========================\n"
                  + "  Ising Symmetry Value: "+str(flip_val)
                  + "\n  # ===========================")
            # Computational basis states with this value of Ising symmetry
            flip_inds = cu.s_val_inds(L, flip_val)
            for k in k_vals:
                if False:  # k == 0 or k == L/2:
                    print("    # -----------------------")
                    print("    Momentum "+str(k)+"")
                    print("    # -----------------------")
                    for inv_val in inv_vals:
                        P = cu.k_inv_proj(flip_inds, L, S=1/2, k=k,
                                          inv_val=inv_val)
                else:
                    print("    # -----------------------")
                    print("    Momentum "+str(k)+", no inversion:")
                    print("    # -----------------------")
                    P = cu.k_inv_proj(flip_inds, L, S=1/2, k=k, inv_val=None)

                # Checking that it is a symmetry of the Hamiltonian
                squareproj = np.conj(P.T) @ P
                non_comm = np.max(np.abs((squareproj @ H - H @ squareproj)))
                proj_list.append(squareproj.toarray())

                print("      Non-commutation: "+str(non_comm)+"\n")
                if non_comm > 1e-10:
                    raise AssertionError("Large non-commutation value.")
        print("    # -----------------------")
        print("    Checking Projector Sum:")
        print("    # -----------------------")
        delta_sum_proj = np.max(np.abs(sum(proj_list) - np.eye(2**L)))
        print("    Maximum element of |1 - sum_i P_i|: "
              + str(delta_sum_proj)+"\n")
        if delta_sum_proj > 1e-10:
            raise AssertionError("Large deviation of sum(P) from identity.")
