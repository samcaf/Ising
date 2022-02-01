#!/usr/bin/env python
# coding: utf-8
from __future__ import absolute_import
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as backend_pdf

# Local imports
from ising.utils.file_utils import eigenfile
import ising.utils.operator_utils as ou
import ising.utils.calculation_utils as cu
import ising.utils.ising_plot_utils as ipu
import ising.models.base as models
from examples.thermalization.params import Ls, MODELS, figLargeOpsPath


def plot_opstring(L, Op, k, evals, evecs):
    op_eev_mid, op_ev_mc, sigmaOp = cu.op_eev_fluct(L, Op, evals, evecs,
                                                    deltaE=.025*L,
                                                    spectrum_center=1/2,
                                                    spectrum_width=20)

    fig = plt.figure()
    plt.plot(op_eev_mid, '.', label='Eigenstate')
    plt.plot(op_ev_mc, label='Microcanonical')
    plt.title(r'Comparison to Microcanonical, $L=%d$' % L, fontsize=16)
    plt.title(r'$L=%d,~\Delta E=0.025L,$' % L
              + r'~$\mathcal{O}=(S^y)^{\otimes %d}\$' % k,
              fontsize=16)
    plt.ylim(-0.5, 0.5)
    plt.legend(fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()

    return fig, op_eev_mid, op_ev_mc, sigmaOp


def plot_operators():
    pass


for model in MODELS:
    sigmaOp_vec = []
    for L in Ls:
        # ----------------------
        # Setting up model
        # ----------------------       
        print("##################################\n"
              + "Calculations for model: "+str(model) + "; L="+str(L)
              + "\n##################################", flush=True)
              
        model_params, symmetries = models.default_params(model, L=L,
                                                         integrable=False)
        # Retrieving stored information
        print("    # ==============================\n"
              + "    # Retrieving stored information\n"
              + "    # ==============================", flush=True)
        proj_dict = load_sparse_csr(projfile(L, S=1/2, **symmetries))
        eigen_dict = np.load(eigenfile(model, **model_params),
                             allow_pickle=True)
        
        # Getting eigensystems
        print("        # --------------------------\n"
              + "        # Getting eigensystems\n"
              + "        # --------------------------\n", flush=True)
        sub_evals = eigen_dict['subspace evals'][()]
        sub_evecs = eigen_dict['subspace evecs'][()]
        all_evals, all_evecs = cu.esys_from_sub_dicts(proj_dict, eigen_dict)

        # Setting up operators
        sigma_list = ou.gen_s0sxsysz(L)[1:]
        # DEBUG: Don't need to do all of this for right now
        central_ops = ou.connected_central_ops(L, sigma_list, verbose=0)

        # Setting up lists to hold operators
        op_eevs_ys = []
        op_ev_mc_ys = []
        op_flucts_ys = []

        all_op_eevs = []
        all_op_ev_mcs = []
        all_op_flucts = []

        # -----------------
        # Finding and Plotting EEVs
        # -----------------
        # Setting up pdf with many plots
        ystring_file = backend_pdf.PdfPages(figLargeOpsPath+model+'_ystringev'
                                            + 'fixed_{}sites.pdf'.format(L))
        for k in range(L):
            y_string = central_ops[k]['y'*k]
            fig, eev, ev_mc, sigmaOp = ipu.plot_opstring(L, y_string, k,
                                                         evals, evecs)
            ystring_file.savefig(fig)

            op_eevs_ys.append(eev)
            op_ev_mc_ys.append(ev_mc)
            op_flucts_ys.append(sigmaOp)

            """
            all_op_eevs_k = []
            all_op_ev_mcs_k = []
            all_op_flucts_k = []

            # Enumerating over connected central operators
            for op_string in central_ops[k]:
                _, eevs = cu.op_ev(op_string, evecs)
                _, ev_mc, fluct = cu.op_eev_fluct(L, op_string, evals, evecs,
                                                  deltaE=.025*L)

                all_op_eevs_k.append(eevs)
                all_op_ev_mcs.append(ev_mc)
                all_op_flucts_k.append(fluct)

                # If the operator is a string of pauli xs,
                # consider it individually as well
                if central_ops[k].key(op_string) == 'x'*k:
                    op_eevs_xs.append(eevs)
                    op_ev_mc_xs.append(ev_mc)
                    op_flucts_xs.append(fluct)

            all_op_eevs.append(all_op_eevs_k)
            all_op_ev_mcs.append(all_op_ev_mcs_k)
            all_op_flucts.append(all_op_flucts_k)
            """
        ystring_file.close()

        fig = plt.figure()
        for k in range(L):
            fig.scatter([k]*len(op_eevs_ys), op_eevs_ys[k])
        fig.plot(range(L), op_ev_mc_ys)
        fig.fill_between(x=range(L), y1=op_ev_mc_ys+op_flucts_ys,
                         y2=op_ev_mc_ys-op_flucts_ys)

        fig.savefig(figLargeOpsPath+model+'_ystringev_{}sites.pdf'.format(L))

        # Store these lists in a dictionary of some sort

        """
        #---------------------------------
        # Plotting EEVs
        #---------------------------------
        # Set up figures:
        fig_eev_1 = aesthetic_fig()
        fig_flucts_1 = aesthetic_fig()
        fig_eev_all = aesthetic_fig()
        fig_flucts_all = aesthetic_fig()

        # Plotting
        # Could potentially set this up beforehand instead of storing data

        # Plotting for single operator
        fig_eev_1[0].plot(range(L), op_eevs_1)
        fig_flucts_1[0].plot(range(L), op_flucts_1)

        # Plotting for central operators
        for k in range(L):
            fig_eev_all[0].scatter(k, all_op_eevs[k])
            fig_flucts_all[0].scatter(k, all_op_flucts[k])
        """
