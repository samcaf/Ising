#!/usr/bin/env python
# coding: utf-8
from __future__ import absolute_import
import numpy as np
import matplotlib.backends.backend_pdf as backend_pdf

# Local imports
from ising.utils.file_utils import figLargeOpsPath, eigenfile
import ising.utils.operator_utils as ou
import ising.utils.ising_plot_utils as ipu
import ising.models.base as models
from examples.params import Ls, MODELS


def plot_operators():
    pass


for model in MODELS:
    sigmaOp_vec = []
    for L in Ls:
        # Get stored eigensystem
        eigen_dict = np.load(eigenfile(model,
                                       **models.default_params(model, L)),
                             allow_pickle=True)
        evals, evecs = eigen_dict['evals'], eigen_dict['evecs']
        sub_evals = eigen_dict['subspace evals']

        # Setting up operators
        sigma_list = ou.gen_s0sxsysz(L)[1:]
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
        ystring_file = backend_pdf.PdfPages(figLargeOpsPath+model+'_ystringeev'
                                            + '_{}sites.pdf'.format(L))
        for k in range(L):
            y_string = central_ops[k]['y'*k]
            fig, eev, eev_mc, sigmaOp = ipu.plot_microcanonical_comparison(
                                        L, y_string, evals, evecs,
                                        deltaE=.025*L)

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

