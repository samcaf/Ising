#!/usr/bin/env python
# coding: utf-8
from __future__ import absolute_import
import numpy as np

# Local imports
from ising.examples.params import Ls, MODELS, default_params
from ising.utils.file_utils import eigenfile
import ising.utils.operator_utils as ou
import ising.utils.computational_utils as cu
# import ising.utils.ising_plot_utils as ipu

for model in MODELS:
    sigmaOp_vec = []
    for L in Ls:
        # Get stored eigensystem
        eigen_dict = np.load(eigenfile(model, **default_params(model, L)),
                             allow_pickle=True)
        evals, evecs = eigen_dict['evals'], eigen_dict['evecs']
        sub_evals = eigen_dict['subspace evals']

        # Setting up operators
        sigma_list = ou.gen_s0sxsysz(L)[1:]
        central_ops = ou.connected_central_ops(L, sigma_list, verbose=0)

        # Setting up lists to hold operators
        op_eevs_xs = []
        op_ev_mc_xs = []
        op_flucts_xs = []

        all_op_eevs = []
        all_op_ev_mcs = []
        all_op_flucts = []

        # -----------------
        # Finding EEVs
        # -----------------
        for k in range(L):
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
