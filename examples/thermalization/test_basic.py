#!/usr/bin/env python
# coding: utf-8
from __future__ import absolute_import
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as backend_pdf

# Local imports
from ising.utils.file_utils import load_sparse_csr
from ising.utils.file_utils import projfile, eigenfile
import ising.utils.operator_utils as ou
import ising.utils.calculation_utils as cu
import ising.utils.ising_plot_utils as ipu
import ising.models.base as models
from examples.thermalization.params import Ls, MODELS, figBasicPath


# ==============================
# Extra plot utilities
# ==============================
def plot_eev_density_symm(L, Op, eigen_dict, proj_dict, path=None):
    # Preparing eigensystem information
    sub_evals = eigen_dict['subspace evals'][()]
    sub_evecs = eigen_dict['subspace evecs'][()]

    # Preparing plot
    fig = plt.figure(figsize=(10, 8))
    plt.ylabel(r'$A_{\alpha,\alpha}$', fontsize=16)
    plt.xlabel(r'$E_{\alpha}/L$', fontsize=16)
    plt.title('Diagonal matrix element density, L=%d' % L, fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    # Making and flatting list of all evals
    evals = [sub_evals[sector] for sector in proj_dict.keys()]
    evals = np.array([ev for eval_list in evals for ev in eval_list])
    op_eev = []

    for sector in proj_dict.keys():
        # Setting up information for this symmetry sector
        sector_evecs = sub_evecs[sector]
        P = proj_dict[sector]
        sector_Op = P @ Op @ np.conj(P).T

        print("            Symmetry sector: " + str(sector))
        print("            Projector shape: " + str(np.shape(P)))
        print("            Sector eval shape: " + str(np.shape(P)))
        print("            Sector evec shape: " + str(np.shape(P)))
        print("            Sector operator shape: " + str(np.shape(P)))

        this_eev = cu.op_ev(sector_Op, sector_evecs)
        op_eev = np.concatenate((op_eev, this_eev))

    # Plotting
    plt.hist2d(evals/L, op_eev, 40)

    plt.tight_layout()
    if path is not None:
        fig.savefig(path, format='pdf')
    return fig


def plot_microcanonical_comparison_symm(L, Op, eigen_dict, proj_dict,
                                        path=None):
    # Preparing eigensystem information
    sub_evals = eigen_dict['subspace evals'][()]
    sub_evecs = eigen_dict['subspace evecs'][()]

    # Preparing plot
    fig = plt.figure(figsize=(10, 8))
    plt.title(r'Comparison to Microcanonical, $L=%d$' % L, fontsize=16)
    plt.title(r'L=%d, $\Delta E=0.025L,~\mathcal{O}=S^z_{L/2}$' % L,
              fontsize=16)
    plt.ylim(-0.5, 0.5)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    op_eev_mid, op_ev_mc, sigmaOp = [], [], []

    for sector in proj_dict.keys():
        # Setting up information for this symmetry sector
        sector_evals = sub_evals[sector]
        sector_evecs = sub_evecs[sector]
        P = proj_dict[sector]
        sector_Op = P @ Op @ np.conj(P).T

        # Preparing fluctuation info
        this_eev, this_mc, this_sig = cu.op_eev_fluct(L, sector_Op,
                                                      sector_evals,
                                                      sector_evecs,
                                                      deltaE=.025*L)
        op_eev_mid = np.concatenate((op_eev_mid, this_eev))
        op_ev_mc = np.concatenate((op_ev_mc, this_mc))
        sigmaOp = np.concatenate((sigmaOp, this_sig))

        # Plotting
        plt.plot(op_eev_mid, '.', label='Eigenstate')
        plt.plot(op_ev_mc, label='Microcanonical')

    plt.tight_layout()
    plt.legend(fontsize=16)
    if path is not None:
        fig.savefig(path, format='pdf')
    return fig, op_eev_mid, op_ev_mc, sigmaOp


def plot_canonical_comparison_symm(L, Op, eigen_dict, proj_dict,
                                   path=None, op_desc=None):
    # Preparing eigensystem information
    sub_evals = eigen_dict['subspace evals'][()]
    sub_evecs = eigen_dict['subspace evecs'][()]

    # Preparing plot
    fig = plt.figure(figsize=(10, 8))
    title = r'EEVs, $L=%d,~\Delta E=0.025L$' % L

    if op_desc is not None:
        title = title+r', $\mathcal{O}=$'+op_desc

    plt.title(title, fontsize=16)
    plt.xlabel(r'$\langle H\rangle/L$', fontsize=16)
    plt.ylabel(r'$\langle \mathcal{O} \rangle$', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    # ------------------------------------------------
    # Preparing for canonical ensemble calculations
    # ------------------------------------------------
    # Making and flatting list of all evals
    evals = [sub_evals[sector] for sector in proj_dict.keys()]
    evals = np.array([ev for eval_list in evals for ev in eval_list])
    Ts = np.logspace(-1, 10, 100)

    op_eev = []
    EList, EListneg = [], []

    for sector in proj_dict.keys():
        # Setting up information for this symmetry sector
        sector_evecs = sub_evecs[sector]
        P = proj_dict[sector]
        sector_Op = P @ Op @ np.conj(P).T

        # Setting up operator eigenstate expectation values
        this_eev = cu.op_ev(sector_Op, sector_evecs)
        op_eev = np.concatenate((op_eev, this_eev))

    # Positive temperature canonical expectation values
    EList = np.zeros(len(Ts))
    OList = np.zeros(len(Ts))
    for t in range(len(Ts)):
        Gibbs = np.exp(-evals/Ts[t])
        Z = np.sum(Gibbs)
        EList[t] = np.dot(evals, Gibbs)/Z
        OList[t] = np.dot(op_eev, Gibbs)/Z

    # Negative temperature canonical expectation values
    EListneg = np.zeros(len(Ts))
    OListneg = np.zeros(len(Ts))
    for t in range(len(Ts)):
        Gibbs = np.exp(evals/Ts[t])
        Z = np.sum(Gibbs)
        EListneg[t] = np.dot(evals, Gibbs)/Z
        OListneg[t] = np.dot(op_eev, Gibbs)/Z

    # Plotting
    plt.plot(evals/L, op_eev, '.', label='Eigenstates')
    plt.plot(EList/L, OList, 'r.-',
             label=r'$\langle \mathcal{O} \rangle_T$, positive T')
    plt.plot(EListneg/L, OListneg, 'm.-',
             label=r'$\langle \mathcal{O} \rangle_T$, negative T')

    plt.tight_layout()
    plt.legend(fontsize=16)
    if path is not None:
        fig.savefig(path, format='pdf')
    return fig


# ==============================
# Making plots
# ==============================

for model in MODELS:
    sigmaOp_vec = []
    for L in Ls:
        # ----------------------
        # Setting up model
        # ----------------------
        print("##################################\n"
              + "# Calculations for model: "+str(model) + "; L="+str(L)
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

        print("    # ==============================\n"
              + "    # Making plots\n"
              + "    # ==============================", flush=True)

        # ----------------------
        # Plotting eigenvalue distribution
        # ----------------------
        print("        # --------------------------\n"
              + "        # Plotting eigenvalue distribution\n"
              + "        # --------------------------\n", flush=True)
        ipu.plot_evaldist(all_evals, path=figBasicPath+model
                          + '_evaldist_{}sites.pdf'.format(L))

        # ----------------------
        # Plotting level spacing
        # ----------------------
        # Setting up for multiple pdfs (for different symmetry sectors)
        # in one file
        print("        # --------------------------\n"
              + "        # Plotting level spacing distributions\n"
              + "        # --------------------------\n", flush=True)

        lvlfile = backend_pdf.PdfPages(
                    figBasicPath+model+'_lvlspace_{}sites.pdf'.format(L))

        # Plotting a histogram of the level spacing distribution
        fig = ipu.plot_lvlspace(all_evals, ensemble='go', nbins=50,
                                title='Full System')
        lvlfile.savefig(fig)

        # Ensuring that the system behaves thermally in symmetry sectors
        for sector in sub_evals.keys():
            fig = ipu.plot_lvlspace(sub_evals[sector], ensemble='go',
                                    nbins=50,
                                    title=sector+' Symmetry Sector')
            lvlfile.savefig(fig)

        lvlfile.close()

        # ----------------------
        # Plotting features of eigenstate expectation values
        # ----------------------
        print("        # --------------------------\n"
              + "        # Plotting EEV density\n"
              + "        # --------------------------\n", flush=True)

        _, _, _, sz = ou.gen_s0sxsysz(L)
        Op = sz[L//2]
        plot_eev_density_symm(L, Op, eigen_dict, proj_dict,
                              path=figBasicPath+model
                              + '_eevdensity_{}sites.pdf'.format(L))

        """
        print("        # --------------------------\n"
              + "        # Plotting comparison to\n"
              + "        # microcanonical ensemble\n"
              + "        # --------------------------\n", flush=True)
        _, _, _, fluct = plot_microcanonical_comparison_symm(
                                        L, Op, eigen_dict, proj_dict,
                                        path=figBasicPath+model+'_mc'
                                        + 'comp_{}sites.pdf'.format(L))
        mean_fluct = np.mean(fluct)
        sigmaOp_vec.append(mean_fluct)

        print("        # --------------------------\n"
              + "        # Plotting comparison to\n"
              + "        # canonical ensemble\n"
              + "        # --------------------------\n", flush=True)
        plot_canonical_comparison_symm(L, Op, eigen_dict, proj_dict,
                                          path=figBasicPath+model
                                          + '_canoncomp_{}sites.pdf'.format(L))
        """

    # ----------------------
    # Microcanonical fluctuations:
    # ----------------------
    print("        # --------------------------\n"
          + "        # Plotting fluctuations relative\n"
          + "        # to microcanonical ensemble\n"
          + "        # --------------------------\n", flush=True)

    Ls_str = ''.join(map(str, Ls))
    fluct_path = figBasicPath+model+'_therm_comp_fluctuations'+Ls_str+'.pdf'

    # Plotting fluctuations of operator expectation values
    # relative to the microcanonical ensemble
    """
    ipu.plot_microcanonical_fluctuations(Ls, sigmaOp_vec, path=fluct_path)
    """
