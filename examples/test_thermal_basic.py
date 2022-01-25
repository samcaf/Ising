#!/usr/bin/env python
# coding: utf-8
from __future__ import absolute_import
import numpy as np
import matplotlib.backends.backend_pdf as backend_pdf

# Local imports
from ising.utils.file_utils import figBasicPath, load_sparse_csr
from ising.utils.file_utils import projfile, eigenfile
import ising.utils.operator_utils as ou
import ising.utils.calculation_utils as cu
import ising.utils.ising_plot_utils as ipu
import ising.models.base as models
from examples.params import Ls, MODELS

for model in MODELS:
    sigmaOp_vec = []
    for L in Ls:
        # ----------------------
        # Setting up model
        # ----------------------
        model_params, symmetries = models.default_params(model, L=L,
                                                         integrable=False)
        # Retrieving stored information
        proj_dict = load_sparse_csr(projfile(L, S=1/2, **symmetries))
        eigen_dict = np.load(eigenfile(model, **model_params),
                             allow_pickle=True)

        # Getting eigensystems
        sub_evals = eigen_dict['subspace evals'][()]
        sub_evecs = eigen_dict['subspace evecs'][()]
        all_evals, all_evecs = cu.esys_from_sub_dicts(proj_dict, eigen_dict)

        # ----------------------
        # Plotting eigenvalue distribution
        # ----------------------
        ipu.plot_evaldist(all_evals, path=figBasicPath+model
                          + '_evaldist_{}sites.pdf'.format(L))

        # ----------------------
        # Plotting level spacing
        # ----------------------
        # Setting up for multiple pdfs (for different symmetry sectors)
        # in one file
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
        _, _, _, sz = ou.gen_s0sxsysz(L)
        Op = sz[L//2]
        ipu.plot_eev_density(L, Op, all_evals, all_evecs,
                             path=figBasicPath+model
                             + '_eevdensity_{}sites.pdf'.format(L))
        _, _, _, fluct = ipu.plot_microcanonical_comparison(
                                        L, Op, all_evals, all_evecs,
                                        deltaE=0.025*L,
                                        path=figBasicPath+model+'_mc'
                                        + 'comp_{}sites.pdf'.format(L))
        mean_fluct = np.mean(fluct)
        sigmaOp_vec.append(mean_fluct)

        ipu.plot_canonical_comparison(L, Op, all_evals, all_evecs,
                                      path=figBasicPath+model
                                      + '_canoncomp_{}sites.pdf'.format(L))

    # ----------------------
    # Microcanonical fluctuations:
    # ----------------------
    Ls_str = ''.join(map(str, Ls))
    fluct_path = figBasicPath+model+'_therm_comp_fluctuations'+Ls_str+'.pdf'

    # Plotting fluctuations of operator expectation values
    # relative to the microcanonical ensemble
    ipu.plot_microcanonical_fluctuations(Ls, sigmaOp_vec, path=fluct_path)
