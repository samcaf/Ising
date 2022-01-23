#!/usr/bin/env python
# coding: utf-8
from __future__ import absolute_import
import numpy as np
import matplotlib.backends.backend_pdf as backend_pdf

# Local imports
from ising.utils.file_utils import figBasicPath, eigenfile
import ising.utils.operator_utils as ou
import ising.utils.ising_plot_utils as ipu
import ising.models.base as models
from examples.params import Ls, MODELS

for model in MODELS:
    sigmaOp_vec = []
    for L in Ls:
        # Get stored eigensystem
        model_params, _ = models.default_params(model, L)
        eigen_dict = np.load(eigenfile(model, **model_params),
                             allow_pickle=True)
        evals, evecs = eigen_dict['evals'], eigen_dict['evecs']
        sub_evals = eigen_dict['subspace evals'][()]

        # Eigenvalue distribution
        ipu.plot_evaldist(evals, path=figBasicPath+model
                          + '_evaldist_{}sites.pdf'.format(L))

        # Level spacing
        lvlfile = backend_pdf.PdfPages(
                    figBasicPath+model+'_lvlspace_{}sites.pdf'.format(L))

        # Overall level spacing distribution
        fig = ipu.plot_lvlspace(evals, ensemble='go', nbins=50,
                                title='Full System')
        lvlfile.savefig(fig)

        # Ensuring that the system behaves thermally in symmetry sectors
        for sector in sub_evals.keys():
            fig = ipu.plot_lvlspace(sub_evals[sector], ensemble='go',
                                    nbins=50,
                                    title=sector+' Symmetry Sector')
            lvlfile.savefig(fig)

        lvlfile.close()

        # Plotting features of eigenstate expectation values
        _, _, _, sz = ou.gen_s0sxsysz(L)
        Op = sz[L//2]
        ipu.plot_eev_density(L, Op, evals, evecs,
                             path=figBasicPath+model
                             + '_eevdensity_{}sites.pdf'.format(L))
        s = ipu.plot_microcanonical_comparison(L, Op, evals, evecs,
                                               deltaE=0.025*L,
                                               path=figBasicPath+model+'_mc'
                                               + 'comp_{}sites.pdf'.format(L))
        sigmaOp_vec.append(s)

        ipu.plot_canonical_comparison(L, Op, evals, evecs,
                                      path=figBasicPath+model
                                      + '_canoncomp_{}sites.pdf'.format(L))

    Ls_str = ''.join(map(str, Ls))
    fluct_path = figBasicPath+model+'_therm_comp_fluctuations'+Ls_str+'.pdf'

    # Microcanonical fluctuations:
    ipu.plot_microcanonical_fluctuations(Ls, sigmaOp_vec, path=fluct_path)
