#!/usr/bin/env python
# coding: utf-8
from __future__ import absolute_import
import numpy as np
import matplotlib.backends.backend_pdf as backend_pdf

# Local imports
from ising.examples.params import Ls, MODELS, default_params
from ising.utils.file_utils import figPath, eigenfile
import ising.utils.operator_utils as ou
import ising.utils.ising_plot_utils as ipu

for model in MODELS:
    sigmaOp_vec = []
    for L in Ls:
        # Get stored eigensystem
        eigen_dict = np.load(eigenfile(model, **default_params(model, L)),
                             allow_pickle=True)
        evals, evecs = eigen_dict['evals'], eigen_dict['evecs']
        sub_evals = eigen_dict['subspace evals'][()]  # extracting dict

        # DEBUG:
        # Problem is that the size of the eigensystem is bigger than |H|
        # lens = [len(sub_evals[k]) for k in sub_evals.keys()]
        # print(lens)
        # print(len(lens))
        # print(sum(lens))

        # repcount = 0
        # for i, v in enumerate(evecs):
        #     if i == 0:
        #         print(v)
        #         print(len(v))
        #     for j, w in enumerate(evecs[:i]):
        #         if np.allclose(v, w) and i != j:
        #             repcount += 1
        # print(repcount)

        # DEBUG:
        # L = [7, 8, 9, 10, 11] -> repcount = [4, 24, 28, 120, 124]
        # jumps of 4 extra repeats within even->odd?

        # print(np.shape(evals))
        # print(np.shape(evecs))
        # print(evals)

        # Eigenvalue distribution
        ipu.plot_evaldist(evals, path=figPath+model
                          + '_evaldist_{}sites.pdf'.format(L))

        # Level spacing
        lvlfile = backend_pdf.PdfPages(
                    figPath+model+'_lvlspace_{}sites.pdf'.format(L))

        # Overall level spacing distribution
        fig = ipu.plot_lvlspace(evals, ensemble='go', nbins=50,
                                title='Full System')
        lvlfile.savefig(fig)

        # Ensuring that the system behaves thermally in symmetry sectors
        if L < 10:
            for sector in sub_evals.keys():
                print('test')
                fig = ipu.plot_lvlspace(sub_evals[sector], ensemble='go',
                                        nbins=50,
                                        title=sector+' Symmetry Sector')
                lvlfile.savefig(fig)

        lvlfile.close()

        # Plotting features of eigenstate expectation values
        _, _, _, sz = ou.gen_s0sxsysz(L)
        Op = sz[L//2]
        ipu.plot_eev_density(L, Op, evals, evecs,
                             path=figPath+model
                             + '_eevdensity_{}sites.pdf'.format(L))
        s = ipu.plot_microcanonical_comparison(L, Op, evals, evecs,
                                               deltaE=0.025*L,
                                               path=figPath+model+'_mccomp_'
                                               + '{}sites.pdf'.format(L))
        sigmaOp_vec.append(s)

        ipu.plot_canonical_comparison(L, Op, evals, evecs, path=figPath+model
                                      + '_canoncomp_{}sites.pdf'.format(L))

    Ls_str = ''.join(map(str, Ls))
    fluct_path = figPath+model+'_therm_comp_fluctuations'+Ls_str+'.pdf'

    # Microcanonical fluctuations:
    ipu.plot_microcanonical_fluctuations(Ls, sigmaOp_vec, path=fluct_path)
