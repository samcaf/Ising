# Basic imports
import numpy as np
import matplotlib.pyplot as plt

# Local imports
import ising.utils.calculation_utils as cu


# ========================
# Thermalization Plot Utils
# ========================
# -------------------
# Eigenvalue Distribution
# -------------------


def plot_evaldist(evals, path=None):
    """Plots the eigenvalue distribution of evals and compares it to a
    Gaussian fit.
    """
    # Getting eigenvalue density
    mu = np.mean(evals)
    sigma = np.std(evals)

    # Plotting eigenvalue density
    fig = plt.figure(figsize=(14, 10))
    plt.hist(evals, bins=20, density=True)
    x = np.linspace(evals[0], evals[-1], 100)
    plt.plot(x, np.exp(-(x-mu)**2/2/sigma**2)/np.sqrt(2*np.pi*sigma**2),
             label='Gaussian fit')
    plt.title('Eigenvalue Distribution', fontsize=16)
    plt.legend(fontsize=20)
    plt.xlabel(r'$E$', fontsize=16)
    plt.ylabel(r'$N(E)$', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.tight_layout()

    if path is None:
        plt.show()
    else:
        fig.savefig(path, format='pdf')


# -------------------
# Level Spacing
# -------------------


def plot_lvlspace(evals, ensemble='go', nbins=50, title=None):
    """Tests level statistics of a set of eigenvalues, comparing their level
    spacing distribution to that of a standard random matrix Gaussian ensemble.
    """
    rlist = np.zeros(len(evals)-2)
    for i in range(len(evals)-2):
        rlist[i] = min([(evals[i+2] - evals[i+1]), (evals[i+1]-evals[i])])
        rlist[i] = rlist[i]/max([(evals[i+2] - evals[i+1]),
                                 (evals[i+1]-evals[i])])

    fig = plt.figure(figsize=(10, 8))
    plt.hist(rlist, range=(0, 1), bins=nbins, density=True)
    plt.title(title)

    x = np.linspace(0, 1, 1000)
    # Gaussian orthogonal result:
    if ensemble == 'go':
        go = 27/4*(x+x**2)/(1+x+x**2)**(5/2)
        plt.plot(x, go, 'r', lw=3)
    # Gaussian unitary result:
    if ensemble == 'gu':
        gu = 2*(4/81*np.pi/np.sqrt(3))**(-1) * ((x+x**2)**2)/(1+x+x**2)**(4)
        plt.plot(x, gu, 'r', lw=3)

    return fig


# -------------------
# Eigenstate EV Distribution
# -------------------

def plot_eev_density(L, Op, evals, evecs, path=None):
    op_eev = cu.op_ev(Op, evecs)

    fig = plt.figure(figsize=(10, 8))
    plt.hist2d(evals/L, op_eev, 40)
    plt.ylabel(r'$A_{\alpha,\alpha}$', fontsize=16)
    plt.xlabel(r'$E_{\alpha}/L$', fontsize=16)
    plt.title('Diagonal matrix element density, L=%d' % L, fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.tight_layout()

    if path is None:
        plt.show()
    else:
        fig.savefig(path, format='pdf')

# -------------------
# Microcanonical Comparison
# -------------------


def plot_microcanonical_comparison(L, Op, evals, evecs, deltaE,
                                   spectrum_center=1/2, spectrum_width=20,
                                   path=None):
    op_eev_mid, op_eev_mc, sigmaOp = cu.op_eev_fluct(L, Op, evals, evecs,
                                                     deltaE,
                                                     spectrum_center,
                                                     spectrum_width)

    fig = plt.figure()
    plt.plot(op_eev_mid, '.', label='Eigenstate')
    plt.plot(op_eev_mc, label='Microcanonical')
    plt.title(r'Comparison to Microcanonical, $L=%d$' % L, fontsize=16)
    plt.title(r'L=%d, $\Delta E=0.025L,~\mathcal{O}=S^z_{L/2}$' % L,
              fontsize=16)
    plt.ylim(-0.5, 0.5)
    plt.legend(fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()

    if path is None:
        plt.show()
    else:
        fig.savefig(path, format='pdf')

    return np.mean(sigmaOp)


# Fluctuations
def plot_microcanonical_fluctuations(Ls, sigmaOp_vec, path=None):
    # Plotting the narrowing of fluctuations with system size
    D = 2**np.array(Ls[:])
    fluct = np.sqrt(sigmaOp_vec[:])

    fig = plt.figure()
    plt.loglog(D, fluct, 'o', r'$\sigma_{\mathcal{O}}$')
    slope, b = np.polyfit(np.log(D), np.log(fluct), 1)
    plt.loglog(D, np.exp(b)*(D**slope), label=r'$y=%.2fx + %.2f$' % (slope, b))
    plt.xlabel(r'$D=2^L$', fontsize=16)
    plt.ylabel(r'$\sigma_{\mathcal{O}}$', fontsize=16)
    plt.title('EEV Fluctuations around Microcanonical Value', fontsize=16)
    plt.legend(fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()

    if path is None:
        plt.show()
    else:
        fig.savefig(path, format='pdf')


# -------------------
# Canonical Comparison
# -------------------


def plot_canonical_comparison(L, Op, evals, evecs, path=None):
    op_eev = cu.op_ev(Op, evecs)

    # Positive temperature canonical expectation values
    Ts = np.logspace(-1, 10, 100)
    EList = np.zeros(len(Ts))
    OList = np.zeros(len(Ts))
    for t in range(len(Ts)):
        Gibbs = np.exp(-evals/Ts[t])
        Z = np.sum(Gibbs)
        EList[t] = np.dot(evals, Gibbs)/Z
        OList[t] = np.dot(op_eev, Gibbs)/Z
    # Negative temperature canonical expectation values
    Ts = np.logspace(-1, 10, 100)
    EListneg = np.zeros(len(Ts))
    OListneg = np.zeros(len(Ts))
    for t in range(len(Ts)):
        Gibbs = np.exp(evals/Ts[t])
        Z = np.sum(Gibbs)
        EListneg[t] = np.dot(evals, Gibbs)/Z
        OListneg[t] = np.dot(Gibbs, op_eev)/Z

    # Plotting
    fig = plt.figure(figsize=(14, 10))
    plt.plot(evals/L, op_eev, '.', label='Eigenstates')
    plt.xlabel(r'$\langle H\rangle/L$', fontsize=16)
    plt.ylabel(r'$\langle \mathcal{O} \rangle$', fontsize=16)
    plt.title(r'EEVs, L=%d, $\Delta E=0.025L,~\mathcal{O}=S^z_{L/2}$' % L,
              fontsize=16)
    plt.plot(EList/L, OList, 'r.-',
             label=r'$\langle \mathcal{O} \rangle_T$, positive T')
    plt.plot(EListneg/L, OListneg, 'm.-',
             label=r'$\langle \mathcal{O} \rangle_T$, negative T')
    plt.legend(fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()

    if path is None:
        plt.show()
    else:
        fig.savefig(path, format='pdf')


# DEBUG:
# I want a method for plotting fluctuations here as well


# ========================
# Entropy Plot Utils
# ========================
# -------------------
# Entropy
# -------------------

def plot_eigenstate_entropies(L, evecs, path=None):
    entropies = []

    for cut_x in range(1, L//2):
        entropies.append([cu.entanglement_entropy(evec, cut_x)
                          for evec in evecs])

    # plot3D(np.array(entropies).flatten(), range(1, L//2), evals)
