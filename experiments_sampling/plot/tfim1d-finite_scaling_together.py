
import matplotlib.pyplot as plt
import numpy as np

import matplotlib
from matplotlib import rc
import matplotlib.patches as mpatches
matplotlib.rcParams.update({'font.size': 30})
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{libertine} \usepackage{amsfonts}'

import sys
sys.path.append('../')

import tfim1d.tfim_1d_analytical

# load data
nk3 = np.load('../tfim1d/data/tfim1d_netket/fs-3.npy')
nk6 = np.load('../tfim1d/data/tfim1d_netket/fs-6.npy')
nk12 = np.load('../tfim1d/data/tfim1d_netket/fs-12.npy')
# nk24 = np.load('../tfim1d/data/tfim1d_netket/fs-24.npy')

fig, ax = plt.subplots(1, 3)
fig.set_size_inches(16, 7)

# energy
jg = np.linspace(0.1, 2, 500)
en = tfim1d.tfim_1d_analytical.energy_per_site(jg)
ax[0].plot(jg, en, 'k--', label=r'analytical, $N = \infty$')
ax[0].plot(nk3[0], nk3[1]/3, 'o--', color='magenta', label=r'VMC, $N = 3$', linewidth='2')
ax[0].plot(nk6[0], nk6[1]/6, 'o--', color='purple', label=r'VMC, $N = 6$', linewidth='2')
ax[0].plot(nk12[0], nk12[1]/12, 'ro--', label=r'VMC, $N = 12$', linewidth='2')
# ax[0].plot(nk24[0], nk24[1]/24, 'bo--', label=r'VMC, $N = 24$', linewidth='2')
ax[0].axvline(x=1, linewidth=0.5, color='k')
ax[0].set_ylabel(r"$E_0/N$")
ax[0].set_xlabel(r"$g$")
ax[0].set_xticks([0, 1, 2])

# # mz
jg2 = np.linspace(0.1, 2, 500)
mz2 = tfim1d.tfim_1d_analytical.m_z(jg2)

# ax[1].plot(jg1, mz1, 'k--')
ax[1].plot(jg2, mz2, 'k--')
ax[1].plot(nk3[0], np.abs(nk3[3]), 'o--', color='magenta', linewidth='2')
ax[1].plot(nk6[0], np.abs(nk6[3]), 'o--', color='purple', linewidth='2')
# ax[1].plot(nk12[0], np.abs(nk12[3]), 'ro--', linewidth='2')
# ax[1].plot(nk24[0], np.abs(nk24[3]), 'bo--', linewidth='2')
ax[1].axvline(x=1, linewidth=0.5, color='k')
ax[1].set_ylabel(r"$M_z$")
ax[1].set_xlabel(r"$g$")
ax[1].set_xticks([0, 1, 2])
ax[1].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

# mx
jg = np.linspace(0.01, 2, 500)
mx = tfim1d.tfim_1d_analytical.m_x(jg)

ax[2].plot(jg, mx, 'k--')
ax[2].axvline(x=1, linewidth=0.5, color='k')
ax[2].plot(nk3[0], np.abs(nk3[5]), 'o--', color='magenta', linewidth='2')
ax[2].plot(nk6[0], np.abs(nk6[5]), 'o--', color='purple', linewidth='2')
# ax[2].plot(nk12[0], np.abs(nk12[5]), 'ro--', linewidth='2')
# ax[2].plot(nk24[0], np.abs(nk24[5]), 'bo--', linewidth='2')
ax[2].set_ylabel(r"$M_x$")
ax[2].set_xlabel(r"$g$")
ax[2].set_xticks([0, 1, 2])
ax[2].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

# plt.subplots_adjust(wspace=0.3)
# fig.legend(loc=9)
# analytical = mpatches.Patch('k--', label=r'analytical, $N = \infty$')
fig.legend(bbox_to_anchor=(0.1,0.8,0.85,0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=3)
fig.tight_layout()
fig.subplots_adjust(top=0.7)   

plt.savefig("../figures/tfim1d_finite_scaling.pdf", bbox_inches='tight')
plt.savefig("../../../Thesis/Chapter5/Figs/Vector/tfim1d_finite_scaling.pdf", bbox_inches='tight')
plt.show()