
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

# load nk data
nk6 = np.load('../tfim1d/data/tfim1d_netket/fs-6.npy')

r = np.load('../tfim1d/data/batch_prop_single6/r.npy')
es6 = np.load('../tfim1d/data/batch_prop_single6/Evg.npy')
sigzs6 = np.load('../tfim1d/data/batch_prop_single6/sigzvg.npy')
sigxs6 = np.load('../tfim1d/data/batch_prop_single6/sigxvg.npy')
ves6 = np.load('../tfim1d/data/batch_prop_single6/varEvg.npy')
vsigzs6 = np.load('../tfim1d/data/batch_prop_single6/varsigzvg.npy')
vsigxs6 = np.load('../tfim1d/data/batch_prop_single6/varsigxvg.npy')

# fig, ax = plt.subplots(1, 3)
# fig.set_size_inches(16, 7)

# # energy
# jg = np.linspace(0.1, 2, 500)
# en = tfim1d.tfim_1d_analytical.energy_per_site(jg)
# ax[0].plot(jg, en, 'k--', label=r'analytical, $N = \infty$')

# ax[0].plot(nk6[0], nk6[1]/6, '--', color='blue', label=r'VMC, $N = 6$', linewidth='2')
# ax[0].plot(r, es6, 'o-', color='blue', label=r'Ours, $N = 6$', linewidth='2')
# ax[0].fill_between(r, es6-ves6, es6+ves6, color='blue', alpha=0.2)

# ax[0].axvline(x=1, linewidth=0.5, color='k')
# ax[0].set_ylabel(r"$E_0/N$")
# ax[0].set_xlabel(r"$g$")
# ax[0].set_xticks([0, 1, 2])

# # # mz
# jg2 = np.linspace(0.1, 2, 500)
# mz2 = tfim1d.tfim_1d_analytical.m_z(jg2)

# ax[1].plot(jg2, mz2, 'k--')
# ax[1].plot(nk6[0], np.abs(nk6[3]), '--', color='blue', linewidth='2')
# ax[1].plot(r, sigzs6, 'o-', color='blue', linewidth='2')
# ax[1].fill_between(r, sigzs6-vsigzs6, sigzs6+vsigzs6, color='blue', alpha=0.2)

# ax[1].axvline(x=1, linewidth=0.5, color='k')
# ax[1].set_ylabel(r"$M_z$")
# ax[1].set_xlabel(r"$g$")
# ax[1].set_xticks([0, 1, 2])
# ax[1].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

# # mx
# jg = np.linspace(0.01, 2, 500)
# mx = tfim1d.tfim_1d_analytical.m_x(jg)

# ax[2].plot(jg, mx, 'k--')
# ax[2].axvline(x=1, linewidth=0.5, color='k')

# ax[2].plot(nk6[0], np.abs(nk6[5]), '--', color='blue', linewidth='2')
# ax[2].plot(r, sigxs6, 'o-', color='blue', linewidth='2')
# ax[2].fill_between(r, sigxs6-vsigxs6, sigxs6+vsigxs6, color='blue', alpha=0.2)


# ax[2].set_ylabel(r"$M_x$")
# ax[2].set_xlabel(r"$g$")
# ax[2].set_xticks([0, 1, 2])
# ax[2].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

# fig.legend(bbox_to_anchor=(0.1,0.85,0.85,0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=3)
# fig.tight_layout()
# fig.subplots_adjust(top=0.8)   

# plt.savefig("../figures/tfim1d_finite_scaling.pdf", bbox_inches='tight')
# plt.savefig("../../../Thesis/Chapter5/Figs/Vector/tfim1d_finite_scaling.pdf", bbox_inches='tight')
# plt.show()

######################################################################################################

matplotlib.rcParams.update({'font.size': 22})

fig, ax = plt.subplots(1, 1)
fig.set_size_inches((8, 5))
jg2 = np.linspace(0.1, 2, 500)
mz2 = tfim1d.tfim_1d_analytical.m_z(jg2)

ax.plot(jg2, mz2, 'k--')
ax.plot(nk6[0], np.abs(nk6[3]), '--', color='blue', linewidth='2', label=r'VMC, $N = 6$')
ax.plot(r, sigzs6, 'o-', color='blue', linewidth='2',label=r'Ours, $N = 6$')
ax.fill_between(r, sigzs6-vsigzs6, sigzs6+vsigzs6, color='blue', alpha=0.2)

ax.axvline(x=1, linewidth=0.5, color='k')
ax.set_ylabel(r"$M_z$")
ax.set_xlabel(r"$g$")
ax.set_xticks([0, 1, 2])
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

plt.legend()
plt.savefig("../figures/tfim1d_finite_scaling-justsigz.pdf", bbox_inches='tight')
plt.savefig("../../../Thesis/Chapter5/Figs/Vector/tfim1d_finite_scaling-justsigz.pdf", bbox_inches='tight')
plt.show()