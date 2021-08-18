
import matplotlib.pyplot as plt
import numpy as np

import matplotlib
from matplotlib import rc
matplotlib.rcParams.update({'font.size': 30})
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{libertine} \usepackage{amsfonts}'

import sys
sys.path.append('../')

# load data
nk3 = np.load('../tfim2d/data/tfim2d_netket/fs-3.npy')
nk6 = np.load('../tfim2d/data/tfim2d_netket/fs-6.npy')
# nk9 = np.load('../tfim2d/data/tfim2d_netket/fs-9.npy')


fig, ax = plt.subplots(1, 3)
fig.set_size_inches(16, 7)

# energy
ax[0].plot(nk3[0], nk3[1]/9, 'o--', color='purple', label=r'VMC, $N = 3$', linewidth='2')
ax[0].plot(nk6[0], nk6[1]/36, 'ro--', label=r'VMC, $N = 6$', linewidth='2')
# ax[0].plot(nk9[0], nk9[1]/9, 'bo--', label=r'VMC, $N = 9$', linewidth='2')
ax[0].set_ylabel(r"$E_0/N$")
ax[0].set_xlabel(r"$J$")
ax[0].set_xticks([0, 0.35, 0.7])

# mz
ax[1].plot(nk3[0], np.abs(nk3[3]), 'o--', color='purple', linewidth='2')
ax[1].plot(nk6[0], np.abs(nk6[3]), 'ro--', linewidth='2')
# ax[1].plot(nk9[0], np.abs(nk9[3]), 'bo--', linewidth='2')
ax[1].set_ylabel(r"$M_z$")
ax[1].set_xlabel(r"$J$")
ax[1].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax[1].set_xticks([0, 0.35, 0.7])

# mx
ax[2].plot(nk3[0], np.abs(nk3[5]), 'o--', color='purple', linewidth='2')
ax[2].plot(nk6[0], nk6[5], 'ro--', linewidth='2')
# ax[2].plot(nk9[0], nk9[5], 'bo--', linewidth='2')
ax[2].set_ylabel(r"$M_x$")
ax[2].set_xlabel(r"$J$")
ax[2].set_xticks([0, 0.35, 0.7])
ax[2].set_yticks([0.4, 0.6, 0.8, 1.0])

fig.legend(bbox_to_anchor=(0.1,0.9,0.85,0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=3)
fig.tight_layout()
fig.subplots_adjust(top=0.85)   

plt.savefig("../figures/tfim2d_finite_scaling.pdf", bbox_inches='tight')
plt.savefig("../../../Thesis/Chapter5/Figs/Vector/tfim2d_finite_scaling.pdf", bbox_inches='tight')
plt.show()