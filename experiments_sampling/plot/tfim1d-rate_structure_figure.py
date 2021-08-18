import matplotlib.pyplot as plt
import numpy as np

import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable

from matplotlib import rc
matplotlib.rcParams.update({'font.size': 22})
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{libertine} \usepackage{amsfonts}'

import sys
sys.path.append('../')

def get_T(rates):
	return rates*np.reciprocal(np.sum(rates, axis=0))

permutations = np.load('../tfim1d/data/tfim1d_structure-L6/configs.npy')
rates1 = np.load('../tfim1d/data/tfim1d_structure-L6/rates1.npy')
rates2 = np.load('../tfim1d/data/tfim1d_structure-L6/rates2.npy')
rates1 = (rates1.squeeze()).T
rates2 = (rates2.squeeze()).T

trans1 = get_T(rates1)
trans2 = get_T(rates2)

dT = np.abs(trans2 - trans1)

fig, ax = plt.subplots(4, 1, sharex='col')
fig.set_size_inches((14, 6))


ax[0].xaxis.tick_top()
a0 = ax[0].imshow(permutations.T, cmap='Greys')
divider = make_axes_locatable(ax[0])
cax = divider.append_axes('right', size='1%', pad=0.05)
cbar = fig.colorbar(a0, cax=cax, orientation='vertical', format='%.0e')
ax[0].text(3, 3.8, '$S$', rotation=0, color='black', size=30)

a1 = ax[1].imshow(rates1, cmap='Blues')
divider = make_axes_locatable(ax[1])
cax = divider.append_axes('right', size='1%', pad=0.05)
cbar = fig.colorbar(a1, cax=cax, orientation='vertical', format='%.0e')
ax[1].text(3, 3.0, r'$\Gamma^{(v_1)}_{s_i \rightarrow -s_i}$', color='black', size=30)

a2 = ax[2].imshow(rates2, cmap='Blues')
divider = make_axes_locatable(ax[2])
cax = divider.append_axes('right', size='1%', pad=0.05)
cbar = fig.colorbar(a2, cax=cax, orientation='vertical', format='%.0e')
ax[2].text(3, 3.0, r'$\Gamma^{(v_2)}_{s_i \rightarrow -s_i}$', color='black', size=30)

a3 = ax[3].imshow(dT, cmap='Greens')
divider = make_axes_locatable(ax[3])
cax = divider.append_axes('right', size='1%', pad=0.05)
cbar = fig.colorbar(a3, cax=cax, orientation='vertical', format='%.0e')
ax[3].text(3, 3.0, r'$\Delta T_{s_i \rightarrow -s_i}$', color='black', size=30)

ax[0].set_yticks([])
ax[1].set_yticks([])
ax[2].set_yticks([])
ax[3].set_yticks([])

plt.savefig("../../../Thesis/Chapter5/Figs/Vector/rate_structure.pdf", bbox_inches='tight')
plt.savefig("../figures/rate_structure.pdf", bbox_inches='tight')
plt.savefig("../figures/rate_structure.png", bbox_inches='tight')
plt.show()