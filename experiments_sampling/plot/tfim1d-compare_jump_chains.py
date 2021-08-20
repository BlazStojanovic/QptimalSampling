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
rates2 = np.load('../tfim1d/data/batch_prop_single6/rates{}.npy'.format(2))

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
cbar = fig.colorbar(a0, cax=cax, orientation='vertical', format='%.3f')

a1 = ax[1].imshow(trans1, cmap='Greens')
divider = make_axes_locatable(ax[1])
cax = divider.append_axes('right', size='1%', pad=0.05)
cbar = fig.colorbar(a1, cax=cax, orientation='vertical', format='%.3f')

a2 = ax[2].imshow(trans2, cmap='Greens')
divider = make_axes_locatable(ax[2])
cax = divider.append_axes('right', size='1%', pad=0.05)
cbar = fig.colorbar(a2, cax=cax, orientation='vertical', format='%.3f')

a3 = ax[3].imshow(dT, cmap='Greens')
divider = make_axes_locatable(ax[3])
cax = divider.append_axes('right', size='1%', pad=0.05)
cbar = fig.colorbar(a3, cax=cax, orientation='vertical', format='%.3f')

ax[0].set_yticks([])
ax[1].set_yticks([])
ax[2].set_yticks([])
ax[3].set_yticks([])

ax[1].set_ylabel("permute")
ax[2].set_ylabel("construct")
ax[3].set_ylabel("$\Delta T$")

plt.savefig("../../../Thesis/Chapter5/Figs/Vector/rate_compare1.pdf", bbox_inches='tight')
plt.savefig("../figures/rate_compare1.pdf", bbox_inches='tight')
plt.savefig("../figures/rate_compare1.png", bbox_inches='tight')
plt.show()


################################################################################################################
permutations = np.load('../tfim1d/data/tfim1d_structure-L6/configs.npy')
rates1 = np.load('../tfim1d/data/batch_prop_single6/rates{}.npy'.format(1))
rates2 = np.load('../tfim1d/data/batch_prop_single6/rates{}.npy'.format(7))

rates1 = (rates1.squeeze()).T
rates2 = (rates2.squeeze()).T

trans1 = get_T(rates1)
trans2 = get_T(rates2)

dT = np.abs(trans2 - trans1)

fig, ax = plt.subplots(3, 1, sharex='col')
fig.set_size_inches((14, 4.5))

a1 = ax[0].imshow(trans1, cmap='Greens')
divider = make_axes_locatable(ax[0])
cax = divider.append_axes('right', size='1%', pad=0.05)
cbar = fig.colorbar(a1, cax=cax, orientation='vertical', format='%.3f')

a2 = ax[1].imshow(trans2, cmap='Greens')
divider = make_axes_locatable(ax[1])
cax = divider.append_axes('right', size='1%', pad=0.05)
cbar = fig.colorbar(a2, cax=cax, orientation='vertical', format='%.3f')

a3 = ax[2].imshow(dT, cmap='Greens')
divider = make_axes_locatable(ax[2])
cax = divider.append_axes('right', size='1%', pad=0.05)
cbar = fig.colorbar(a3, cax=cax, orientation='vertical', format='%.3f')

ax[0].set_yticks([])
ax[1].set_yticks([])
ax[2].set_yticks([])

ax[0].set_ylabel("$h=0.2$")
ax[1].set_ylabel("$h=1.8$")
ax[2].set_ylabel(r"$\Delta T$")

plt.savefig("../../../Thesis/Chapter5/Figs/Vector/rate_compare2.pdf", bbox_inches='tight')
plt.savefig("../figures/rate_compare2.pdf", bbox_inches='tight')
plt.savefig("../figures/rate_compare2.png", bbox_inches='tight')
plt.show()