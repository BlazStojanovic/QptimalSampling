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

permutations = np.load('../tfim1d/data/batch_prop_single6/configs.npy')

fig, ax = plt.subplots(8, 1, sharex='col')
fig.set_size_inches((14, 12))

ax[0].xaxis.tick_top()
a0 = ax[0].imshow(permutations.T, cmap='Greys')
divider = make_axes_locatable(ax[0])
cax = divider.append_axes('right', size='1%', pad=0.05)
cbar = fig.colorbar(a0, cax=cax, orientation='vertical', format='%.2f')
ax[0].text(3, 3.8, '$S$', rotation=0, color='black', size=30)

for i in range(1, 8):
	rates = np.load('../tfim1d/data/batch_prop_single6/rates{}.npy'.format(i))
	rates = (rates.squeeze()).T

	a1 = ax[i].imshow(rates, cmap='Blues')
	ax[i].set_yticks([])
	ax[i].set_ylabel(r'h = {:.1f}'.format(0.2+i*0.2))

	divider = make_axes_locatable(ax[i])
	cax = divider.append_axes('right', size='1%', pad=0.05)
	cbar = fig.colorbar(a1, cax=cax, orientation='vertical', format='%.3e')
	# ax[1].text(3, 3.0, r'$\Gamma^{(v_1)}_{s_i \rightarrow -s_i}$', color='black', size=30)

plt.savefig("../../../Thesis/Chapter5/Figs/Vector/troub_rates.pdf", bbox_inches='tight')
plt.savefig("../figures/troub_rates.pdf", bbox_inches='tight')
plt.show()