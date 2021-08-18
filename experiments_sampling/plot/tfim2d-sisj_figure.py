import matplotlib.pyplot as plt
import numpy as np

import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable

from matplotlib import rc
matplotlib.rcParams.update({'font.size': 18})
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{libertine} \usepackage{amsfonts}'

import sys
sys.path.append('../')

## TODOOOOOOOOOOOOOOOOOOOOO
fig, ax = plt.subplots(1, 3)
fig.set_size_inches(16, 7)

# load data
sisj24 = np.reshape(np.arange(24*24), (24, 24))

a00 = ax[0].imshow(sisj24, origin='lower', cmap='Blues') # TODO add , vmin=0, vmax=1 after results
divider = make_axes_locatable(ax[0])
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(a00, cax=cax, orientation='vertical')
ax[0].set_ylabel(r"i")
ax[0].set_xlabel(r"j")

a00 = ax[1].imshow(sisj24, origin='lower', cmap='Blues')
divider = make_axes_locatable(ax[1])
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(a00, cax=cax, orientation='vertical')
ax[1].set_ylabel(r"i")
ax[1].set_xlabel(r"j")

a00 = ax[2].imshow(sisj24, origin='lower', cmap='Blues')
divider = make_axes_locatable(ax[2])
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(a00, cax=cax, orientation='vertical')
ax[2].set_ylabel(r"i")
ax[2].set_xlabel(r"j")

ax[0].set_title(r"$\langle \hat \sigma^z_i \hat \sigma^z_j \rangle$, $J/h=!!!$")
ax[1].set_title(r"$\langle \hat \sigma^z_i \hat \sigma^z_j \rangle$, $J/h=!!!$")
ax[2].set_title(r"$\langle \hat \sigma^z_i \hat \sigma^z_j \rangle$, $J/h=!!!$")

plt.subplots_adjust(wspace=0.3)
# plt.savefig("../../../Thesis/Chapter5/Figs/Raster/tfim1d-sisj-z.png", bbox_inches='tight')
# plt.savefig("../figures/tfim1d-sisj-z.png", bbox_inches='tight')
# plt.show()
