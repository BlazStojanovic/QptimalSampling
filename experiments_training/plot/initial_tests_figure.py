import matplotlib.pyplot as plt
import numpy as np

import matplotlib
from matplotlib import rc
from mpl_toolkits.axes_grid1 import make_axes_locatable

matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{libertine} \usepackage{amsfonts}'

# hparams searched
widths =  (np.arange(11)*2 + 1).astype(int) # TODO
layers =  (np.arange(11) + 2).astype(int)
b =  (np.arange(0, 11)*10 + 2).astype(int)
t =  (np.arange(0, 11)*10 + 2).astype(int)

time_nb = np.load('../data/initial_tests_fig/NbT_timeapprox.npy')
time_wl = np.load('../data/initial_tests_fig/WL_timeapprox.npy')
iter_nb = np.load('../data/initial_tests_fig/NbT_sizeapprox.npy')
iter_wl = np.load('../data/initial_tests_fig/WL_sizeapprox.npy')
avg_nb = np.load('../data/initial_tests_fig/NbT_avg_loss.npy')
avg_wl = np.load('../data/initial_tests_fig/WL_avg_loss.npy')
var_nb = np.load('../data/initial_tests_fig/NbT_var_loss.npy')
var_wl = np.load('../data/initial_tests_fig/WL_var_loss.npy')
# nb_full = np.load('../data/initial_tests_fig/NbT_full.npy')
# wl_full = np.load('../data/initial_tests_fig/WL_full.npy')

fig, ax = plt.subplots(2, 2, gridspec_kw={'width_ratios': [1, 1]})
fig.set_size_inches(10, 10)

# avg plots
a00 = ax[0, 0].imshow(avg_nb, cmap='Blues')
divider = make_axes_locatable(ax[0, 0])
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(a00, cax=cax, orientation='vertical')

# https://matplotlib.org/stable/tutorials/colors/colormapnorms.html

a10 = ax[1, 0].imshow(avg_wl, cmap='Blues')
divider = make_axes_locatable(ax[1, 0])
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(a10, cax=cax, orientation='vertical')

# var plots
a01 = ax[0, 1].imshow(var_nb, cmap='Blues')
divider = make_axes_locatable(ax[0, 1])
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(a01, cax=cax, orientation='vertical')

a11 = ax[1, 1].imshow(var_wl, cmap='Blues')
divider = make_axes_locatable(ax[1, 1])
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(a11, cax=cax, orientation='vertical')

ax[0, 0].set_xticks(np.arange(11))
ax[0, 0].set_xticklabels(t)
ax[0, 0].set_yticks(np.arange(11))
ax[0, 0].set_yticklabels(b)
ax[0, 1].set_xticks(np.arange(11))
ax[0, 1].set_xticklabels(t)
ax[0, 1].set_yticks(np.arange(11))
ax[0, 1].set_yticklabels(b)
ax[1, 0].set_xticks(np.arange(11))
ax[1, 0].set_xticklabels(layers)
ax[1, 0].set_yticks(np.arange(11))
ax[1, 0].set_yticklabels(widths)
ax[1, 1].set_xticks(np.arange(11))
ax[1, 1].set_xticklabels(layers)
ax[1, 1].set_yticks(np.arange(11))
ax[1, 1].set_yticklabels(widths)

ax[0, 0].set_ylabel('$N_b$')
ax[0, 0].set_xlabel('$T$')
ax[1, 0].set_ylabel('$N_w$')
ax[1, 0].set_xlabel('$N_l$')
ax[0, 0].set_title('Average loss, $N_b$ vs $T$')
ax[1, 0].set_title('Average loss, $N_w$ vs $N_l$')
ax[0, 1].set_ylabel('$N_b$')
ax[0, 1].set_xlabel('$T$')
ax[1, 1].set_ylabel('$N_w$')
ax[1, 1].set_xlabel('$N_l$')
ax[0, 1].set_title('Variance of loss, $N_b$ vs $T$')
ax[1, 1].set_title('Variance of loss, $N_w$ vs $N_l$')

# axis sharing
ax[0, 1].sharey(ax[0, 0])
ax[1, 1].sharey(ax[1, 0])

plt.setp(ax[0, 0].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
plt.setp(ax[0, 1].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
plt.setp(ax[1, 0].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
plt.setp(ax[1, 1].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

plt.subplots_adjust(wspace=0.3, hspace=0.35)
plt.savefig("../../../Thesis/Chapter5/Figs/Raster/avg_var_loss.png", bbox_inches='tight')
plt.savefig("../figures/avg_var_loss.png", bbox_inches='tight')
plt.show()

# times plots

# sizes plots
