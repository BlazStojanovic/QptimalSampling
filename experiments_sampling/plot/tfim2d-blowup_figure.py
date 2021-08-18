
import matplotlib.pyplot as plt
import numpy as np

import matplotlib
from matplotlib import rc
matplotlib.rcParams.update({'font.size': 22})
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{libertine} \usepackage{amsfonts}'
import matplotlib.lines as mlines

import sys
sys.path.append('../')

losses = np.load('../tfim2d/data/rate_blowup/losses.npy')
iters = np.load('../tfim2d/data/rate_blowup/iters.npy')
times = np.load('../tfim2d/data/rate_blowup/times.npy')

fig = plt.figure(figsize=(12, 7))
gs = fig.add_gridspec(2,3)

ax1 = fig.add_subplot(gs[0, :])
ax11 = ax1.twinx()
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1])
ax4 = fig.add_subplot(gs[1, 2])

epochs = np.arange(1, 31)

# top figure with two axis, loss and no. iterations
ax11.plot(epochs, iters[0], 'bv-')
ax11.plot(epochs, iters[2], 'ro-')
ax11.plot(epochs, iters[3], 's-', color='purple')

ax1.plot(epochs, losses[0], 'kv--')
ax1.plot(epochs, losses[2], 'ko--')
ax1.plot(epochs, losses[3], 'ks--')
ax1.set_xlabel('epoch')
ax1.set_ylabel('loss')
ax11.set_ylabel('No. jumps in $\Sigma_{[0, T=2]}$')

x = np.linspace(0, 0.2, 200)
lam = 9
ax2.hist(times[0], 50, density=True, facecolor='b', alpha=0.75, label='$\tau$', edgecolor='black')
lam1 = 1/np.average(times[0])
ax2.plot(x, lam*np.exp(-lam*x), color='black', linestyle='--', linewidth=2)
ax2.plot(x, lam1*np.exp(-lam1*x), color='blue', linestyle='--', linewidth=2)

ax3.hist(times[2], 50, density=True, facecolor='r', alpha=0.75, label='$\tau$', edgecolor='black')
lam1 = 1/np.average(times[2])
ax3.plot(x, lam*np.exp(-lam*x), color='black', linestyle='--', linewidth=2)
ax3.plot(x, lam1*np.exp(-lam1*x), color='red', linestyle='--', linewidth=2)

ax4.hist(times[3], 50, density=True, facecolor='purple', alpha=0.75, label='$\tau$', edgecolor='black')
lam1 = 1/np.average(times[3])
ax4.plot(x, lam*np.exp(-lam*x), color='black', linestyle='--', linewidth=2)
ax4.plot(x, lam1*np.exp(-lam1*x), color='purple', linestyle='--', linewidth=2)

ax2.text(0.015, 24, r"$\lambda \sim {:.2e}$".format(1/np.average(times[0])))
ax2.text(0.08, 5, r"$\lambda_0 \sim {:.1f}$".format(9))
ax3.text(0.003, 175, r"$\lambda \sim {:.2e}$".format(1/np.average(times[2])))
ax4.text(0.001, 460, r"$\lambda \sim {:.2e}$".format(1/np.average(times[3])))

ax2.set_xlim([0, 0.2])
ax3.set_xlim([0, 0.03])
ax4.set_xlim([0, 0.01])
ax2.set_xlabel(r'$\tau$')
ax3.set_xlabel(r'$\tau$')
ax4.set_xlabel(r'$\tau$')
ax2.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
ax3.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
ax4.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
ax11.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
ls = mlines.Line2D([], [], color='k', linestyle='--', label='Losses')
it = mlines.Line2D([], [], color='k', linestyle='-', label='No. jumps')
ax11.legend(handles=[ls, it])
ax2.set_ylabel(r"$P(\tau) \sim \lambda e^{-\lambda x}$")

plt.subplots_adjust(hspace=0.4)
plt.savefig("../../../Thesis/Chapter5/Figs/Vector/scale_blowup.pdf", bbox_inches='tight')
plt.savefig("../figures/scale_blowup.pdf", bbox_inches='tight')
plt.show()