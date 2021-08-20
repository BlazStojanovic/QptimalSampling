import matplotlib.pyplot as plt
import numpy as np

import matplotlib
from matplotlib import rc
matplotlib.rcParams.update({'font.size': 18})
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{libertine} \usepackage{amsfonts}'
import matplotlib.lines as mlines

import sys
sys.path.append('../')

## for sig z
fig, ax = plt.subplots(3, 3)
fig.set_size_inches(12, 10.5)

times = np.load('../tfim1d/data/batch_prop_single6/timesF.npy')
r = np.load('../tfim1d/data/batch_prop_single6/rF.npy')
t = np.load('../tfim1d/data/batch_prop_single6/tF.npy')

print(np.shape(times[0, 0, :]))

for i, g in enumerate(r):
	for j, T in enumerate(t):
		x = np.linspace(0, 0.5, 200)
		lam = 6*g
		ax[j, i].hist(times[i, j, :].squeeze(), 50, density=True, facecolor='b', alpha=0.75, label='$\tau$', edgecolor='black')
		lam1 = 1/np.average(times[i, j, :])
		ax[j, i].plot(x, lam*np.exp(-lam*x), color='black', linestyle='--', linewidth=2)
		ax[j, i].plot(x, lam1*np.exp(-lam1*x), color='blue', linestyle='--', linewidth=2)

		ax[j, i].set_title("$h = {}$, $T = {}$".format(g, T))
		ax[j, i].set_ylim(0, 8)
		ax[j, i].set_xlim(0, 0.6)
		
		ax[j, i].text(0.3, 5, r"$\lambda \sim {:.2f}$".format(lam1))
		ax[j, i].text(0.3, 6, r"$\lambda_0 \sim {:.2f}$".format(lam))

# share appropriate axis
ax[0, 0].set_ylabel(r'$\tau \sim P(\lambda)$')
ax[1, 0].set_ylabel(r'$\tau \sim P(\lambda)$')
ax[2, 0].set_ylabel(r'$\tau \sim P(\lambda)$')

ax[2, 0].set_xlabel(r'$\tau$')
ax[2, 1].set_xlabel(r'$\tau$')
ax[2, 2].set_xlabel(r'$\tau$')

plt.subplots_adjust(wspace=0.2, hspace=0.5)
plt.savefig("../../../Thesis/Chapter5/Figs/Vector/troub.pdf", bbox_inches='tight')
plt.savefig("../figures/troub.pdf", bbox_inches='tight')
plt.show()


