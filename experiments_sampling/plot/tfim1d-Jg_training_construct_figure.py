import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
from jax import vmap

import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable

from matplotlib import rc
matplotlib.rcParams.update({'font.size': 22})
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{libertine} \usepackage{amsfonts}'

import sys

sys.path.append('../')
sys.path.append('../../')

fig = plt.figure(figsize=(12, 5))
r = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
L = 6

colors = plt.cm.coolwarm(np.linspace(0,1,len(r)))

def alpha(r, mn, ma):
	return mn + r*(ma-mn)/1.6

for i, g in enumerate(r):
	loss = np.load('../tfim1d/data/batch_prop_single{}/g{}/final/loss.npy'.format(L, int(100*g)))
	epochs = np.arange(1, jnp.shape(loss)[0]+1)

	plt.plot(epochs, loss, '-', color=colors[i], label=r'$h={}$'.format(g))


plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
plt.legend()

plt.xlabel('epoch')
plt.ylabel('loss')

plt.yscale('log')
plt.savefig("../../../Thesis/Chapter5/Figs/Vector/constr-g-loss.pdf", bbox_inches='tight')
plt.savefig("../figures/constr-g-loss.pdf", bbox_inches='tight')
plt.show()