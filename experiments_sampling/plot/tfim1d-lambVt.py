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

g = 0.6
L = 6
lam0 = g*L

experiment_name = "batch_prop_troub6"
lambs = np.load("../tfim1d/data/"+experiment_name+"/lambs.npy")
vlambs = np.load("../tfim1d/data/"+experiment_name+"/vlambs.npy")
t = np.load("../tfim1d/data/"+experiment_name+"/Tts.npy")

fig = plt.figure(figsize=(8, 5))
plt.plot([0, 100], [lam0, lam0], 'k--', linewidth=1.5, label=r'$\lambda_0$')
plt.plot(t, lambs, 'r^', label=r'estim. $\lambda$')
plt.xlim([0, 85])
plt.xlabel('T')
plt.ylabel(r'$\lambda$')
plt.legend()

plt.savefig("../../../Thesis/Chapter5/Figs/Vector/Tvlamb.pdf", bbox_inches='tight')
plt.savefig("../figures/Tvlamb.pdf", bbox_inches='tight')
plt.show()
