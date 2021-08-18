import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
from jax import vmap

import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable

from matplotlib import rc
matplotlib.rcParams.update({'font.size': 18})
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{libertine} \usepackage{amsfonts}'

import sys
sys.path.append('../')
sys.path.append('../../')

import Operators as op

dtraj = np.load('../tfim1d/data/tfim1d_structure-L8/disc_trajectories.npy')
ctraj = np.load('../tfim1d/data/tfim1d_structure-L8/cont_trajectories.npy')
ctimes = np.load('../tfim1d/data/tfim1d_structure-L8/cont_times.npy')
dtimes = np.arange(1, jnp.shape(dtraj)[1]+1)

dim = 1
skip = 0
lattice_model = 'ising'

sigz_single = op.get_sigz_single(dim, lattice_model)
sigz = vmap(sigz_single, in_axes=(0), out_axes=(0)) # vectorise over batch

dsigz = sigz(dtraj)
dsigz = jnp.multiply(jnp.cumsum(dsigz, axis=1), jnp.reciprocal(dtimes))
adsigz = jnp.average(dsigz, axis=0)
vdsigz = jnp.var(dsigz, axis=0, ddof=1)
vdsigz = jnp.sqrt(vdsigz)

csigz = sigz(ctraj)
csigz = jnp.multiply(csigz, ctimes)
csigz = jnp.multiply(jnp.cumsum(csigz, axis=1), jnp.reciprocal(jnp.cumsum(ctimes, axis=1)))
acsigz = jnp.average(csigz, axis=0)
vcsigz = jnp.var(csigz, axis=0, ddof=1)
vcsigz = jnp.sqrt(vcsigz)

plt.figure(figsize=(8, 5))

plt.plot(dtimes, adsigz, 'r', label='discrete')
plt.fill_between(dtimes, adsigz - vdsigz, adsigz + vdsigz, color='red', alpha=0.2)

plt.plot(acsigz, 'b', label='continuous')
plt.fill_between(dtimes, acsigz - vcsigz, acsigz + vcsigz, color='blue', alpha=0.2)

plt.ylabel(r'$\langle \sigma_z \rangle$')
plt.xlabel('MC step')
plt.ylim([0.265, 0.29])
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
plt.legend()
plt.savefig("../../../Thesis/Chapter5/Figs/Vector/sampling_example.pdf", bbox_inches='tight')
plt.savefig("../figures/sampling_example.pdf", bbox_inches='tight')
plt.show()

