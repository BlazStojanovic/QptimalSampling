"""
Jg ratio sampling, computing sigma z, sigma x and E for different g/J ratios
and different Lattice sizes
"""

import sys
sys.path.append('../../../')
sys.path.append('../../')
sys.path.append('../')

import Sampler as sa
import Operators as op

import configs.ising1d_configs as iconf

from jax import vmap
import jax
import jax.numpy as jnp
import numpy as np

import matplotlib.pyplot as plt

dim = 1
lattice_model = 'ising'

sigz_single = op.get_sigz_single(dim, lattice_model)
# sigx_single = op.get_sigx_single(dim, lattice_model)

Szfun = vmap(sigz_single, in_axes=(0), out_axes=(0))
# Sxfun = vmap(sigx_single, in_axes=(0), out_axes=(0))

if __name__ == '__main__':
	r = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8]

	# storage
	Es = np.zeros(len(r))
	sigz = np.zeros(len(r))
	sigx = np.zeros(len(r))
	varEs = np.zeros(len(r))
	varsigz = np.zeros(len(r))
	varsigx = np.zeros(len(r))

	L = 6
	J = 1.0

	key = jax.random.PRNGKey(-1)
	no_sampl = int(10000)
	no_chain = 6

	dtimes = jnp.arange(1, no_sampl+1)

	for i, g in enumerate(r):

		config = iconf.get_defaults()
		config.L = L
		config.g = g
		config.J = J

		experiment_name = "batch_prop_single{}".format(L)
		input_dir = "g{}/final/".format(int(100*g))
		output_dir = "/sampling"

		## Discrete Trajectories
		sampler_disc = sa.Sampler(experiment_name, input_dir, output_dir, config)
		sampler_disc.setup_experiment_folder()
		sampler_disc.load_sampler()
		sampler_disc.initialise_chains(no_chains=no_chain, no_steps=no_sampl)

		# generate discrete samples
		Ts, Fs, S0 = sampler_disc.generate_samples(key, method='cont')
		trajectories = sampler_disc.f2t()

		# evaluate energy, sigz and sigx		
		## sz
		sz = Szfun(trajectories)
		dsigz = jnp.multiply(jnp.cumsum(sz, axis=1), jnp.reciprocal(dtimes))
		adsigz = jnp.average(dsigz, axis=0)
		vdsigz = jnp.var(dsigz, axis=0, ddof=1)
		vdsigz = jnp.sqrt(vdsigz)
		
		sigz[i] = jnp.average(adsigz[-100:])
		varsigz[i] = jnp.average(vdsigz[-100:])

		## sx
		# sx = sxfun(trajectories)
		# dsigx = jnp.multiply(jnp.cumsum(sx, axis=1), jnp.reciprocal(dtimes))
		# adsigx = jnp.average(dsigx, axis=0)
		# vdsigx = jnp.var(dsigx, axis=0, ddof=1)
		# vdsigx = jnp.sqrt(vdsigx)
		
		# sigx[i] = jnp.average(adsigx[-100:])
		# varsigx[i] = jnp.average(vdsigx[-100:])

	# store files
	np.save('../data/batch_prop_single{}/r.npy'.format(L), r)
	np.save('../data/batch_prop_single{}/Evg.npy'.format(L), Es)
	np.save('../data/batch_prop_single{}/varEvg.npy'.format(L), varEs)
	np.save('../data/batch_prop_single{}/sigzvg.npy'.format(L), sigz)
	np.save('../data/batch_prop_single{}/varsigzvg.npy'.format(L), varsigz)
	np.save('../data/batch_prop_single{}/sigxvg.npy'.format(L), sigx)
	np.save('../data/batch_prop_single{}/varsigxvg.npy'.format(L), varsigx)