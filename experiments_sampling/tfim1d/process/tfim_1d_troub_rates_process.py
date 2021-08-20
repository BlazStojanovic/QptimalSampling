"""
Jg ratio sampling, computing sigma z, sigma x and E for different J/g ratios
and different Lattice sizes
"""

import sys
sys.path.append('../../../')
sys.path.append('../../')
sys.path.append('../')

import Sampler as sa
import Operators as op

import configs.ising1d_configs as iconf

import jax
import jax.numpy as jnp
import numpy as np

import matplotlib.pyplot as plt

def get_rates_and_configs(L, g):
	config = iconf.get_defaults()
	L = 6
	J = 1.0
	config.L = L
	config.g = g
	config.J = J

	key = jax.random.PRNGKey(-1)
	no_sampl = int(10000)
	no_chain = 6

	experiment_name = "batch_prop_single{}".format(L)
	input_dir = "g{}/final/".format(int(100*g))
	output_dir = "/sampling"

	## Discrete Trajectories
	sampler = sa.Sampler(experiment_name, input_dir, output_dir, config)
	sampler.setup_experiment_folder()
	sampler.load_sampler()
	sampler.initialise_chains(no_chains=no_chain, no_steps=no_sampl)
	configs, rates = sampler.visualise_rates()

	print(jnp.sum(rates))
	return configs, rates

if __name__ == '__main__':
	L = 6
	g = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]

	for i in range(0, 8):
		configs, rates = get_rates_and_configs(L, g[i])

		np.save('../data/batch_prop_single{}/configs.npy'.format(L), configs)
		np.save('../data/batch_prop_single{}/rates{}.npy'.format(L, i+1), rates)
