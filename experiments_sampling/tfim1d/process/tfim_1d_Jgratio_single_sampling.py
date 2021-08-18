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

if __name__ == '__main__':
	
	L = 6
	g = 1.0
	J = 1.0
	N = int(10**6)

	config = iconf.get_defaults()

	config.L = L
	config.g = g
	config.J = J

	experiment_name = "tfim1d_gjrat_single{}".format(L)
	input_dir = "gjrat{}/checkpoints10/".format(int(100*g))
	output_dir = "/sampling"


	sampler = sa.Sampler(experiment_name, input_dir, output_dir, config)
	sampler.setup_experiment_folder()
	sampler.load_sampler()

	key = jax.random.PRNGKey(4)
	
	no_chains = 1
	no_sampl = 1000
	sampler.initialise_chains(no_chains, no_sampl)

	Ts, Fs, S0 = sampler.generate_samples(key, method='cont')
	trajectories = sampler.f2t()

	sampler.time_flip_statistics()
	sampler.visualise_rates()
