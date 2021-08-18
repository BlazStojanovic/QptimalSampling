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

	key = jax.random.PRNGKey(-1)
	key1, key2 = jax.random.split(key)

	L = 8
	g = 1.0
	J = 1.0
	i = 0
	no_sampl = int(100000)
	no_chain = 10

	config = iconf.get_defaults()
	config.L = L
	config.g = g
	config.J = J

	config = iconf.get_structure()
	config.L = L
	experiment_name = "tfim1d_structure-L{}".format(L)
	input_dir = "run-{}/final/".format(int(i))
	output_dir = "/sampling"

	sampler_cont = sa.Sampler(experiment_name, input_dir, output_dir, config)
	sampler_cont.setup_experiment_folder()
	sampler_cont.load_sampler()
	sampler_cont.initialise_chains(no_chains=no_chain, no_steps=no_sampl)

	## Continuous Trajectories
	Ts, Fs, S0 = sampler_cont.generate_samples(key1, method='cont')
	trajectories = sampler_cont.f2t()

	# print(Fs)

	np.save('../data/tfim1d_structure-L{}/cont_trajectories.npy'.format(L), trajectories)
	np.save('../data/tfim1d_structure-L{}/cont_times.npy'.format(L), Ts)

	## Discrete Trajectories
	sampler_disc = sa.Sampler(experiment_name, input_dir, output_dir, config)
	sampler_disc.setup_experiment_folder()
	sampler_disc.load_sampler()
	sampler_disc.initialise_chains(no_chains=no_chain, no_steps=no_sampl)

	# generate discrete samples
	Ts, Fs, S0 = sampler_disc.generate_samples(key2, method='disc')
	trajectories = sampler_disc.f2t()

	# print(Fs)

	np.save('../data/tfim1d_structure-L{}/disc_trajectories.npy'.format(L), trajectories)
