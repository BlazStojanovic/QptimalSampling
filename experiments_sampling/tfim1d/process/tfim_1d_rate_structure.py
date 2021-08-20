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

def get_rates_and_configs(L, i):
	config = iconf.get_structure()
	config.L = L
	config.batch_type = 'construct'
	experiment_name = "tfim1d_structure-L{}".format(L)
	input_dir = "run-{}/final/".format(int(i))
	output_dir = "/sampling"

	sampler = sa.Sampler(experiment_name, input_dir, output_dir, config)
	sampler.setup_experiment_folder()
	sampler.load_sampler()
	configs, rates = sampler.visualise_rates()

	print(jnp.sum(rates))

	return configs, rates


if __name__ == '__main__':
	
	L = 6
	i = [0, 1]
	
	configs, rates = get_rates_and_configs(L, i[0])
	np.save('../data/tfim1d_structure-L{}/configs.npy'.format(L), configs)
	np.save('../data/tfim1d_structure-L{}/rates1.npy'.format(L), rates)

	configs, rates = get_rates_and_configs(L, i[1])
	np.save('../data/tfim1d_structure-L{}/rates2.npy'.format(L), rates)


	L = 8
	i = [0, 1]
	
	configs, rates = get_rates_and_configs(L, i[0])
	np.save('../data/tfim1d_structure-L{}/configs.npy'.format(L), configs)
	np.save('../data/tfim1d_structure-L{}/rates1.npy'.format(L), rates)

	configs, rates = get_rates_and_configs(L, i[1])
	np.save('../data/tfim1d_structure-L{}/rates2.npy'.format(L), rates)
