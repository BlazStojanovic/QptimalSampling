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
	
	# Not roboust times 1st

	r = [0.6, 1.0, 1.4]
	t = [1, 2, 4]

	N = 10000
	times = np.zeros((3, 3, N))

	L = 6
	J = 1.0
	epochs = 100
	key = jax.random.PRNGKey(1010)
	no_chain = 1

	# open folder and sample times	
	for i, g in enumerate(r):
		for j, T in enumerate(t):
			conf = iconf.get_defaults()
			conf.batch_type = 'construct'
			conf.batch_Tvar = False

			conf.batch_size = 15
			conf.num_epochs = 100

			conf.T = T
			conf.g = g
			conf.L = L
			conf.J = 1.0

			configi = iconf.get_structure()
			experiment_name = "/batch_prop_troub{}".format(L)
			input_dir = "g{}T{}/final/".format(int(100*g), T)
			output_dir = "/sampling"

			print(input_dir)

			sampler_cont = sa.Sampler(experiment_name, input_dir, output_dir, conf)
			sampler_cont.setup_experiment_folder()
			sampler_cont.load_sampler()
			sampler_cont.initialise_chains(no_chains=no_chain, no_steps=N)

			## Continuous Trajectories
			Ts, Fs, S0 = sampler_cont.generate_samples(key, method='cont')

			print(jnp.shape(Ts))
			times[i, j, :] = Ts

	np.save('../data/batch_prop_single{}/timesF.npy'.format(L), times)
	np.save('../data/batch_prop_single{}/rF.npy'.format(L), r)
	np.save('../data/batch_prop_single{}/tF.npy'.format(L), t)