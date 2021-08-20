
import sys
sys.path.append('../../../')
sys.path.append('../../')
sys.path.append('../')

import Sampler as sa
import Operators as op

import configs.ising2d_configs as iconf

import jax
import jax.numpy as jnp
import numpy as np

import matplotlib.pyplot as plt

if __name__ == '__main__':
	config = iconf.get_defaults()

	seed = [0, 11, 22, 33, 44]
	L = 3
	g = 1.0
	J = 1.0

	config.L = L
	config.g = g
	config.J = J
	config.batch_type = 'permute'
	config.training_mode = 'adaptive'
		
	N = 10000
	no_chains = 1
	
	losses = np.zeros((len(seed), config.num_epochs)) # store losses
	iters = np.zeros((len(seed), config.num_epochs)) # store losses
	times = np.zeros((len(seed), N)) # store losses

	# store iterations and learning loss
	for j, i in enumerate(seed):
		experiment_name = "rate_blowup".format(L)
		input_dir = "{}/final/".format(seed[j])
		output_dir = "/blowup"

		sampler = sa.Sampler(experiment_name, input_dir, output_dir, config)
		sampler.setup_experiment_folder()
		sampler.load_sampler()

		losses[j] = sampler.training_loss
		iters[j] = sampler.validations[:, 2]

		sampler.initialise_chains(no_chains, N)

		key = jax.random.PRNGKey(4)
		Ts, Fs, S0 = sampler.generate_samples(key, method='cont')

		print(Ts)
		trajectories = sampler.f2t()

		times[j] = Ts


	np.save("../data/"+experiment_name+"/losses.npy", losses)
	np.save("../data/"+experiment_name+"/iters.npy", iters)
	np.save("../data/"+experiment_name+"/times.npy", times)
