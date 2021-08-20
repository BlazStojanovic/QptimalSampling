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


# sampling check!	
config = iconf.get_defaults()

seed = 0
Tts = [1, 2, 4, 5, 10, 20, 40, 60, 80]
L = 6
g = 0.6
J = 1.0

config.L = L
config.g = g
config.J = J
config.num_epochs = 1000

N = 10000
no_chains = 5

lambs = np.zeros(9)
vlambs = np.zeros(9)

for j, t in enumerate(Tts):

	# store iterations and learning loss
	experiment_name = "batch_prop_troub{}".format(L)
	input_dir = "g{}T{}/final/".format(int(100*g), t)
	output_dir = "/sampling"

	sampler = sa.Sampler(experiment_name, input_dir, output_dir, config)
	sampler.setup_experiment_folder()
	sampler.load_sampler()

	sampler.initialise_chains(no_chains, N)

	key = jax.random.PRNGKey(seed)
	Ts, Fs, S0 = sampler.generate_samples(key, method='cont')

	out = np.average(Ts, axis=1)
	print(out)
	print(np.reciprocal(out))
	lambs[j] = np.average(np.reciprocal(out))
	vlambs[j] = np.sqrt(np.var(Ts, ddof=1))



np.save("../data/"+experiment_name+"/lambs.npy", lambs)
np.save("../data/"+experiment_name+"/vlambs.npy", vlambs)
np.save("../data/"+experiment_name+"/Tts.npy", Tts)