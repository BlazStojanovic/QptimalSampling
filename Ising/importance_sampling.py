"""
Importance sampling for the Ising model
"""

import sys
import json

import ml_collections 
import jax
import jax.numpy as jnp

from flax import serialization

sys.path.append("../")

from qsampling_utils.sampl_utils import step_max, step_gumbel
from qsampling_utils.pCNN import pCNN, CircularConv, check_pcnn_validity

import train_rates as tr
import ising_loss as il

def load_rates(dir_path):
	"""
	Loads rates from one of the experiment folders, 
	returns everything needed for importance sampling the model
	"""

	# load hyperparams
	with open(path+'/config.json', 'r') as json_file:
	    config = json.load(json_file)
	config = ml_collections.ConfigDict(config)

	# load binary
	f = open(path+'/params.txt', 'rb')
	b = f.read()
	
	# construct appropriate model
	key = jax.random.PRNGKey(1)
	if config.architecture == "pCNN":
		print("hello there pcnn")
		params, model = tr.get_parameterisation(key, 
			config.lattice_size, 
			config.hid_channels, 
			config.out_channels, 
			config.kernel_size, 
			config.layers)
	else:
		raise ValueError("Only pCNN supported at the moment.")

	ser = serialization.from_bytes(params, b)

	return model, ser, config


def ground_state_estimate(key, model, params, config, therm_percent, Nsampl):
	# random state
	S0 = tr.initialise_lattice(key, config.lattice_size)
	key, subkey = jax.random.split(key)

	E = 0.0
	sig = 0.0
	T = 0.0

	config.batch_size = 1
	config.T = 10

	# obtain trajectory
	times, flips = tr.get_trajectory1(key, model, params, S0, config)

	# permute the trajectory so you get a batch of trajectories with same endpoints
	Ts, Fs = tr.get_batch(key, config.batch_size, times, flips)

	# get the trajectories
	trajectories =  tr.flip_to_trajectory(S0, jnp.shape(Ts)[1], config.batch_size, Fs, config.lattice_size)

	rate_transitions = il.get_rate_transitions(config.J, config.g, config.lattice_size, model, params)

	# print("Doublecheck that (Nb, Nt, L, L, 1), ", jnp.shape(trajectories))
	# print("Doublecheck that (Nb, Nt, 1), ", jnp.shape(Ts))

	logRN = 0.0
	V = il.ising_potentialV(trajectories, config.J, config.g)
	Vt = jnp.multiply(Ts.squeeze(), V.squeeze())

	T1 = il.passive_difference(trajectories, config.J, config.g, model, params)
	T1t = jnp.multiply(Ts.squeeze(), T1.squeeze())

	T2 = rate_transitions(trajectories, Fs)
	T2s = T2

	E = V + T1 + T2

	return E/T, sig

if __name__ == '__main__':
	path = "ising_rates/79" # path to the directory with learned rates
	key = jax.random.PRNGKey(1)
	Nsampl = 1000
	therm_percent = 0.0
	# l = 6 # lattice size, fix

	model, params, config = load_rates(path)

	E, sig = ground_state_estimate(key, model, params, config, therm_percent, Nsampl)

	print("Energy E0 = {} +- {}".format(E, sig))