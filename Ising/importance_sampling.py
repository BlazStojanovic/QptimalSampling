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

import train_rates
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
	key = jax.random.PRNGKey(99)
	if config.architecture == "pCNN":
		print("hello there pcnn")
		params, model = train_rates.get_parameterisation(key, 
			config.lattice_size, 
			config.hid_channels, 
			config.out_channels, 
			config.kernel_size, 
			config.layers)
	else:
		raise ValueError("Only pCNN supported at the moment.")

	# print(b)
	# print(params)
	ser = serialization.from_bytes(params, b)
	# print(ser)

	return model, ser, config


def ground_state_estimate(key, model, params, config, therm_percent, Nsampl):
	# random state
	S0 = train_rates.initialise_lattice(key, config.lattice_size)
	key, subkey = jax.random.split(key)

	E = 0.0
	sig = 0.0
	T = 0.0

	# thermalisation
	for i in range(int(Nsampl*therm_percent)):
		rates = model.apply({'params': params['params']}, S0)
		tau, s, key = step_gumbel(key, rates[0, :, :, 0])
		# change current state

		S0 = S0.at[0, s // l, s % l, 0].multiply(-1)

	for i in range(int(Nsampl*(1.0-therm_percent))):
		# print(i)
		rates = model.apply({'params': params['params']}, S0)
		# print(rates[0, :, :, 0])
		tau, s, key = step_gumbel(key, rates[0, :, :, 0])

		# energy contribution of first state
		V = il.ising_potential_single(S0, config.J, config.g)*tau
		T = il.passive_difference_single(S0, config.J, config.g, model, params)*tau		
		print(V, T)

		E = (T+V)*tau

		# time
		T += tau

		# change current state
		S0 = S0.at[0, s // l, s % l, 0].multiply(-1)
		

	return E/T, sig

if __name__ == '__main__':
	path = "ising_rates/6" # path to the directory with learned rates
	key = jax.random.PRNGKey(87654321)
	Nsampl = 200
	therm_percent = 0.2
	l = 6 # lattice size, fix

	model, params, config = load_rates(path)

	E, sig = ground_state_estimate(key, model, params, config, therm_percent, Nsampl)

	print("Energy E0 = {} +- {}".format(E, sig))