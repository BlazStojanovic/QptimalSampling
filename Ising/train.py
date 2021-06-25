"""
Training the Ising model
"""

import sys
sys.path.append('../')

from qsampling_utils.loss import ising_endpoint_loss
from qsampling_utils.sampler import step_max, step_gumbel
from qsampling_utils.pCNN import pCNN, CircularConv, check_pcnn_validity

from absl import logging
import ml_collections

from flax import linen as nn
from flax.metrics import tensorboard
from flax.training import train_state

import jax
import jax.numpy as jnp
import jax.random as rnd
import numpy as np

import optax


def get_parameterisation(key, lattice_size, hid_channels, out_channels, kernel, layers):
	"""
	Constructs the parameterisation of the drift for the ising model with lattice side = lattice_size

	Params:
	-------
	key -- PRNGKey
	lattice_size -- size of the Ising lattice, assumed to be square
	hid_channels, out_channels, -- no. of hidden and output channels of the CNN
	kernel -- Tuple of kernel size e.g. (3,3)
	layers -- No layers of the network, including input and output layers

	Returns:
	--------
	initial_params -- initialisation parameters of the CNN
	pcnn -- pCNN object

	"""

	# initialisation size sample
	init_val = jnp.ones((1, lattice_size, lattice_size, 1), jnp.float32)
	
	# check correctness of params
	check_pcnn_validity(lattice_size, kernel, layers)

	# periodic CNN object
	pcnn = pCNN(conv=CircularConv, 
				act=nn.relu,
 				hid_channels=hid_channels, 
 				out_channels=out_channels,
				K=kernel, 
				layers=5, 
				strides=(1,1))

	# initialise the network
	initial_params = pcnn.init(key, init_val)['params']
	# initial_params = pcnn.init({'params':key}, jnp.zeros((1,7,7,1)))

	return initial_params, pcnn


def compute_metrics():
	pass

@jax.jit
def train_step():
	pass

@jax.jit
def eval_step():
	pass

def train_epoch():
	pass

def eval_model():
	pass


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
	"""
	Train basic 2D ising model case with endpoint loss 
	
	Params:
	-------
	config -- Hyperparameter configuration for training and evaluation.
	workdir --  Directory where the tensorboard summaries are written to.
	
	Returns:
	--------
	the trained optimizer, which will contain the found optimised parameterisation of the rates
	
	"""
	
	key = rnd.PRNGKey(0)

	# init tensorboard
	summary_writer = tensorboard.SummaryWriter(workdir)
	summary_writer.hparams(dict(config))


	# variational approximation of the rates
	params, pcnn = get_parameterisation(key, 
				config.lattice_size, 
				config.hid_channels, 
				config.out_channels, 
				config.kernel_size, 
				config.layers)

	key, subkey = rnd.split(key)

	# Adam optimiser init
	tx = optax.adam(config.learning_rate, b1=config.b1, b2=config.b2) # returns gradient transform

	# construct the train state
	state = train_state.TrainState.create(apply_fn=pcnn.apply, params=params, tx=tx)


	# train rates
	for epoch in range(1, config.num_epochs+1):
		
		# split subkeys for shuffling purpuse
		key, subkey = rnd.split(key)

		# optimisation step on one batch
		state = train_epoch()




	# return the trained rates
	return None