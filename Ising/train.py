"""
Training the Ising model
"""

import sys
sys.path.append('../')

# from qsampling_utils.loss import ising_endpoint_loss, ising_potential
from qsampling_utils.sampler import step_max, step_gumbel
from qsampling_utils.pCNN import pCNN, CircularConv, check_pcnn_validity
from ising_loss import ising_endpoint_loss, ising_potential

from absl import logging
import ml_collections

import matplotlib.pyplot as plt

from flax import linen as nn
from flax.metrics import tensorboard
from flax.training import train_state

import jax
import jax.numpy as jnp
import jax.random as rnd
from jax import jit

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
				act=nn.softplus,
 				hid_channels=hid_channels, 
 				out_channels=out_channels,
				K=kernel, 
				layers=layers, 
				strides=(1,1))

	# initialise the network
	initial_params = pcnn.init({'params':key}, init_val)

	return initial_params, pcnn

def visualise_trajectory(times, Ss, flips):
	"""
	Quick tool, mostly for debugging purposes. Displays all states in a trajectory 
	in a grid. Highlights spin flips.
	"""
	N = jnp.shape(times)[0]
	m = int(jnp.ceil(jnp.sqrt(N)))
	fig, ax = plt.subplots(m, m)
	fig.set_size_inches(15, 10)
	for i in range(m):
		for j in range(m):
			if N >= (i*m + j):
				ax[i, j].set_title("t = {:.3f}, flip = {}".format(times[i*m + j, 1], flips[i*m + j, 1]))
				ax[i, j].imshow(Ss[i*m + j, :, :, 0])
			ax[i, j].axis('off') 
	plt.show()	

def initialise_lattice(key, lattice_size):
	"""
	Returns lattice initialised to plus-minus ones, which represent spins up and down.
	"""
	# return jax.lax.stop_gradient((rnd.choice(key, 2, shape=(1, lattice_size, lattice_size, 1))*(-2)+1))
	return ((rnd.choice(key, 2, shape=(1, lattice_size, lattice_size, 1))*(-2)+1))

def get_trajectory(key, pcnn, params, S0, T,  Ns, lattice_size):
	"""
	Obtain a single trajectory of the Ising model CTMC 
	parameterised by the current rates
	
	Params:
	-------
	key -- PRNGKey
	S0 -- Initial state of the Ising model, device array of shape (lattice_size, lattice_size)
	T -- Time in which we consider the CTMC

	Returns:
	--------
	times -- Array of times that were spent in each state
	Ss -- States 
	flips -- Stores actions at each step, I.e. which spin was flipped

	"""
	

	# TODO, this is a provisional implementation. It will cause issues later! 
	# Performance issues due to wierd memory allocation and uncerntainty of the length of the trajectory
	# Consider another implementation, which is of fixed length of the trajectory! 
	# Tleft = T
	# while Tleft > 0.0: 
	# 	pass	 
	# 

	# CAREFUL: this might be a potential source of errors, a fixed trajectory length implementation 
	# might somehow mess up with the loss calculations from batch to batch, for example the CNN can be optimised 
	# for smaller and smaller exponentials in order to generate longer (in time) trajectories which decreases the
	# loss, since it contains the # - E_0 * T term. CHECK IF THIS IS THE CASE!

	Ss = jnp.zeros((Ns, 1, lattice_size, lattice_size, 1))
	l = lattice_size

	times = jnp.zeros((Ns, 1))
	flips = jnp.zeros((Ns, 1), dtype=jnp.int32)

	# initial rates
	rates = pcnn.apply({'params': params['params']}, S0)

	def loop_fun(i, loop_state):
		S0, times, flips, Ss, rates, key = loop_state # TODO, think about removing S0 from the loop_state
		tau, s, key = step_max(key, rates[0, :, :, 0])

		# change current state
		S0 = S0.at[0, s // l, s % l, 0].multiply(-1) # TODO, recheck this may be a source of a hard to spot bug
		# store all of the values
		times = times.at[i, 0].set(tau)
		flips = flips.at[i, 0].set(s)
		Ss = Ss.at[i+1, :, :, :, :].set(S0)

		# get rates
		rates = pcnn.apply({'params': params['params']}, S0)

		return S0, times, flips, Ss, rates, key

	# loop 
	loop_state = S0, times, flips, Ss, rates, key
	S0, times, flips, Ss, rates, key = jax.lax.fori_loop(0, Ns-1, loop_fun, loop_state)

	# waiting time in the last state, the flip can be discarded
	tau, s, key = step_max(key, rates[0, :, :, 0])
	times = times.at[-1, 0].set(tau)
	flips = flips.at[-1, 0].set(s)
	
	return times, flips

# @jit
def flip_to_trajectory(S0, Nt, Nb, Fs, lattice_size):
	"""
	Construct trajectory from the initial state and spin flips

	Params:
	-------
	S0 -- Initial configuration, assumed to be square with lattice_size side (1, l, l, 1) shape
	Nt -- Number of states in trajectory
	Fs -- Spin flips (Nb, Nt, 1) shape
	lattice_size -- lattice size

	Returns:
	--------
	trajectories -- (Nb, Nt, lattice_size, lattice_size, 1)

	"""

	l = lattice_size

	trajectories = jnp.repeat(S0[jnp.newaxis, ...], Nb, axis=0)
	trajectories = jnp.repeat(trajectories, Nt, axis=1)

	# def loop_fun(i, loop_state): # meant to go from 1 to Nt
	for i in range(1, Nt): # TODO speedup with jax.lax.fori_loop()

		# trajectories, Fs, l = loop_state

		# at i-th time step, we multiply appropriate pixels with -1
		# print(l.astype(int))
		F = jnp.ones((Nb, l, l, 1))
		F = F.at[jnp.arange(Nb), Fs[:, i-1, 0] // l, Fs[:, i-1, 0] % l, :].set(-1)

		# print(jnp.shape(F))
		# print(jnp.shape(trajectories[:, i, :, :, :]))

		trajectories = trajectories.at[:, i, :, :, :].set(jnp.multiply(trajectories[:, i-1, :, :, :], F))

		# return trajectories, Fs, l		
	

	# loop_state = trajectories, Fs, lattice_size
	# trajectories, Fs, lattice_size = jax.lax.fori_loop(0, Nt, loop_fun, loop_state)

	return trajectories

def get_batch(key, Nb, times, flips):
	"""
	Returns a batch of trajectories with the same endpoints that are permutations of a 
	single input trajectory. 

	Params:
	-------
	key -- PRNGKey
	Nb -- No. trajectories in the batch
	times -- times of trajectory
	flips -- which action (which spin do we flip) is taken at each step of the trajectory
	
	Returns:
	--------
	trajectories -- (Nb, Nt, 1, lattice_size, lattice_size, 1), batch of trajectories
	times -- (Nb, Nt), batch of times
	flips -- (Nb, Nt), batch of flips
	"""

	# stack the same trajectory Nb times, by creating a new axis 0 and repeating
	Ts = jnp.repeat(times[jnp.newaxis, ...], Nb, axis=0)
	Fs = jnp.repeat(flips[jnp.newaxis, ...], Nb, axis=0)

	@jit
	def loop_fun(i, loop_state):
		# trajectories, Ts, Fs, key = loop_state
		Ts, Fs, key = loop_state

		# create permutation, use the same key for all three permutations. TODO possible source of errors if the permutations are not the same
		Ts = Ts.at[i, 0:-1].set(rnd.permutation(key, Ts[0, 0:-1]))
		Fs = Fs.at[i, 0:-1].set(rnd.permutation(key, Fs[0, 0:-1]))

		# new key for new permutations
		key, subkey = rnd.split(key)
		
		# return trajectories, Ts, Fs, key
		return Ts, Fs, key

	# permute to get Nb trajectories
	loop_state = Ts, Fs, key
	Ts, Fs, key = jax.lax.fori_loop(1, Nb, loop_fun, loop_state)

	return Ts, Fs

def train_epoch(config, key, state, epoch, model, params):
	"""
	Training for a single epoch
	
	Params:
	-------
	config -- the configuration dictionary
	key -- PRNGKey
	state -- training state
	epoch -- index of epoch

	"""

	# randomly generate an initial state S0
	S0 = initialise_lattice(key, config.lattice_size)

	# obtain trajectory
	times, flips = get_trajectory(key, model, params, S0, config.T,  config.trajectory_length, config.lattice_size)

	# permute the trajectory so you get a batch of trajectories with same endpoints
	Ts, Fs = get_batch(key, config.batch_size, times, flips)

	# get the trajectories
	trajectories =  flip_to_trajectory(S0, config.trajectory_length, config.batch_size, Fs, config.lattice_size)

	# metrics
	batch_metrics = []

	# make training step and store metrics
	def loss_fn(params):
		return ising_endpoint_loss(trajectories, Ts, Fs, model, params, config.J, config.g, config.lattice_size)

	grad_fn = jax.value_and_grad(loss_fn)
	vals, grads = grad_fn(state.params)
	# print(grads)
	state = state.apply_gradients(grads=grads)

	# compute mean of metrics across each batch in epoch.
	# batch_metrics_np = jax.device_get(batch_metrics)
	# epoch_metrics_np = {k: np.mean([metrics[k] for metrics in batch_metrics_np]) for k in batch_metrics_np[0]}
	# print('train epoch: %d, loss: %.4f, accuracy: %.4f' % (epoch, epoch_metrics_np['loss'], epoch_metrics_np['energy'] * 100))
	
	print('train_epoch: %d, loss: %.4f' % (epoch, vals))

	return state


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
		state = train_epoch(config, key, state, epoch, pcnn, params)



	# return the trained rates
	return None



def estimate_energy(key, config, pcnn, params):
	"""
	Estimate energy from samples using the current sample rates. 
	"""

	E = 0 
	sigma = 0

	S0 = initialise_lattice(key, config.lattice_size)

	for i in range(int(config.energy_samples*config.skip_percentage)):
		rates = pcnn.apply({'params': params['params']}, S0)
		tau, s, key = step_gumbel(key, rates[0, :, :, 0]) # TODO, recheck this it may not be the same as sampling with Hastings
		# change current state

		S0 = S0.at[0, s // l, s % l, 0].multiply(-1) # TODO, recheck this may be a source of a hard to spot bug

	for i in range(int(config.energy_samples*config.skip_percentage)):
		rates = pcnn.apply({'params': params['params']}, S0)
		tau, s, key = step_gumbel(key, rates[0, :, :, 0])
		# change current state

		S0 = S0.at[0, s // l, s % l, 0].multiply(-1)
		E = Ising_energy()


