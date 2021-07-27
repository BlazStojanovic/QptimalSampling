"""
Trainer class, a class for training and conducting training experiments for learning the rates 
for lattice models
"""

import os
import ml_collections
import time
import json
import numpy as np

from dataclasses import dataclass

# qsampling utils imports 
from qsampling_utils.sampl_utils import step_max, step_gumbel
from qsampling_utils.pCNN import pCNN, CircularConv, check_pcnn_validity

# Lattice imports
from Ising.ising_loss import ising_endpoint_loss, get_Ieploss

# Jax imports
import jax
import jax.numpy as jnp
import jax.random as rnd
from jax import jit, vmap, pmap
import optax

# Flax imports
from flax import linen as nn
from flax import serialization
from flax.training import train_state
import flax.training.checkpoints as checks

@dataclass
class Trainer:
	experiment_name: str
	config: ml_collections.ConfigDict
	output_dir: str

	def train_rates(self, prngn=5, save_ckps=True, save_final=True):

		"""
		train a model to find optimal rates for a lattice system
	
		Params:
		-------

		Returns:
		--------
		sampler -- Sampler object, which allows for importance sampling from the rates
		
		"""
		self.setup_experiment_folder()

		params, model, tx, state, loss_, valids_, epoch_start = self.init_training()

		key = rnd.PRNGKey(prngn)
		for epoch in range(epoch_start, self.config.num_epochs+1):
			# split subkeys for shuffling purpuse
			key, subkey = rnd.split(key)

			# optimisation step on one batch
			state, vals, eest, epoch_time, it = self.train_epoch(subkey, state, epoch, model, params)
			loss_ = loss_.at[epoch-1].set(vals)
			valids_ = valids_.at[epoch-1, 0].set(eest)
			valids_ = valids_.at[epoch-1, 1].set(epoch_time)
			valids_ = valids_.at[epoch-1, 2].set(it)

			if ((epoch % self.config.chpt_freq == 0) and epoch > 0) and save_ckps:
				self.save_chp(epoch, state, loss_, valids_)

		if save_final:
			self.save_experiment(epoch, state, loss_, valids_)

		# construct sampler object with learned rates			
		sampler = (state, vals, params)

		# return TODO
		return sampler

	def init_training(self, prngn=0):
		# PRNGKey is fixed for reproducibility
		key = rnd.PRNGKey(prngn)

		# variational approximation of the rates
		params, model = self.get_rate_parametrisation(key)

		# optimiser init
		tx = self.get_optimiser()

		# construct train state
		state = self.get_train_state(params, model, tx)

		if self.from_beg:
					# construct storage
			loss_ = jnp.zeros((self.config.num_epochs,))
			valids_ = jnp.zeros((self.config.num_epochs, self.config.no_valids))

			epoch_start = 1
		else: 
			raise NotImplementedError

		return params, model, tx, state, loss_, valids_, epoch_start

	def setup_epoch(self):
		# setup functions for train epoch
		def init_lat(key, dim, L):
			"""
			Initialise the lattice with an appropriate shape
			"""
			if self.config.lattice_model == 'ising':
				print("Solving for L = {}, J = {}, g = {}".format(self.config.L, self.config.J, self.config.g))
				print("T = {}, batch = {}".format(self.config.T, self.config.batch_size))
				print("-----------------------------------------------------------------------------------")

				if dim == 1:
					return rnd.choice(key, 2, shape=(1, L, 1, 1))*(-2)+1
				elif dim == 2:
					return rnd.choice(key, 2, shape=(1, L, L, 1))*(-2)+1
				else:
					raise ValueError("Only dims 1 or 2 allowed")
				
			elif self.config.lattice_model == 'heisenberg':
				raise ValueError("hesenberg model not yet implemented")
			else:
				raise ValueError("only ising and heisenberg models available")

		def const_traj(key, model, params, S0, latt_model, dim, T, Nmax, l):
			"""
			Obtain a single trajectory parameterised by the rates
			
			Params:
			-------
			key -- PRNGKey
			S0 -- Initial state
			T -- Time in which we consider the CTMC

			Returns:
			--------
			times -- Array of times that were spent in each state
			Ss -- States 
			flips -- Stores actions at each step, I.e. which spin was flipped, or which two spins were exchanged
			"""

			if latt_model == 'ising':
				times = jnp.zeros((Nmax, 1))
				flips = jnp.zeros((Nmax, 1), dtype=jnp.int32)

				# initial rates
				rates = model.apply({'params': params['params']}, S0)

				# be careful, the Nmax must be sufficiently large for the simulation to work, JAX wont cry out 
				# if out of bounds assignment is called on times or flips! 
				# def len_check(i):
				# 	return i >= Nmax

				# def trf(operand):
				# 	return jnp.pad(operand, (0, Nmax)) # if i geq Nmax pad array, check edge case?

				# def faf(operand):
				# 	return operand

				def loop_fun(loop_state):
					S0, times, flips, rates, key, it, time, Tmax = loop_state
					tau, s, key = step_gumbel(key, rates[0, :, :, 0])

					# change current state
					S0 = S0.at[0, s // l, s % l, 0].multiply(-1) # this should work for 1d as well, s // l will always be 0
					times = times.at[it, 0].set(tau)
					flips = flips.at[it, 0].set(s)

					# get rates
					rates = model.apply({'params': params['params']}, S0)
					it += 1
					time += tau

					# pred = len_check(it)
					# times = jax.lax.cond(pred, trf, faf, times)
					# flips = jax.lax.cond(pred, trf, faf, flips)

					return S0, times, flips, rates, key, it, time, Tmax

				def cond_fun(loop_state):
					S0, times, flips, rates, key, it, time, Tmax = loop_state
					return time < Tmax

				# loop 
				it = 0
				time = 0
				loop_state = S0, times, flips, rates, key, it, time, T
				S0, times, flips, rates, key, it, time, T = jax.lax.while_loop(cond_fun, loop_fun, loop_state)

				# print(time)
				# print(it)
				# fix the last time
				times = times.at[it-1, 0].add((T-time))
				# print(jnp.sum(times))

				# # waiting time in the last state, the flip can be discarded
				# tau, s, key = step_max(key, rates[0, :, :, 0])
				# times = times.at[-1, 0].set(tau)
				# flips = flips.at[-1, 0].set(s)
				
				return times, flips, it # just up to last iteration
			
			elif self.config.lattice_model == 'heisenberg':
				raise NotImplementedError

		def get_btc(key, Nb, times, flips):
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

		def f2p(S0, Nt, Nb, Fs, l):
			"""
			Construct trajectory from the initial state and spin flips

			Params:
			-------
			S0 -- Initial configuration, assumed to be square with lattice_size side (1, l, l, 1) shape
			Nt -- Number of states in trajectory
			Fs -- Spin flips (Nb, Nt, 1) shape
			l -- lattice size

			Returns:
			--------
			trajectories -- (Nb, Nt, lattice_size, lattice_size, 1)

			"""
			trajectories = jnp.repeat(S0[jnp.newaxis, ...], Nb, axis=0)
			trajectories = jnp.repeat(trajectories, Nt, axis=1)

			def loop_fun(i, loop_state): # meant to go from 1 to Nt
				trajectories = loop_state

				# at i-th time step, we multiply appropriate pixels with -1
				F = jnp.ones((Nb, l, l, 1))
				F = F.at[jnp.arange(Nb), Fs[:, i-1, 0] // l, Fs[:, i-1, 0] % l, :].set(-1)

				trajectories = trajectories.at[:, i, :, :, :].set(jnp.multiply(trajectories[:, i-1, :, :, :], F))

				return trajectories

			loop_state = trajectories

			trajectories = jax.lax.fori_loop(1, Nt, loop_fun, loop_state)

			return trajectories

		## lattice initialisation
		self.initialise_lattice = lambda key: init_lat(key, self.config.dim, self.config.L)
		self.initialise_lattice = jit(self.initialise_lattice) # jit the init

		## sampling a single path
		ctr = lambda model, params, key, S0: const_traj(key, model, params, S0, 
													self.config.lattice_model, 
													self.config.dim, 
													self.config.T, 
													self.config.t_vector_increment, 
													self.config.L)
		self.get_trajectory = nn.jit(ctr)
		# self.get_trajectory = jit(self.get_trajectory)

		## permute to get a batch with fixed endpoints
		gbtc = lambda key, times, flips: get_btc(key, self.config.batch_size, times, flips)
		self.get_batch = jit(gbtc)
		# self.get_trajectory = jit(self.get_trajectory)

		## construct state trajectories from the flips
		ftt = lambda S0, Nt, Fs: f2p(S0, Nt, self.config.batch_size, Fs, self.config.L)
		self.flip_to_trajectory = ftt

		## loss function
		if self.config.lattice_model == 'ising':
			self.lossf = get_Ieploss(self.config.J, self.config.g, self.config.L, self.config.dim)
		else:
			raise NotImplementedError

		def train_epoch(key, state, epoch, model, params):
			"""
			Training for a single epoch
			
			Params:
			-------
			config -- the configuration dictionary
			key -- PRNGKey
			state -- training state
			epoch -- index of epoch

			"""
			start_time = time.time()

			# randomly generate an initial state S0
			S0 = self.initialise_lattice(key)

			key, subkey = rnd.split(key)

			# obtain trajectory
			times, flips, it = self.get_trajectory(model, params, key, S0)
			times, flips = times[:it, :], flips[:it, :]
			key, subkey = rnd.split(key)

			# permute the trajectory so you get a batch of trajectories with same endpoints
			Ts, Fs = self.get_batch(key, times, flips)

			# get the trajectories
			trajectories =  self.flip_to_trajectory(S0, jnp.shape(Ts)[1], Fs)

			# make training step and store metrics
			def loss_fn(params):
				return self.lossf(model, params, trajectories, Ts, Fs)

			# get rid of trajectories, flips and arrays
			grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
			(vals, eest), grads = grad_fn(state.params)
			state = state.apply_gradients(grads=grads)

			epoch_time = time.time() - start_time
			print('train_epoch: {} in {:0.2f} sec, loss: {:.4f}, no iterations: {}, energy est: {:.4f}'.format(epoch, epoch_time, vals, it, eest))	
			# logging.debug('train_epoch: {} in {:0.2f} sec, loss: {:.4f}, no iterations: {}, energy est: {:.4f}'.format(epoch, epoch_time, vals, it, eest))	

			return state, vals, eest, epoch_time, it

		self.train_epoch = train_epoch

	def setup_experiment(self):
		# setup rate param function
		if self.config.architecture == "pcnn":
			def get_pcnn(key):
				"""
				Constructs the parameterisation of the drift with the parameters in self.config

				Returns:
				--------
				initial_params -- initialisation parameters of the CNN
				pcnn -- pCNN object

				"""

				# initialisation size sample
				if self.config.dim == 2:
					init_val = jnp.ones((1, self.config.L, self.config.L, 1), jnp.float32)
				elif self.config.dim == 1:
					init_val = jnp.ones((1, 1, self.config.L, 1), jnp.float32)
				else:
					raise ValueError("Only one and two dimensions supported (dim=1 or dim=2)")

				# check correctness of params
				check_pcnn_validity(self.config.L, self.config.kernel_size, self.config.layers)

				# periodic CNN object
				pcnn = pCNN(conv=CircularConv, 
							act=nn.softplus,
			 				hid_channels=self.config.hid_channels, 
			 				out_channels=self.config.out_channels,
							K=self.config.kernel_size, 
							layers=self.config.layers, 
							strides=(1,1))

				# initialise the network
				initial_params = pcnn.init({'params': key}, init_val)
				return initial_params, pcnn
			self.get_rate_parametrisation = get_pcnn
		else:
			raise ValueError("Only periodic CNN (pcnn) available.")

		# setup optimiser
		if self.config.optimizer == "adam":
			def f():
				return optax.adam(self.config.learning_rate, b1=self.config.b1, b2=self.config.b2)
			self.get_optimiser = f
		else:
			raise ValueError("Only adam at the moment.")

		# setup trainstate
		def gts(params, model, tx):
			return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

		self.get_train_state = gts

		# setup epoch step
		self.setup_epoch()

		# indicate setup from scratch
		self.from_beg = True

	def setup_experiment_folder(self):

		# default path
		path = 'data/' + self.experiment_name + '/' + self.output_dir

		# check if folder exists
		try:	
			if not os.path.exists(path):
				os.makedirs(path)
			if not os.path.exists(path + '/checkpoints'):
				os.mkdir(path + '/checkpoints')
			if not os.path.exists(path + '/final'):
				os.mkdir(path + '/final')
		except OSError:
			print('creation of out directory failed')
			exit(0)

	def save_chp(self, epoch, state, loss, valid, keep=10):
		path = 'data/' + self.experiment_name + '/' + self.output_dir + '/checkpoints'
		checks.save_checkpoint(path, state, epoch, keep=keep, overwrite=True)
		
		# save configuration
		with open(path+'/config.json', 'w') as fp:
			json.dump(self.config.to_dict(), fp)

		# save everything else
		np.save(path+'/'+'loss.npy', loss)
		np.save(path+'/valids.npy', valid)
		
		params = state.params
		bytes_output = serialization.to_bytes(params)
		f = open(path+'/params.txt', 'wb')
		f.write(bytes_output)
		f.close()

	def save_experiment(self, epoch, state, loss, valid, prefix='checkpoint_final'):
		path = 'data/' + self.experiment_name + '/' + self.output_dir + '/final'
		checks.save_checkpoint(path, state, epoch, overwrite=True)
		
		# save configuration
		with open(path+'/config.json', 'w') as fp:
			json.dump(self.config.to_dict(), fp)

		# save everything else
		np.save(path+'/'+'loss.npy', loss)
		np.save(path+'/valids.npy', valid)
		
		params = state.params
		bytes_output = serialization.to_bytes(params)
		f = open(path+'/params.txt', 'wb')
		f.write(bytes_output)
		f.close()

	def setup_from_checkpoint(self):
		pass



