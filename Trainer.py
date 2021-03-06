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
import resource

# qsampling utils imports 
from qsampling_utils.sampl_utils import step_max, step_gumbel
from qsampling_utils.pCNN import *

# Lattice imports
from Ising.ising_loss import ising_endpoint_loss, get_Ieploss, get_logRN

# Jax imports
import jax
from jax.interpreters import xla
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

		params, model, tx, state, loss_, valids_, epoch_start = self.init_training(prngn=prngn)

		key = rnd.PRNGKey(prngn)
		key, subkey = rnd.split(key)
		for epoch in range(epoch_start, self.config.num_epochs+1):
			# split subkeys for shuffling purpuse
			key, subkey = rnd.split(key)

			# optimisation step on one batch
			if self.config.training_mode == 'adaptive':
				# the rates adapt, we are truly sampling from the variational rates
				state, vals, eest, epoch_time, it = self.train_epoch(subkey, state, epoch, model, state.params) 
			elif self.config.training_mode == 'initial':
				# the rates stay the same, we are sampling from the initial rates
				state, vals, eest, epoch_time, it = self.train_epoch(subkey, state, epoch, model, params)
			elif self.config.training_mode == 'passive':
				# sample from the passive rates
				raise NotImplementedError
				state, vals, eest, epoch_time, it = self.train_epoch(subkey, state, epoch, model, params)

			print('Memory usage: {} (Mb)'.format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*0.001))

			loss_ = loss_.at[epoch-1].set(vals)
			valids_ = valids_.at[epoch-1, 0].set(eest)
			valids_ = valids_.at[epoch-1, 1].set(epoch_time)
			valids_ = valids_.at[epoch-1, 2].set(it)

			if jnp.isnan(vals):
				break

			if ((epoch % self.config.chpt_freq == 0) and epoch > 0) and save_ckps:
				self.save_chp(epoch, state, loss_, valids_)

		if save_final:
			self.save_experiment(epoch, state, loss_, valids_)

		# construct sampler object with learned rates			
		sampler = (state, vals, params)

		# return
		return sampler

	def init_training(self, prngn=0):
		# PRNGKey is fixed for reproducibility
		key = rnd.PRNGKey(prngn)

		# optimiser init
		tx = self.get_optimiser()

		# variational approximation of the rates
		# params, model = self.get_rate_parametrisation(key)
		params, model = self.params, self.model

		# construct storage
		loss_ = jnp.zeros((self.config.num_epochs,))
		valids_ = jnp.zeros((self.config.num_epochs, self.config.no_valids))
	
		if self.from_beg:
			epoch_start = 1
		else: 
			loss_ = jnp.zeros((self.config.num_epochs,))
			valids_ = jnp.zeros((self.config.num_epochs, self.config.no_valids))

			# load loss and rates from checkpoint
			loaded_loss = np.load(self.load_folder + 'loss.npy')
			loaded_valids = np.load(self.load_folder + 'valids.npy')

			loss_ = loss_.at[:np.shape(loaded_loss)[0]].set(loaded_loss)
			valids_ = valids_.at[:np.shape(loaded_valids)[0], :].set(loaded_valids)

			epoch_start = jnp.shape(loaded_loss)[0] # start at the checkpoint

			# use learned parameters from last chkp instead of the random ones
			# params = self.load_params(self.load_folder)
			params = self.load_params_old_pcnn(self.load_folder)

		# construct train state
		state = self.get_train_state(params, model, tx)

		return params, model, tx, state, loss_, valids_, epoch_start

	def setup_epoch_permute(self):
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

			# # make training step and store metrics
			def loss_fn(params):
				return self.lossf(model, params, trajectories, Ts, Fs)

			# get rid of trajectories, flips and arrays
			grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
			(vals, eest), grads = grad_fn(state.params)
			
			state = state.apply_gradients(grads=grads)

			epoch_time = time.time() - start_time
			print('train_epoch: {} in {:0.2f} sec, loss: {:.4f}, no iterations: {}, energy est: {:.4f}'.format(epoch, epoch_time, vals, it, eest))	

			return state, vals, eest, epoch_time, it

		self.train_epoch = train_epoch

	def setup_epoch_split(self):
		raise NotImplementedError

	def setup_epoch_single(self):

		self.logRNf = get_logRN(self.config.J, self.config.g, self.config.L, self.config.dim, self.model) # signature (params, trajectory, times)

		def general_lossf(key, model, params, N_s, N_even, N_b, T, Tvar=False):

			if self.config.dim == 2:
				increment = 2*N_s
			elif self.config.dim == 1:
				increment = 2*N_s

			initial_size = (N_s - N_even)*3 + N_even*2 # initial trajectory length, evens start at 2 and odd start at 3 flips

			# storage for log RN
			logRN = jnp.zeros((N_b, 1))

			# randomly generate an initial state S0
			S0 = self.initialise_lattice(key)
			key, subkey = rnd.split(key)

			# randomly pick spins
			if self.config.dim == 2:
				spins = rnd.choice(key, jnp.arange(self.config.L**2), shape=(N_s, 1), replace=False) # between 0 and L^2 - 1
			elif self.config.dim == 1:
				spins = rnd.choice(key, jnp.arange(self.config.L), shape=(N_s, 1), replace=False) # between 0 and L - 1

			# randomly assign odd/even to spins
			evens = rnd.choice(subkey, jnp.arange(N_s), shape=(N_even, 1), replace=False) # between 0 and Neven
			key, subkey = rnd.split(key)

			# get repeats
			if self.config.dim == 2:
				repeats = jnp.zeros((1, self.config.L*2), dtype=jnp.int32)
				adds = jnp.zeros((1, self.config.L*2), dtype=jnp.int32)
			elif self.config.dim == 1:
				repeats = jnp.zeros((1, self.config.L), dtype=jnp.int32)
				adds = jnp.zeros((1, self.config.L), dtype=jnp.int32)

			# how many times an index is repeated, 3 times for odd 2 for even, this is doubled each iteration
			repeats = repeats.at[0, spins[:, 0]].set(3)
			adds = adds.at[0, spins[:, 0]].set(2)
			repeats = repeats.at[0, spins[evens[:, 0], 0]].add(-1)

			# all indices
			if self.config.dim == 2:
				indices = jnp.arange(self.config.L**2, dtype=jnp.int32)
			elif self.config.dim == 1:
				indices = jnp.arange(self.config.L, dtype=jnp.int32)

			# print(spins[:, 0])
			# print(evens[:, 0])

			# calculate log RNs for each trajectory in batch
			for i in range(N_b):
				N_it = initial_size + i*increment
				# times = rnd.exponential(key, shape=(N_it+1, 1)) # +1 because of additional last state
				times = rnd.exponential(key, shape=(N_it, 1))
				times = times/jnp.sum(times)*T # normalise times to T
				times = jax.lax.stop_gradient(times)
				flips = jnp.repeat(indices, i*adds + repeats, axis=0)
				flips = rnd.permutation(subkey, flips) # permute the flips
				# flips = jnp.append(flips, 0) # append last meaningless flip
				
				# construct trajectory
				trajectory = self.f2ps(S0, N_it+1, flips)
				trajectory = self.f2ps(S0, N_it, flips)

				# calculate log RN
				logRN = logRN.at[i].set(self.logRNf(params, trajectory, times.T, flips.T))

				key, subkey = rnd.split(key)

			# print(jax.lax.stop_gradient(logRN.T))

			return jnp.var(logRN), 0 # second return supposed to be eest

		self.loss_fn = lambda params, model, key: general_lossf(key, model, params, self.config.batch_Na, self.config.batch_Neven, self.config.batch_size, self.config.T, Tvar=self.config.batch_Tvar)

		def train_epoch(key, state, epoch, model, params):
			"""
			Training for a single epoch with single batch type
			
			Params:
			-------
			config -- the configuration dictionary
			key -- PRNGKey
			state -- training state
			epoch -- index of epoch

			"""
			start_time = time.time()

			grad_fn = jax.value_and_grad(self.loss_fn, has_aux=True, argnums=(0))
			(vals, eest), grads = grad_fn(state.params, model, key)
			
			state = state.apply_gradients(grads=grads)

			epoch_time = time.time() - start_time
			print('train_epoch: {} in {:0.2f} sec, loss: {:.4f}'.format(epoch, epoch_time, vals))	

			return state, vals, eest, epoch_time, 0

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
					init_val = jnp.ones((1, self.config.L, 1), jnp.float32)
				else:
					raise ValueError("Only one and two dimensions supported (dim=1 or dim=2)")

				# check correctness of params
				check_pcnn_validity(self.config.L, self.config.kernel_size, self.config.layers, self.config.dim)

				# periodic CNN object
				if self.config.dim == 2:
					pcnn = pCNN2d(conv=CircularConv2d, 
							act=nn.softplus,
			 				hid_channels=self.config.hid_channels, 
			 				out_channels=self.config.out_channels,
							K=self.config.kernel_size, 
							layers=self.config.layers, 
							strides=(1,1))
				
				elif self.config.dim == 1:
					pcnn = pCNN1d(conv=CircularConv1d, 
						act=nn.softplus,
		 				hid_channels=self.config.hid_channels, 
		 				out_channels=self.config.out_channels,
						K=self.config.kernel_size, 
						layers=self.config.layers, 
						strides=(1,))

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
					return rnd.choice(key, 2, shape=(1, L, 1))*(-2)+1
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

				def loop_fun(loop_state):
					S0, times, flips, rates, key, it, time, Tmax = loop_state

					# change current state
					if dim == 2:
						tau, s, key = step_gumbel(key, rates[0, :, :, 0])
						S0 = S0.at[0, s // l, s % l, 0].multiply(-1)
					elif dim == 1:
						tau, s, key = step_gumbel(key, rates[0, :, 0])
						S0 = S0.at[0, s % l, 0].multiply(-1)

					times = times.at[it, 0].set(tau)
					flips = flips.at[it, 0].set(s)

					# get rates
					rates = model.apply({'params': params['params']}, S0)
					it += 1
					time += tau
					return S0, times, flips, rates, key, it, time, Tmax

				def cond_fun(loop_state):
					S0, times, flips, rates, key, it, time, Tmax = loop_state
					return time < Tmax

				# loop 
				it = 0
				time = 0
				loop_state = S0, times, flips, rates, key, it, time, T
				S0, times, flips, rates, key, it, time, T = jax.lax.while_loop(cond_fun, loop_fun, loop_state)

				# fix the last time
				times = times.at[it-1, 0].add((T-time))
				
				return times, flips, it
			
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

				# create permutations
				Ts = Ts.at[i, 0:-1].set(rnd.permutation(key, Ts[0, 0:-1]))
				key, subkey = rnd.split(key)
				Fs = Fs.at[i, 0:-1].set(rnd.permutation(key, Fs[0, 0:-1]))

				# new key for new permutations
				key, subkey = rnd.split(key)
				
				# return trajectories, Ts, Fs, key
				return Ts, Fs, key

			# permute to get Nb trajectories
			loop_state = Ts, Fs, key
			Ts, Fs, key = jax.lax.fori_loop(1, Nb, loop_fun, loop_state)

			return Ts, Fs

		def f2p_single(S0, Nt, Fs, l, dim):
			"""
			Construct single trajectory
			"""
			trajectory = jnp.repeat(S0[jnp.newaxis, ...], 1, axis=0)
			trajectory = jnp.repeat(trajectory, Nt, axis=1)

			def loop_fun(i, loop_state): # meant to go from 1 to Nt
				trajectory = loop_state

				# at i-th time step, we multiply appropriate pixels with -1
				if dim == 2:
					F = jnp.ones((1, l, l, 1))
					F = F.at[:, Fs[i-1] // l, Fs[i-1] % l, :].set(-1)
					trajectory = trajectory.at[:, i, :, :, :].set(jnp.multiply(trajectory[:, i-1, :, :, :], F))
				elif dim == 1:
					F = jnp.ones((1, l, 1))
					F = F.at[:, Fs[i-1], :].set(-1)
					trajectory = trajectory.at[:, i, :, :].set(jnp.multiply(trajectory[:, i-1, :, :], F))

				return trajectory

			loop_state = trajectory
			trajectory = jax.lax.fori_loop(1, Nt, loop_fun, loop_state)

			return trajectory

		self.f2ps = jit(lambda S0, Nt, Fs: f2p_single(S0, Nt, Fs, self.config.L, self.config.dim), static_argnums=1)

		def f2p(S0, Nt, Nb, Fs, l, dim):
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
				if dim == 2:
					F = jnp.ones((Nb, l, l, 1))
					F = F.at[jnp.arange(Nb), Fs[:, i-1, 0] // l, Fs[:, i-1, 0] % l, :].set(-1)
					trajectories = trajectories.at[:, i, :, :, :].set(jnp.multiply(trajectories[:, i-1, :, :, :], F))
				elif dim == 1:
					F = jnp.ones((Nb, l, 1))
					F = F.at[jnp.arange(Nb), Fs[:, i-1, 0], :].set(-1)
					trajectories = trajectories.at[:, i, :, :].set(jnp.multiply(trajectories[:, i-1, :, :], F))

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
		# self.get_trajectory = ctr

		## permute to get a batch with fixed endpoints
		gbtc = lambda key, times, flips: get_btc(key, self.config.batch_size, times, flips)
		self.get_batch = jit(gbtc)
		# self.get_trajectory = jit(self.get_trajectory)

		## construct state trajectories from the flips
		ftt = lambda S0, Nt, Fs: f2p(S0, Nt, self.config.batch_size, Fs, self.config.L, self.config.dim)
		self.flip_to_trajectory = ftt

		## loss function
		if self.config.lattice_model == 'ising':
			self.lossf = get_Ieploss(self.config.J, self.config.g, self.config.L, self.config.dim)
		else:
			raise NotImplementedError

		key = rnd.PRNGKey(1991929)
		self.params, self.model = self.get_rate_parametrisation(key)

		# setup epoch step
		if self.config.batch_type == 'permute':
			self.setup_epoch_permute() 
		elif self.config.batch_type == 'split':
			self.setup_epoch_split()
		elif self.config.batch_type == 'construct':
			self.setup_epoch_single()

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
		path = 'data/' + self.experiment_name + '/' + self.output_dir + '/checkpoints{}'.format(epoch) 
		if not os.path.exists(path):
				os.mkdir(path)
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

	def load_params(self, dir_path):
		"""
		Loads rates from one of the experiment folders, 
		returns everything needed for importance sampling the model
		"""

		# load binary
		f = open(dir_path+'/params.txt', 'rb')
		b = f.read()
		
		key = jax.random.PRNGKey(0)
		params, model = self.get_rate_parametrisation(key)

		# get params
		ser = serialization.from_bytes(params, b)
		
		return ser

	def load_params_old_pcnn(self, dir_path):
		"""
		Loads rates from one of the experiment folders, 
		returns everything needed for importance sampling the model
		"""

		def gp(key, lattice_size, hid_channels, out_channels, kernel, layers):
			
			# initialisation size sample
			init_val = jnp.ones((1, lattice_size, lattice_size, 1), jnp.float32)
			
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

		# load binary
		f = open(dir_path+'/params.txt', 'rb')
		b = f.read()
		
		key = jax.random.PRNGKey(0)
		params, model = self.get_rate_parametrisation(key)
		ser = serialization.from_bytes(params, b)

		return ser

	def load_from_chp(self, dir_path):
		self.from_beg = False
		self.load_folder = dir_path