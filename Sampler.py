"""
Sampler class, a class for sampling from the parameterised rates
"""
import os
import sys
from dataclasses import dataclass
import ml_collections

import Trainer as tr
import jax.numpy as jnp
import jax

import flax.linen as nn

from qsampling_utils.sampl_utils import step_max, step_gumbel
from qsampling_utils.pCNN import *

import numpy as np
import matplotlib.pyplot as plt
import itertools

@dataclass
class Sampler:
	experiment_name: str
	input_dir: str
	output_dir: str
	config: ml_collections.ConfigDict
	
	def load_sampler(self):

		out = "../data/" + self.experiment_name + "/" + self.input_dir
		bench = tr.Trainer(experiment_name=self.experiment_name,
				 config=self.config,
				 output_dir=self.input_dir)

		bench.setup_experiment()
		bench.load_from_chp(out)
		params, model, tx, state, loss_, valids_, epoch_start = bench.init_training(prngn=111)

		self.params = params
		self.model = model
		self.training_loss = loss_
		self.validations = valids_

		# reuse trainer functions
		self.initialise_lattice = bench.initialise_lattice
		self.flip_to_trajectory = bench.flip_to_trajectory

		# setup traj functions
		fcont = lambda key, S0, no_steps: self.const_traj_continuous(key, self.model, self.params, S0, self.config.lattice_model, self.config.dim, no_steps, self.config.L)
		self.ctcont = fcont
		fdisc = lambda key, S0, no_steps: self.const_traj_discrete(key, self.model, self.params, S0, self.config.lattice_model, self.config.dim, no_steps, self.config.L)
		self.ctdisc = fdisc

	def const_traj_discrete(self, key, model, params, S0, latt_model, dim, Nlen, l):
		"""
		Obtain a single trajectory parameterised by the rates

		"""
		if latt_model == 'ising':
			times = jnp.ones((Nlen, 1)) # times are set to 1 for each state visited
			flips = jnp.zeros((Nlen, 1), dtype=jnp.int32)

			def metropolis_step(key, S0):
				# takes model and returns flip and next state
				G0 = model.apply({'params': params['params']}, S0)
				lamb0 = jnp.sum(G0)

				# propose flip with initial rates
				key, subkey = jax.random.split(key)
				if dim == 1:
					f1 = jax.random.categorical(key, jnp.log(jnp.ravel(G0[0, :, 0]/lamb0, order='C')))
				elif dim == 2:
					f1 = jax.random.categorical(key, jnp.log(jnp.ravel(G0[0, :, :, 0]/lamb0, order='C')))

				# get new state 
				if dim == 2:
					S1 = S0.at[0, f1 // l, f1 % l, 0].multiply(-1)
				elif dim == 1:
					S1 = S0.at[0, f1 % l, 0].multiply(-1)

				G1 = model.apply({'params': params['params']}, S1)	
				lamb1 = jnp.sum(G1)

				# get w ratio, w = Ts1->s0/Ts0->s1 Gs0->s1
				if dim == 2:
					w = lamb0/lamb1
				elif dim == 1:
					w = lamb0/lamb1
				
				r = jax.random.uniform(subkey, dtype=jnp.float32) # uniform random between 0 and 1
				# determine new state
				trfun = lambda cp: cp[1]
				fafun = lambda cp: cp[0]
				S0 = jax.lax.cond(r < w, trfun, fafun, operand=(S0, S1)) # either choose or accept 
				
				# determine flip stored
				trfun = lambda s: s

				if dim == 2:
					fafun = lambda s: l**2 + 1 # out of bounds trick, this is ugly but works with the current flip to trajectory functions, as jax.lax.at[i].set wont update oob indices!
				elif dim == 1:
					fafun = lambda s: l+10000

				f1 = jax.lax.cond(r < w, trfun, fafun, operand=f1)

				key, subkey = jax.random.split(subkey)

				return S0, f1, key

			def loop_fun(it, loop_state):
				S0, flips, key = loop_state

				# change current state
				S0, s, key = metropolis_step(key, S0)
				flips = flips.at[it, 0].set(s)

				return S0, flips, key

			# loop 
			loop_state = S0, flips, key
			S0, flips, key = jax.lax.fori_loop(0, Nlen, loop_fun, loop_state)
			
			return times, flips
		
		elif self.config.lattice_model == 'heisenberg':
			raise NotImplementedError


	def const_traj_continuous(self, key, model, params, S0, latt_model, dim, Nlen, l):
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
				times = jnp.zeros((Nlen, 1))
				flips = jnp.zeros((Nlen, 1), dtype=jnp.int32)

				# initial rates
				rates = model.apply({'params': params['params']}, S0)

				def loop_fun(it, loop_state):
					S0, times, flips, rates, key, time = loop_state

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
					return S0, times, flips, rates, key, time

				# loop 
				time = 0
				loop_state = S0, times, flips, rates, key, time
				S0, times, flips, rates, key, time = jax.lax.fori_loop(0, Nlen, loop_fun, loop_state)
				
				return times, flips
			
			elif self.config.lattice_model == 'heisenberg':
				raise NotImplementedError
		

	def f2t(self):
		trajectories = jnp.repeat(self.initials, self.Nt, axis=1)
		def loop_fun(i, loop_state): # meant to go from 1 to Nt
			trajectories = loop_state

			l = self.config.L
			# at i-th time step, we multiply appropriate pixels with -1
			if self.config.dim == 2:
				F = jnp.ones((self.Nc, l, l, 1))
				F = F.at[jnp.arange(self.Nc), self.Fs[:, i-1] // l, self.Fs[:, i-1] % l, :].set(-1)
				trajectories = trajectories.at[:, i, :, :, :].set(jnp.multiply(trajectories[:, i-1, :, :, :], F))
			elif self.config.dim == 1:
				F = jnp.ones((self.Nc, l, 1))
				F = F.at[jnp.arange(self.Nc), self.Fs[:, i-1], :].set(-1)
				trajectories = trajectories.at[:, i, :, :].set(jnp.multiply(trajectories[:, i-1, :, :], F))

			return trajectories


		loop_state = trajectories
		trajectories = jax.lax.fori_loop(1, self.Nt, loop_fun, loop_state)

		return trajectories

	def initialise_chains(self, no_chains=1, no_steps=int(10**4)):
		if self.config.dim == 2:
			self.initials = jnp.zeros((no_chains, 1, self.config.L, self.config.L, 1))
		elif self.config.dim == 1:
			self.initials = jnp.zeros((no_chains, 1, self.config.L, 1))

		# setup sampling functions
		gcont = lambda key, S0: self.ctcont(key, S0, no_steps)
		# self.gct = nn.jit(gcont)
		self.gct = gcont

		gdisc = lambda key, S0: self.ctdisc(key, S0, no_steps)
		# self.gdt = nn.jit(gdisc)
		self.gdt = gdisc

		self.Nc = no_chains
		self.Nt = no_steps
		
	def generate_samples(self, key, method='cont'):
		self.Ts = jnp.zeros((self.Nc, self.Nt), dtype=jnp.float32)
		self.Fs = jnp.zeros((self.Nc, self.Nt), dtype=jnp.int32)

		if method=='cont':
			for i in range(np.shape(self.Ts)[0]):
				key, subkey = jax.random.split(key)
				S0 = self.initialise_lattice(key)
				ts, fs = self.gct(subkey, S0)
				self.initials = self.initials.at[i, None, :, :, :].set(S0)
				self.Ts = self.Ts.at[i, :].set(ts[:, 0])
				self.Fs = self.Fs.at[i, :].set(fs[:, 0])

		elif method=='disc':
			for i in range(np.shape(self.Ts)[0]):
				key, subkey = jax.random.split(key)
				S0 = self.initialise_lattice(key)
				ts, fs = self.gdt(subkey, S0)
				self.initials = self.initials.at[i, None, :, :, :].set(S0)
				self.Ts = self.Ts.at[i, :].set(ts[:, 0])
				self.Fs = self.Fs.at[i, :].set(fs[:, 0])

		return self.Ts, self.Fs, S0
			
	def time_flip_statistics(self):
		""""""
		print("Sampler summary, showing distributions of tau and flips")
		x = np.linspace(0, 2, 200)
		lam = self.config.L**(self.config.dim)*self.config.g
		for i, t in enumerate(self.Ts):
			fig, ax = plt.subplots(1, 2)
			ax[0].set_title("Time sum: {}".format(jnp.sum(self.Ts[i])))
			ax[0].hist(self.Ts[i], 50, density=True, facecolor='b', alpha=0.75, label='tau distribution {}'.format(i))
			ax[0].plot(x, lam*np.exp(-lam*x), color='black', linestyle='--', linewidth=2)
			ax[1].hist(self.Fs[i], 50, density=True, facecolor='r', alpha=0.75, label='flips distribution {}'.format(i))
			plt.show()

	def setup_experiment_folder(self):
		# default path
		path = '../data/' + self.experiment_name + '/' + self.input_dir

		# check if folder exists
		try:	
			if not os.path.exists(path):
				os.makedirs(path)
			if not os.path.exists(path + self.output_dir):
				os.mkdir(path + self.output_dir)
		except OSError:
			print('creation of out directory failed')
			exit(0)

	def visualise_rates(self, plot=False):
		l = self.config.L
		if self.config.dim != 1:
			raise NotImplementedError("Warning visualising rates only implemented for 1D systems!")
		
		print("The model has {} states, be careful visualising rates!".format(2**(l)))

		# get all permutations
		permutations = np.zeros((2**l, l))
		inp = [0]*l
		c = 1
		for i in range(l):
			inp[i] = 1
			states = list(set(map(tuple, list(itertools.permutations(inp)))))
			nperm = len(states)
			permutations[c:c+nperm, :] = states
			c += nperm

		permutations = 2*permutations-1
		batch = permutations[:, :, None]
		rates = self.model.apply({'params': self.params['params']}, batch)

		if plot is True:
			# fig 
			fig, ax = plt.subplots(2, 1)
			fig.set_size_inches((12, 3))

			ax[0].imshow(permutations.T, cmap='Greys')
			ax[0].xaxis.tick_top()
			ax[1].imshow((rates.squeeze()).T, cmap='Blues')
			ax[0].sharex(ax[1])
			plt.show()

		return permutations, rates
