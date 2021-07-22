"""
Testing 2nd logRN term vectorisation

"""

import unittest

import jax
import jax.random as rnd
import jax.numpy as jnp
import jax.random as rnd

import tensorflow as tf

import sys
sys.path.append("../")

import ising_loss as il
import train_rates as tr

class SampleTest(unittest.TestCase):

	def setUp(self):
		# no gpu memory allocation by tf
		tf.config.experimental.set_visible_devices([], 'GPU')
		
		self.key = rnd.PRNGKey(0)
		key = self.key
		self.J, self.g = 1.0, 1.0
		self.l = 3

		self.S0 = jnp.array([[-1, -1, -1],
					   		 [-1, +1, -1],
				       		 [-1, -1, -1]])

		self.params, self.model = tr.get_parameterisation(key, 3, 5, 1, (3, 3), 5) # see train_rates
		
		# construct trajectories (4, 10)
		self.trajectories = jnp.zeros((4, 10, self.l, self.l, 1))

		self.flips = jnp.array([[3, 6, 3, 8, 1, 4, 5, 2, 2, 1],
							    [6, 4, 2, 7, 7, 7, 2, 1, 1, 0],
							    [1, 1, 0, 4, 8, 2, 1, 5, 3, 2],
							    [2, 3, 8, 6, 2, 3, 5, 5, 6, 7]])

		for i in range(4):
			for j in range(10):
				self.trajectories = self.trajectories.at[i, j, :, :, :].set(tr.initialise_lattice(key, self.l)[0, :, :, :])
				key, subkey = rnd.split(key)

	def test_batch_potential_shape(self):
		"""Testing output shape of vectorized over batch method"""
		# operates on batch of shape (Nb, L, L, 1), Nb = 4, L = 3, and returns (4,)
		key = self.key

		batch = self.trajectories[:, 0, :, :, :]
		f = self.flips[:, 0]

		rate_transition = il.get_rate_transition(self.J, self.g, self.l, self.model, self.params)

		lRN2 = rate_transition(batch, f)
		self.assertEqual((4, 1), jnp.shape(lRN2))
		
	def test_batch_potential(self):
		"""Testing equivalence between Single logrn 3rd term calls and vectorized version"""
		# operates on batch of shape (Nb, L, L, 1), Nb = 4, L = 3
		key = self.key

		batch = self.trajectories[:, 0, :, :, :] # test on first timestep
		f = self.flips[:, 0]
		rate_transition = il.get_rate_transition(self.J, self.g, self.l, self.model, self.params)
		lRN2 = rate_transition(batch, f)
		
		for i in range(4):
			# print(jnp.shape(batch[i, None, :, :, :]))
			lrn2 = il.rate_transition_single(batch[i, :, :, :], f[i], self.J, self.g, self.l, self.model, self.params)
			self.assertAlmostEqual(lrn2, lRN2[i], places=5)

	def test_full_batch_potential_shape(self):
		"""Test trajectory batch logrn 2nd term shape"""

		rate_transitions = il.get_rate_transitions(self.J, self.g, self.l, self.model, self.params)
		lRN2V = rate_transitions(self.trajectories, self.flips)
		# print(lRN2V)
		self.assertEqual((4, 10, 1), jnp.shape(lRN2V))

	def test_full_batch_potential(self):
		"""Test trajectory batch potential shape"""

		rate_transitions = il.get_rate_transitions(self.J, self.g, self.l, self.model, self.params)
		lRN2V = rate_transitions(self.trajectories, self.flips)
		# print(lRN2V)
		
		for i in range(4):
			for j in range(10):
				batch = self.trajectories[:, j, :, :, :]
				lrn2 = il.rate_transition_single(self.trajectories[i, j, :, :, :], self.flips[i, j], self.J, self.g, self.l, self.model, self.params)
				# print(lrn2)
				self.assertAlmostEqual(lRN2V[i, j], lrn2[0], places=5)

if __name__ == '__main__':
	unittest.main()