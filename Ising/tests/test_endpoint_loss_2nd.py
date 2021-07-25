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

		self.S0 = jnp.array([[-1, -1, -1],
					   		 [-1, +1, -1],
				       		 [-1, -1, -1]])

		self.params, self.model = tr.get_parameterisation(key, 3, 5, 1, (3, 3), 5) # see train_rates
		
		# construct trajectories (4, 10)
		self.trajectories = jnp.zeros((4, 10, 3, 3, 1))

		for i in range(4):
			for j in range(10):
				self.trajectories = self.trajectories.at[i, j, :, :, :].set(tr.initialise_lattice(key, 3)[0, :, :, :])
				key, subkey = rnd.split(key)

	def test_batch_potential_shape(self):
		"""Testing output shape of vectorized over batch method"""
		# operates on batch of shape (Nb, L, L, 1), Nb = 4, L = 3, and returns (4, 1)
		key = self.key

		batch = self.trajectories[:, 0, :, :, :] # test on first timestep
		# print(jnp.shape(batch))

		lRN1 = il.passive_difference_single(batch, self.J, self.g, self.model, self.params)

		self.assertEqual((4,), jnp.shape(lRN1))
		
	def test_batch_potential(self):
		"""Testing equivalence between Single logrn 2nd term calls and vectorized version"""
		# operates on batch of shape (Nb, L, L, 1), Nb = 4, L = 3
		key = self.key

		batch = self.trajectories[:, 0, :, :, :] # test on first timestep
		lRN1 = il.passive_difference_single(batch, self.J, self.g, self.model, self.params)

		# print(lRN1)

		for i in range(4):
			# print(jnp.shape(batch[i, None, :, :, :]))
			lrn1 = il.passive_difference_single(batch[i, None, :, :, :], self.J, self.g, self.model, self.params)
			# print(lrn1)
			self.assertAlmostEqual(lrn1, lRN1[i], places=5)

	def test_full_batch_potential_shape(self):
		"""Test trajectory batch logrn 2nd term shape"""
		lRN1V = il.passive_difference(self.trajectories, self.J, self.g, self.model, self.params) 
		# print(lRN1V)
		self.assertEqual((4, 10,), jnp.shape(lRN1V))

	def test_full_batch_potential(self):
		"""Test trajectory batch potential shape"""
		lRN1V = il.passive_difference(self.trajectories, self.J, self.g, self.model, self.params) 

		
		for i in range(4):
			for j in range(10):
				batch = self.trajectories[:, j, :, :, :]
				lrn1 = il.passive_difference_single(batch[i, None, :, :, :], self.J, self.g, self.model, self.params)
				# print(lrn1)
				self.assertAlmostEqual(lRN1V[i, j], lrn1[0], places=5)

if __name__ == '__main__':
	unittest.main()