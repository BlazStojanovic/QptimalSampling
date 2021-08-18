"""
Testing ising potential term

"""

import unittest

import jax
import jax.random as rnd
import jax.numpy as jnp
import tensorflow as tf

import sys

sys.path.append("../")
import ising_loss as il
import train_rates

class SampleTest(unittest.TestCase):

	def setUp(self):
		# no gpu memory allocation by tf
		tf.config.experimental.set_visible_devices([], 'GPU')
		self.rates = jnp.ones((3, 3))
		self.S0 = jnp.array([[-1, -1, -1],
					   [-1, +1, -1],
				       [-1, -1, -1]])
		
		self.key = rnd.PRNGKey(0)
		self.J = 1.0
		self.g = 1.0

	def test_single_potential(self):
		"""Testing Single Ising potential for 3x3 lattice"""
		
		S = jnp.reshape(self.S0, (1, 3, 3, 1))

		# potential should be
		# -J*(2 + 1 + 2 + 1 -2 + 1 + 2 + 1 + 2) -g*(9) = -19

		self.assertEqual(il.ising_potential_single(S, self.J, self.g), -19)

	def test_batch_potential_shape(self):
		"""Testing output shape of vectorized over batch method"""
		# operates on batch of shape (Nb, L, L, 1), Nb = 4, L = 3, and returns (4, 1)
		key = self.key

		batch = jnp.zeros((4, 1, 3, 3, 1))

		for i in range(4):
			batch = batch.at[i, :, :, :, :].set(train_rates.initialise_lattice(key, 3))
			key, subkey = rnd.split(key)

		V = il.ising_potential(batch, self.J, self.g)
		self.assertEqual((4, 1), jnp.shape(V))
		

	def test_batch_potential(self):
		"""Testing equivalence between Single Ising potential calls and vectorized version"""
		# operates on batch of shape (Nb, L, L, 1), Nb = 4, L = 3
		key = self.key

		batch = jnp.zeros((4, 3, 3, 1))

		for i in range(4):
			batch = batch.at[i, :, :, :].set(train_rates.initialise_lattice(key, 3)[0, :, :, :])
			key, subkey = rnd.split(key)

		V = il.ising_potential(batch, self.J, self.g)
		# print(V)
		for i in range(4):
			# print(batch[i, :, :, 0])
			vs = il.ising_potential_single(batch[i, :, :, :], self.J, self.g)
			# print(vs)
			self.assertEqual(vs, V[i])

	def test_full_batch_potential_shape(self):
		"""Test trajectory batch potential shape"""
		key = self.key
		batch = jnp.zeros((4, 10, 3, 3, 1))

		for i in range(4):
			for j in range(10):
				batch = batch.at[i, j, :, :, :].set(train_rates.initialise_lattice(key, 3)[0, :, :, :])
				key, subkey = rnd.split(key)

		Vs = il.ising_potentialV(batch, self.J, self.g)
		# print(Vs)
		self.assertEqual((4, 10,), jnp.shape(Vs))

	def test_full_batch_potential(self):
		"""Test trajectory batch potential shape"""
		key = self.key
		batch = jnp.zeros((4, 10, 3, 3, 1))

		for i in range(4):
			for j in range(10):
				batch = batch.at[i, j, :, :, :].set(train_rates.initialise_lattice(key, 3)[0, :, :, :])
				key, subkey = rnd.split(key)

		Vs = il.ising_potentialV(batch, self.J, self.g)
		
		for i in range(4):
			for j in range(10):
				# print(batch[i, j, :, :, :])
				vs = il.ising_potential_single(batch[i, j, :, :, :], self.J, self.g)
				# print(vs)
				self.assertEqual(Vs[i, j], vs)

if __name__ == '__main__':
	unittest.main()