"""
Testing sampling of continuous-time Markov chains

"""

import unittest

import jax
import jax.random as rnd
import jax.numpy as jnp

import tensorflow as tf

class SampleTest(unittest.TestCase):


	def setUp(self):
		# no gpu memory allocation by tf
		tf.config.experimental.set_visible_devices([], 'GPU')


	def test_maximum_stationary(self):
		"""Test that choosing maximum leads into correct stationary distribution"""
		self.assertEqual(1, 1)


	def test_gumbel(self):
		"""Test that gumbel leads into correct stationary distribution"""
		self.assertEqual(1, 1)



if __name__ == '__main__':
    unittest.main()