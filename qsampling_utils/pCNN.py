"""
CNN that respects periodic boudary conditions, 
and one in which all of the inputs are correlated with each other
"""

from flax import linen as nn

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from functools import partial
from typing import Any, Callable, Tuple, Union, Iterable
from pprint import pprint

ModuleDef = Any

class CircularConv2d(nn.Module):
	"""Circular convolution, equivalent to padding_mode='circular' in Torch"""
	channels: int
	K: Union[int, Iterable[int]]
	strides: Tuple[int, int] = (1,1)
	@nn.compact
	def __call__(self, x):
		padding = [(0, 0)] + [(k//2, k//2) for k in self.K] + [(0, 0)]
		x = jnp.pad(x, padding, mode='wrap')
		return nn.Conv(self.channels, 
					   self.K,
					   strides=self.strides, 
					   padding='VALID')(x)

class CircularConv1d(nn.Module):
	"""Circular convolution, equivalent to padding_mode='circular' in Torch"""
	channels: int
	K: Union[int, Iterable[int]]
	strides: Tuple[int] = (1)
	@nn.compact
	def __call__(self, x):
		padding = [(0, 0), (self.K//2, self.K//2), (0, 0)]
		x = jnp.pad(x, padding, mode='wrap')
		return nn.Conv(self.channels, 
					   self.K,
					   strides=self.strides, 
					   padding='VALID')(x)

class PeriodicBlock2d(nn.Module):
	"""Single block of the pCNN"""
	conv: ModuleDef
	act: Callable
	channels: int
	K: Union[int, Iterable[int]]
	strides: Tuple[int, int] = (1, 1)
	@nn.compact
	def __call__(self, x,):
		y = self.conv(self.channels, self.K, strides=self.strides)(x)
		return self.act(y)

class PeriodicBlock1d(nn.Module):
	"""Single block of the pCNN"""
	conv: ModuleDef
	act: Callable
	channels: int
	K: Union[int, Iterable[int]]
	strides: Tuple[int] = (1)

	@nn.compact
	def __call__(self, x,):
		y = self.conv(self.channels, self.K, strides=self.strides)(x)
		return self.act(y)

class pCNN2d(nn.Module):
	"""periodic all-pixle-correlated CNN"""
	conv: ModuleDef
	act: Callable
	hid_channels: int
	out_channels: int
	K: Union[int, Iterable[int]]
	layers: int # TODO add check, because this has to be a certain fixed value
	strides: Tuple[int, int] = (1,1)

	@nn.compact
	def __call__(self, x):
		for i in range(0, self.layers):
			x = PeriodicBlock2d(conv=self.conv, 
						   act=self.act, 
						   channels=self.hid_channels,
						   K=self.K,  
						   strides=self.strides)(x)

		return PeriodicBlock2d(conv=self.conv, 
						   act=nn.softplus,
						   channels=self.out_channels,
						   K=self.K,  
						   strides=self.strides)(x)

class pCNN1d(nn.Module):
	"""periodic all-pixle-correlated CNN"""
	conv: ModuleDef
	act: Callable
	hid_channels: int
	out_channels: int
	K: int
	layers: int # TODO add check, because this has to be a certain fixed value
	strides: Tuple[int] = (1)

	@nn.compact
	def __call__(self, x):
		for i in range(0, self.layers):
			x = PeriodicBlock1d(conv=self.conv, 
						   act=self.act, 
						   channels=self.hid_channels,
						   K=self.K,  
						   strides=self.strides)(x)

		return PeriodicBlock1d(conv=self.conv, 
						   act=nn.softplus,
						   channels=self.out_channels,
						   K=self.K,  
						   strides=self.strides)(x)


def check_pcnn_validity(lattice_size, K, layers, dim):
	"""
	Checks if the hyperparameters of the pCNN are such that 
	each input pixel is correlated with every other input pixel
	"""

	if dim == 2:
		# we assume that the kernel is square for now
		assert K[0] == K[1], "The kernel must be square!"

		if K[0] % 2 != 0:
			min_num_layers = (lattice_size - 1) // (K[0] - 1)
		else:
			raise ValueError("Kernel size must be odd"
				+ "L-1 must be multiple of K-1, error because L = {} and K = {}".format(lattice_size, K))
		# check depth
		if layers <= min_num_layers:
			raise ValueError("Minimum number of layers should be atleast {}, but is {}.".format(min_num_layers+1, layers))
	elif dim == 1:
		if K % 2 != 0:
				min_num_layers = (lattice_size - 1) // (K - 1)
		else:
			raise ValueError("Kernel size must be odd"
				+ "L-1 must be multiple of K-1, error because L = {} and K = {}".format(lattice_size, K))
		# check depth
		if layers <= min_num_layers:
			raise ValueError("Minimum number of layers should be atleast {}, but is {}.".format(min_num_layers, layers))
	else:
		raise NotImplementedError

if __name__ == '__main__':
	"""
	Example of basic usage
	"""

	# K = (3, 3) # must be odd for this to work
	# out_channels = 1
	# hid_channels = 2

	# key = random.PRNGKey(123456789)
	# pcnn = pCNN2d(conv=CircularConv2d,
	# 			act=nn.relu,
 # 				hid_channels=hid_channels, 
 # 				out_channels=out_channels,
	# 			K=K, 
	# 			layers=5, 
	# 			strides=(1,1))
	
	# variables = pcnn.init({'params':key}, jnp.zeros((1,7,7,1)))

	# # pprint(variables['params'])
	# # params = CircularConv(channels, K, strides=(1,1)).init(key, jnp.zeros((1,7,7,1)))

	# # x = random.normal(key, (1,7,7,1))
	# x = random.choice(key, 2, shape=(1, 7, 7, 1))*(-2)+1
	# print(x)
	# out = pcnn.apply({'params': variables['params']}, x)
	# pprint(jnp.shape(out))
	# pprint(out)


	K = 3 # must be odd for this to work
	out_channels = 1
	hid_channels = 15

	key = random.PRNGKey(123456789)
	pcnn = pCNN1d(conv=CircularConv1d,
				act=nn.relu,
 				hid_channels=hid_channels, 
 				out_channels=out_channels,
				K=K, 
				layers=5, 
				strides=(1,))
	
	variables = pcnn.init({'params':key}, jnp.zeros((1,7,1)))

	# pprint(variables['params'])
	# params = CircularConv(channels, K, strides=(1,1)).init(key, jnp.zeros((1,7,7,1)))

	# x = random.normal(key, (1,7,7,1))
	x = random.choice(key, 2, shape=(1, 7, 1))*(-2)+1
	print(x)
	out = pcnn.apply({'params': variables['params']}, x)
	print(jnp.shape(out))
	print(out)