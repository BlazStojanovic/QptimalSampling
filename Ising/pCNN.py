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

class CircularConv(nn.Module):
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

class PeriodicBlock(nn.Module):
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

class pCNN(nn.Module):
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
			x = PeriodicBlock(conv=self.conv, 
						   act=self.act, 
						   channels=self.hid_channels,
						   K=self.K,  
						   strides=self.strides)(x)

		return PeriodicBlock(conv=self.conv, 
						   act=self.act, 
						   channels=self.out_channels,
						   K=self.K,  
						   strides=self.strides)(x)


# TODO, this is just bad design. But should be enough for now. Fix this when you have time.
def check_pCNN(lattice_size, K, layers):
	""" Check if the pCNN correlates each pixel with each other """
	if (lattice_size - 1) % (K - 1) == 0 and K % 2 != 0:
            num_inner_layers = (lattice_size - 1) // (K - 1)
        else:
            raise ValueError("Kernel size K and lattice size should be odd"
                             + "L-1 must be a multiple of K-1")

    if layers < num_inner_layers:
    	raise ValueError("The number of layers {} is to small, should be at least {}".format(layers, num_inner_layers))


if __name__ == '__main__':
	"""
	Example of basic usage
	"""

	K = (3, 3) # must be odd for this to work
	out_channels = 1
	hid_channels = 3

	key = random.PRNGKey(123456789)
	pcnn = pCNN(conv=CircularConv, 
				act=nn.relu,
 				hid_channels=hid_channels, 
 				out_channels=out_channels,
				K=K, 
				layers=5, 
				strides=(1,1))
	
	variables = pcnn.init({'params':key}, jnp.zeros((1,7,7,1)))

	pprint(variables['params'])
	# params = CircularConv(channels, K, strides=(1,1)).init(key, jnp.zeros((1,7,7,1)))

	x = random.normal(key, (1,7,7,1))
	out = pcnn.apply({'params': variables['params']}, x)
	pprint(jnp.shape(out))
