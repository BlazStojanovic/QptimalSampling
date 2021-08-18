"""
Group equivariant CNN that respects p4 symmetry and uses periodic padding.
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
