"""
Operators.
"""


# Jax imports
import jax
import jax.numpy as jnp
import jax.random as rnd
from jax import jit, vmap, pmap
import optax


def sigma_z():
	pass

def sigma_x():
	pass

def sigma_y():
	pass

if __name__ == '__main__':
	s = jnp.array([[-1, 1, 1],
				   [1, -1, 1],
				   [1, -1, 1]])