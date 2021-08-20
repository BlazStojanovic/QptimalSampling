"""
Operators.
"""

# Jax imports
import jax
import jax.numpy as jnp
import jax.random as rnd
from jax import jit, vmap, pmap
import optax

def get_sigz_single(dim, lattice_model):
	if dim == 1:
		sigz = lambda S: (jnp.abs(jnp.sum((S), axis=(-1, -2))/jnp.sum(jnp.ones_like(S), axis=(-1, -2))))
	elif dim == 2:
		sigz = lambda S: jnp.abs(jnp.sum((S), axis=(-1, -2, -3))/jnp.sum(jnp.ones_like(S), axis=(-1, -2, -3)))
	return sigz

def sigz_single(S, f, J, g, lattice_size, model, params, dim):
	if dim == 2:
		rates = model.apply({'params': params['params']}, S[None, :, :, :]) # get current rates
		rate = rates[f//lattice_size, f%lattice_size, 0] # the rate corresponding to going to next state

	elif dim == 1:
		rates = model.apply({'params': params['params']}, S[None, :, :]) # get current rates
		rate = rates[f%lattice_size, 0] # the rate corresponding to going to next state

	return rate

def get_energy_Ising_single(dim, lattice_model, model, params, inps):
	pass
