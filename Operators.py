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
		sigz = lambda S: jnp.abs(jnp.sum(S, axis=(-1, -2))/jnp.sum(jnp.ones_like(S), axis=(-1, -2)))
	elif dim == 2:
		sigz = lambda S: jnp.abs(jnp.sum(S, axis=(-1, -2, -3))/jnp.sum(jnp.ones_like(S), axis=(-1, -2, -3)))
	return sigz

def get_sigma_x_single(dim, lattice_model, model, params):
	if lattice_mode == 'ising':
		if dim == 1:
			# sigx = lambda S: jnp.sum(model.apply({'params': params['params']}, S))/jnp.sum(jnp.ones_like(S))
		elif dim == 2:
			sigx = lambda S: jnp.sum(model.apply({'params': params['params']}, S))/jnp.sum(jnp.ones_like(S))
	elif lattice_model == 'xy':
		raise NotImplementedError
	return sigx

def get_energy_Ising_single(dim, lattice_model, model, params, inps):
	if lattice_model == 'ising':
		pass

