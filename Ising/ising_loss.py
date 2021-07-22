"""
Calculating endpoint loss for the 2D Ising model. Along with helper functions
"""

import jax.numpy as jnp
from jax import jit, vmap
import jax

from flax.linen import jit as fjit

@jit
def ising_potential_single(S, J, g):
	"""
	Ising potential

	Params:
	-------
	S -- state (1, l, l, 1)
	J -- interaction coefficient
	g -- coupling coefficient
	
	"""

	z = S
	zu = jnp.roll(z, 1, axis=-3)
	zr = jnp.roll(z, 1, axis=-2)

	# first term
	potential = z*zu + z*zr
	potential = -J*jnp.sum(potential, axis=(-1, -2, -3))

	#second term
	potential -= g*jnp.sum(jnp.ones(jnp.shape(S)), axis=(-1, -2, -3))

	return potential

# vectorize the ising potential
ising_potential = vmap(ising_potential_single, in_axes=(0, None, None), out_axes=(0))
ising_potentialV = vmap(ising_potential, in_axes=(0, None, None), out_axes=(0))

def passive_difference_single(S, J, g, model, params):
	"""
	Returns the sum of the difference between the passive and variational rates for
	the current state. Necessary for the second term of the endpoint loss.

	Params:
	-------
	S -- state
	J -- interaction coefficient
	g -- coupling coefficient
	model -- variational approximation of the rates
	params -- params of the variational approximation

	Returns:
	--------
	rate_difference
	"""

	rates = jnp.sum(model.apply({'params': params['params']}, S), axis=(-1, -2, -3))
	passive_rates = g*jnp.sum(jnp.ones(jnp.shape(S)), axis=(-1, -2, -3))
	
	return passive_rates - rates

passive_difference = vmap(passive_difference_single, in_axes=(0, None, None, None, None), out_axes=(0))

def rate_transition_single(S, f, J, g, lattice_size, model, params):
	"""
	Returns a term for a single n of the third loss term (transition rate between subsequent states)
	Params:
	-------
	S -- state
	J -- interaction coefficient
	g -- coupling coefficient
	model -- variational approximation of the rates
	params -- params of the variational approximation

	Returns:
	--------
	diff -- single term of the third loss
	"""

	# TODO this is just for the square lattice atm

	# from variational rates
	# print(f)
	# print(S)

	rates = model.apply({'params': params['params']}, S[None, :, :, :]) # get current rates
	# rates = jnp.ones(jnp.shape(S))
	rate = rates[f//lattice_size, f%lattice_size, 0] # the rate corresponding to going to next state

	# from passive rates
	passive = g 

	return jnp.log(rate/passive)

def get_rate_transition(J, g, l, model, params):
	def nf(Ss, Fs):
		return rate_transition_single(Ss, Fs, J, g, l, model, params)
	return vmap(nf, in_axes=(0, 0), out_axes=(0))

def get_rate_transitions(J, g, l, model, params):
	nf = get_rate_transition(J, g, l, model, params)
	return vmap(nf, in_axes=(1, 1), out_axes=1)

def ising_endpoint_loss(trajectories, Ts, Fs, model, params, J, g, lattice_size):
	"""
	Loss for fixed trajectory endpoints

	Params:
	-------
	trajectories -- Batch of trajectories with shape (Nb, Nt, l, l, 1) where 
	Ts is the no. of time steps, Nb is the batch size and l is the lattice size
	times -- Batch of holding times with shape (Nb, Nt, 1)
	Fs -- Batch of spin flips with shape (Nb, Nt, 1)
	model, params -- the variational parameterisation alongside the optimizable params

	Returns:
	loss -- logRN loss of the batch
	"""

	# construct function for each term
	rate_transitions = get_rate_transitions(J, g, lattice_size, model, params)

	# print("Doublecheck that (Nb, Nt, L, L, 1), ", jnp.shape(trajectories))
	# print("Doublecheck that (Nb, Nt, 1), ", jnp.shape(Ts))

	logRN = 0.0
	V = ising_potentialV(trajectories, J, g)
	Vt = jnp.sum(jnp.multiply(Ts.squeeze(), V.squeeze()), axis=1)

	T1 = passive_difference(trajectories, J, g, model, params)
	T1t = jnp.sum(jnp.multiply(Ts.squeeze(), T1.squeeze()), axis=1)

	T2 = rate_transitions(trajectories, Fs)
	T2s = jnp.sum(T2, axis=1).squeeze()

	# print(jax.lax.stop_gradient(T2s))
	# print(jnp.shape(T2))
	# print(jnp.shape(T2s))
	# print(jnp.shape(T1t))
	# print(jnp.shape(Vt))

	logRN = T1t + T2s #+ Vt
	Eest = Vt[0] + T1t[0] + T2s[0]

	# # print("Times after sampling (should equal T), ", jnp.sum(Ts[:, :, 0], axis=1))
	# print("Length of trajectory, ", jnp.sum(Ts[:, :, 0], axis=1))
	print("logRN + E0 of the first trajectory (should get closer and closer to E0), ", jax.lax.stop_gradient(Eest)/jnp.sum(Ts[0, :, 0]))
	# print("First term contribution to logRN, ", jax.lax.stop_gradient(Vt))
	# print("Second term contribution to logRN, ", jax.lax.stop_gradient(T1t))
	# print("Third term contribution to logRN, ", jax.lax.stop_gradient(T2.T))
	# print("logRN of each permutation", jax.lax.stop_gradient(logRN))

	return jnp.var(logRN, ddof=1)