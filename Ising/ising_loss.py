"""
Calculating endpoint loss for the 2D Ising model. Along with helper functions
"""

import jax.numpy as jnp
from jax import jit, vmap
import jax

from flax.linen import jit as fjit

import logging

def ising_potential_single(S, J, g, dim):
	"""
	Ising potential

	Params:
	-------
	S -- state (1, l, l, 1) or (1, l, 1)
	J -- interaction coefficient
	g -- coupling coefficient
	
	"""
	z = S

	if dim == 2:
		zu = jnp.roll(z, 1, axis=-3)
		zr = jnp.roll(z, 1, axis=-2)

		# first term
		potential = z*zu + z*zr
		potential = -J*jnp.sum(potential, axis=(-1, -2, -3)) # (1, l, l, 1) are (-4, -3, -2, -1)
		potential -= g*jnp.sum(jnp.ones(jnp.shape(S)), axis=(-1, -2, -3))

	elif dim == 1:
		zr = jnp.roll(z, 1, axis=-2) # (1, l, 1) are (-3, -2, -1)
		potential = z*zr
		potential =  -J*jnp.sum(potential, axis=(-1, -2))
		potential -= g*jnp.sum(jnp.ones(jnp.shape(S)), axis=(-1, -2))

	return potential

# vectorize the ising potential
ising_potential = vmap(ising_potential_single, in_axes=(0, None, None, None), out_axes=(0))
ising_potentialV = vmap(ising_potential, in_axes=(0, None, None, None), out_axes=(0))

def passive_difference_single(S, J, g, model, params, dim):
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
	if dim == 2:
		rates = jnp.sum(model.apply({'params': params['params']}, S), axis=(-1, -2, -3))
		passive_rates = g*jnp.sum(jnp.ones(jnp.shape(S)), axis=(-1, -2, -3))
	elif dim == 1:
		rates = jnp.sum(model.apply({'params': params['params']}, S), axis=(-1, -2))
		passive_rates = g*jnp.sum(jnp.ones(jnp.shape(S)), axis=(-1, -2))
	
	return passive_rates - rates

passive_difference = vmap(passive_difference_single, in_axes=(0, None, None, None, None, None), out_axes=(0))

def rate_transition_single(S, f, J, g, lattice_size, model, params, dim):
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
	if dim == 2:
		rates = model.apply({'params': params['params']}, S[None, :, :, :]) # get current rates
		# rates = jnp.ones(jnp.shape(S))
		rate = rates[f//lattice_size, f%lattice_size, 0] # the rate corresponding to going to next state

		# print(rate)

		# from passive rates
		passive = g
		# print(jnp.log(rate/passive))

	elif dim == 1:
		rates = model.apply({'params': params['params']}, S[None, :, :]) # get current rates
		# rates = jnp.ones(jnp.shape(S))
		rate = rates[f%lattice_size, 0] # the rate corresponding to going to next state

		# from passive rates
		passive = g

	return jnp.log(rate/passive)

def get_rate_transition(J, g, l, model, params, dim):
	def nf(Ss, Fs):
		return rate_transition_single(Ss, Fs, J, g, l, model, params, dim)
	return vmap(nf, in_axes=(0, 0), out_axes=(0))

def get_rate_transitions(J, g, l, model, params, dim):
	nf = get_rate_transition(J, g, l, model, params, dim)

	return vmap(nf, in_axes=(1, 1), out_axes=1)

def ising_endpoint_loss(trajectories, Ts, Fs, model, params, J, g, lattice_size, dim):
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
	rate_transitions = get_rate_transitions(J, g, lattice_size, model, params, dim)

	logRN = 0.0
	V = ising_potentialV(trajectories, J, g, dim)
	Vt = jnp.sum(jnp.multiply(Ts.squeeze(), V.squeeze()), axis=1)

	T1 = passive_difference(trajectories, J, g, model, params, dim)
	T1t = jnp.sum(jnp.multiply(Ts.squeeze(), T1.squeeze()), axis=1)

	T2 = rate_transitions(trajectories, Fs)
	T2s = jnp.sum(T2, axis=1).squeeze()

	logRN = T1t + T2s + Vt
	Eest = Vt[0] + T1t[0] + T2s[0]

	# print(15*'-' + 'new epoch', 15*'-')
	# print("potential: ", jax.lax.stop_gradient(V.T))
	# print("logrn T1: ", jax.lax.stop_gradient(T1.T))
	# print("logrn T2: ", jax.lax.stop_gradient(T2.squeeze()))
	# print("Permuted times: ", Ts[:, :, 0])
	# print("Permuted flips: ", Fs[:, :, 0])
	# print("times shape: ", jnp.shape(Ts))
	# print("flips shape: ", jnp.shape(Fs))
	# print("trajectories shape: ", jnp.shape(trajectories))
	# print("Times after sampling (should equal T), ", jnp.sum(Ts[:, :, 0], axis=1))
	# print("Length of trajectory, ", jnp.sum(Ts[:, :, 0], axis=1))
	# print("logRN + E0 of the first trajectory (should get closer and closer to E0), ", jax.lax.stop_gradient(Eest)/jnp.sum(Ts[0, :, 0]))
	# print("First term contribution to logRN, ", jax.lax.stop_gradient(Vt))
	# print("Second term contribution to logRN, ", jax.lax.stop_gradient(T1t))
	# print("Third term contribution to logRN, ", jax.lax.stop_gradient(T2s))
	# print("logRN of each permutation", jax.lax.stop_gradient(logRN))

	return jnp.var(logRN, ddof=1), jax.lax.stop_gradient(Eest)/jnp.sum(Ts[0, :, 0])

# def Ieploss(trajectories, Ts, Fs, model, params, J, g, lattice_size, dim)

# 	grt = lambda: model, J, g, lattice_size

# 	def ipl(model, params, trajectories, Ts, Fs):
# 			rate_transitions = get_rate_transitions(J, g, lattice_size, model, params, dim)

# 			logRN = 0.0
# 			V = ising_potentialV(trajectories, J, g, dim)
# 			Vt = jnp.sum(jnp.multiply(Ts.squeeze(), V.squeeze()), axis=1)

# 			T1 = passive_difference(trajectories, J, g, model, params, dim)
# 			T1t = jnp.sum(jnp.multiply(Ts.squeeze(), T1.squeeze()), axis=1)

# 			T2 = rate_transitions(trajectories, Fs)
# 			T2s = jnp.sum(T2, axis=1).squeeze()

# 			logRN = T1t + T2s + Vt
# 			Eest = Vt[0] + T1t[0] + T2s[0]

# 	return ipl


def get_Ieploss(J, g, lattice_size, dim):
	"""
	Return Ising endpoint loss for a fixed lattice size, J and g constants
	"""
	f = lambda model, params, trajectories, Ts, Fs: ising_endpoint_loss(trajectories, Ts, Fs, model, params, J, g, lattice_size, dim)

	return f





