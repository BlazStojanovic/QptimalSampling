"""
Calculating endpoint loss for the 2D Ising model. Along with helper functions
"""

import jax.numpy as jnp
from jax import jit, vmap

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
	zr = jnp.roll(z, -1, axis=-2)

	# first term
	potential = z*zu + z*zr
	potential = -J*jnp.sum(potential, axis=(-1, -2, -3))

	#second term
	potential -= g*jnp.sum(jnp.ones(jnp.shape(S)), axis=(-1, -2, -3, -4))

	return potential

# vectorize the ising potential
ising_potential = vmap(ising_potential_single, in_axes=(0, None, None), out_axes=(0))

def passive_ising(S0, g):
	return g*jnp.ones(jnp.shape(S0))

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
	passive_rates = g*jnp.sum(jnp.ones(jnp.shape(S)), axis=(-1, -2, -3, -4))

	return passive_rates - rates

passive_difference = vmap(passive_difference_single, in_axes=(0, None, None, None, None), out_axes=(0))

# def rate_transition_single(S, f, J, g, lattice_size, model, params):
# 	"""
# 	Returns a term for a single n of the third loss term (transition rate between subsequent states)
	
# 	Params:
# 	-------
# 	S -- state
# 	J -- interaction coefficient
# 	g -- coupling coefficient
# 	model -- variational approximation of the rates
# 	params -- params of the variational approximation

# 	Returns:
# 	--------
# 	diff -- single term of the third loss
# 	"""

# 	from variational rates
# 	rates = model.apply({'params': params['params']}, S) # get rate corresponding to spin flip f
# 	print(jnp.shape(rates))
# 	rate = rates[jnp., 0, f//lattice_size, f%lattice_size, 0]


# 	from passive rates
# 	passive = g

# 	return jnp.log(rate/passive)

# gotta map over first two arguments
# rate_transition = vmap(rate_transition_single, in_axes=(0, None, None, None, None, None, None), out_axes=(0)) 
# rate_transition = vmap(rate_transition, in_axes=(None, 0, None, None, None, None, None), out_axes=(0)) 


def rate_transition(trajectories, Fs, J, g, l, model, params):
	
	Nb, Nt = jnp.shape(trajectories)[0], jnp.shape(trajectories)[1]
	# print(Nb, Nt)

	loss_ = jnp.zeros((jnp.shape(trajectories)[0]))

	# transitions between all states
	for i in range(jnp.shape(trajectories)[1]):
		rates = model.apply({'params': params['params']}, trajectories[:, i, :, :, :]) # (Nb, l, l, 1)
		transition_rates = rates[jnp.arange(Nb), Fs[:, i-1, 0] // l, Fs[:, i-1, 0] % l, :]
		# print(transition_rates)

		loss_ += jnp.log(transition_rates/g)

	return loss_


def ising_endpoint_loss(trajectories, Ts, Fs, model, params, J, g, lattice_size):
	"""
	Loss for fixed trajectory endpoints

	Params:
	-------
	trajectories -- Batch of trajectories with shape (Nt, Nb, l, l, 1) where 
	Ts is the no. of time steps, Nb is the batch size and l is the lattice size
	times -- Batch of holding times with shape (Nb, Nt, 1)
	Fs -- Batch of spin flips with shape (Nb, Nt, 1)
	model, params -- the variational parameterisation alongside the optimizable params

	Returns:
	loss -- logRN loss of the batch
	"""
	
	# 1st term - Feynman-Kac weight on the trajectory
	# print(ising_potential(trajectories, J, g))
	# print(Ts[:, :, 0])

	logRN = jnp.sum(jnp.multiply(ising_potential(trajectories, J, g), Ts[:, :, 0]), axis=1)

	# 2nd term - difference between rows of rate matrices at each step 
	# TODO rejoin second and third term into a single calculation for efficiency
	logRN += jnp.sum(jnp.multiply(passive_difference(trajectories, J, g, model, params) , Ts[:, :, 0]), axis=1) 

	# 3rd term - difference 
	# print(jnp.shape(trajectories))
	# print(jnp.shape(Fs))
	# print(jnp.shape(rate_transition(trajectories, Fs, J, g, lattice_size, model, params)))
	# logRN += jnp.sum(rate_transition(trajectories, Fs, J, g, lattice_size, model, params), axis=1)
	logRN += jnp.sum(rate_transition(trajectories, Fs, J, g, lattice_size, model, params), axis=1)


	return jnp.mean(logRN)