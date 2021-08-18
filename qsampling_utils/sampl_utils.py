"""
Sampling utilities for continuous-time Markov chains (CTMC)

"""

import jax
import jax.random as rnd
import jax.numpy as jnp

from jax import jit

@jit
def step_max(key, G):
	"""
	Make single step with the 
	hold time jump chain definition of a CTMC

	Params:
	-------
	key -- PRNGKey
	G -- rate matrix

	Returns:
	--------
	tau -- holding time in the state
	s -- index of next state, take care, function flattens the input G with 'C' style ordering (row major)
	subkey -- new PRNGKey, split from initial key

	"""

	# sample times from exponential distribution ~Exp(1) and rescale according to rates
	times = jnp.multiply(rnd.exponential(key, shape=jnp.shape(G)), jnp.reciprocal(G))
	key, subkey = rnd.split(key)

	# holding time is the minimum time of all the samples, the next state is this times corresponding state
	tau = jnp.min(jnp.ravel(times, order='C'))
	s = jnp.argmin(jnp.ravel(times, order='C'))

	return tau, s, subkey

@jit
def step_gumbel(key, G):
	"""
	Make single step with the 
	competing exponentials definition of a CTMC

	We make use of the Gumbel trick. The distribution of
	the minimal of independent exponential distributions
	with {t1, t2, ..., tN} with parameters {l1, l2, ..., lN}
	is distributed as

	V = min{t1, t2, ..., tN} ~ Exp(l1+l2+...+lN).

	And the minimizer i* is independent of V with multinomial 
	distribution of \pi, with \pi_i = \frac{li}{l1 + l2 + ... + lN}. 

	i* = arg min_i {ti} ~ Multinomial(\pi; \pi_i = \frac{li}{l1 + l2 + ... + lN}.) 

	This means, we can sample the time in the state from the exponential
	distribution and the state change independently.

	Params:
	-------
	key -- PRNGKey
	G -- transition rates for all adjacent states, 
	i.e. in the Ising model rates of spin flips

	Returns:
	--------
	tau -- holding time in the state
	s -- index of next state, take care, function flattens the input G with 'C' style ordering (row major)
	subkey -- new PRNGKey, split from initial key

	"""
	# sum of all the rates
	lamb = jnp.sum(G) 

	# rnd.exponential samples X ~ Exp(1), 
	# for exponential distr. X/lambda ~ Exp(lambda)
	tau = rnd.exponential(key) / lamb
	
	# generate new random keys
	key, subkey = rnd.split(key) 

	# sample from the categorical distribution to choose action
	s = rnd.categorical(key, jnp.log(jnp.ravel(G, order='C')/lamb)) # categorical takes logits as input

	return tau, s, subkey

