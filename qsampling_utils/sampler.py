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


if __name__ == '__main__':
	# Just some quick testing, TODO do not forget to delete later

	# test rate matrix
	G = jnp.array([[1.2, 0.2, 0.7],
				   [3.0, 0.4, 0.1],
				   [1.1, 2.3, 0.9],
				   [0.7, 1.8, 0.3]])

	N = 1000000 # iterations

	print("Testing gumbel approach for N = {}".format(N))

	key = rnd.PRNGKey(42069)
	counts_gumbel = jnp.array([0.0, 0.0, 0.0, 0.0])
	sum_times_gumbel = jnp.array([0.0, 0.0, 0.0, 0.0])

	state = 2 # initialise starting state

	@jit
	def bodyf(i, S):
		# decompose S into components
		state, G, key, counts_gumbel, sum_times_gumbel = S


		lamb = jnp.arange
		tau, s, key = step_gumbel(key, G[state, :])
		

		s = jax.lax.cond(s+1>=state, lambda x: x+1, lambda x: x, operand=s)

		counts_gumbel = jax.ops.index_add(counts_gumbel, jax.ops.index[state], 1)
		sum_times_gumbel = jax.ops.index_add(sum_times_gumbel, jax.ops.index[state], tau)

		state = s
		return state, G, key, counts_gumbel, sum_times_gumbel


	S = state, G, key, counts_gumbel, sum_times_gumbel
	S = jax.lax.fori_loop(0, N, bodyf, S)
	state, G, key, counts_gumbel, sum_times_gumbel = S	

	print("Frequency of each state: ", counts_gumbel)
	print("Time spent in each state: ", sum_times_gumbel)


	print("===============================================================================")
	print("Testing max approach for N = {}".format(N))

	key = rnd.PRNGKey(42069)
	counts_gumbelm = jnp.array([0.0, 0.0, 0.0, 0.0])
	sum_times_gumbelm = jnp.array([0.0, 0.0, 0.0, 0.0])

	@jit
	def bodyf(i, S):
		# decompose S into components
		state, G, key, counts_gumbel, sum_times_gumbel = S


		lamb = jnp.arange
		tau, s, key = step_max(key, G[state, :])
		

		s = jax.lax.cond(s+1>=state, lambda x: x+1, lambda x: x, operand=s)

		counts_gumbel = jax.ops.index_add(counts_gumbel, jax.ops.index[state], 1)
		sum_times_gumbel = jax.ops.index_add(sum_times_gumbel, jax.ops.index[state], tau)

		state = s
		return state, G, key, counts_gumbel, sum_times_gumbel


	S = state, G, key, counts_gumbelm, sum_times_gumbelm
	S = jax.lax.fori_loop(0, N, bodyf, S)
	state, G, key, counts_gumbelm, sum_times_gumbelm = S	

	print("Frequency of each state: ", counts_gumbelm)
	print("Time spent in each state: ", sum_times_gumbelm)

	print("===============================================================================")
	print("Frequency ratio: ", counts_gumbel*jnp.reciprocal(counts_gumbelm))
	print("Time ratio: ", sum_times_gumbel*jnp.reciprocal(sum_times_gumbelm))
