import numpy as jnp
import matplotlib.pyplot as plt
from scipy.integrate import *

def e_q(g, q, j=1.0):
	return jnp.sqrt(1 + 1/4*jnp.square(4*j)*jnp.reciprocal(jnp.square(2*g)) - 4*j*jnp.reciprocal(2*g)*jnp.cos(q))

def energy_per_site(g, j=1.0):
	f = lambda q: e_q(g, q)
	return -g*(quad(f, 0, jnp.pi)[0])/jnp.pi # g = 1 case

def m_x(g, j=1.0):
	f = lambda q: (1 - j/g*jnp.cos(q))/e_q(g, q, j=j)
	return (quad(f, 0, jnp.pi)[0])/jnp.pi

def m_z(g, j=1.0):
	if j > g:
		return jnp.power(1-jnp.square(g)*jnp.reciprocal(jnp.square(j)), 1/8.)
	else:
		return 0

m_z = jnp.vectorize(m_z)
m_x = jnp.vectorize(m_x)
energy_per_site = jnp.vectorize(energy_per_site)