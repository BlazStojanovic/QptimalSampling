import numpy as jnp
import matplotlib.pyplot as plt
from scipy.integrate import *

def e_q(jg, q):
	return jnp.sqrt(1 + jnp.square(jg)/4 - jg*jnp.cos(q))

def energy_per_site(jg):
	f = lambda q: e_q(jg, q)
	return -0.5*(quad(f, 0, jnp.pi)[0])/jnp.pi # g = 1 case

def m_z(jg):
	f = lambda q: (1 - 0.5*jg*jnp.cos(q))/e_q(jg, q)
	return (quad(f, 0, jnp.pi)[0])/jnp.pi

def m_x(jg):
	# if jg > 2:
	return jnp.power(1-4*jnp.reciprocal(jnp.square(jg)), 1/8.)
	# else:
	# 	return 0


m_z = jnp.vectorize(m_z)
# m_x = jnp.vectorize(m_x)
energy_per_site = jnp.vectorize(energy_per_site)

if __name__ == '__main__':
	jg = jnp.linspace(0, 4, 300)
	en = energy_per_site(jg)
	mz = m_z(jg)

	plt.plot(jg, en)
	plt.show()
	plt.plot(jg, mz)
	plt.show()	