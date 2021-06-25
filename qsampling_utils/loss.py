"""
Loss functions for learning optimal rate matrix of Lattice models

"""


def FKac_weight(V, taui):
	pass


def ising_endpoint_loss(X, S, model):
	"""
	Loss for fixed trajectory endpoints

	params:
	-------
	X - tensor of trajectories in the batch with shape
	(Nb - no. in batch, Nt - no. time steps, Ng - no. adjacent states, Ng - no. adjacent states)

	

	model - model class (Ising, XY, etc.)

	"""
	
	logRN = 0.0 # TODO think of vectorized case

	# 1st term - Feynman-Kac weight on the trajectory
	pass


def ising_endpoint_loss(X, S):
	pass