"""Default Hyperparameter configuration for Ising model case"""

import ml_collections

def get_config():
	"""Get the default hyperparams"""
	config = ml_collections.ConfigDict()

	# optimisation hyperparameters
	config.learning_rate = 0.1
	config.b1 = 0.9
	config.b2 = 0.999

	config.batch_size = 128
	config.num_epochs = 100

	# architecture params
	config.out_channels = 1
	config.hid_channels = 3
	config.kernel_size = (3,3)
	config.layers = 5

	# physics model parameters
	config.lattice_size = 7
	config.T = 20
	config.J = 1.1

	return config
