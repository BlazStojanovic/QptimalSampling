"""Default Hyperparameter configuration for Ising model case"""

import ml_collections

def get_config():
	"""Get the default hyperparams"""
	config = ml_collections.ConfigDict()

	# optimisation hyperparameters
	config.learning_rate = 0.01
	config.b1 = 0.9
	config.b2 = 0.999

	config.batch_size = 128
	config.num_epochs = 15

	# architecture params
	config.out_channels = 1
	config.hid_channels = 2
	config.kernel_size = (3,3)
	config.layers = 4

	# physics model parameters
	config.lattice_size = 5
	config.T = 10
	config.trajectory_length = 32 # this is for the alternative sampling approach
	config.J = 0.32758
	config.g = 1.0

	# evaluation step params
	config.energy_samples = 500
	config.skip_percentage = 0.2

	return config
