"""Default Hyperparameter configuration for Ising model case"""

import ml_collections

def get_config():
	"""Get the default hyperparams"""
	config = ml_collections.ConfigDict()

	# optimisation hyperparameters
	config.learning_rate = 0.01
	config.b1 = 0.9
	config.b2 = 0.999

	config.batch_size = 4
	config.num_epochs = 10

	# architecture params
	config.loss_type = 'endpointloss'
	config.architecture = 'pCNN'
	config.out_channels = 1
	config.hid_channels = 5
	config.kernel_size = (3,3)
	config.layers = 3

	# physics model parameters
	config.lattice_size = 3
	config.T = 2
	config.max_trajectory_length = 10000
	config.trajectory_length = 5 # this is for the alternative sampling approach
	config.J = 1.0
	config.g = 1.0


	return config
