""" default hyperparam config for 1d TFIM """

import ml_collections

def get_defaults():
	config = ml_collections.ConfigDict()

	# learninig hyperparameters
	### rates
	config.learning_rate = 0.01
	config.b1 = 0.9
	config.b2 = 0.999

	### type
	config.optimizer = 'adam'

	### training details
	config.batch_size = 50
	config.num_epochs = 120
	config.chpt_freq = 10 # checkpoint frequency

	### architecture params
	config.loss_type = 'endpoint'
	config.architecture = 'pcnn'
	config.out_channels = 1
	config.hid_channels = 15
	config.kernel_size = 5
	config.layers = 8

	### validation
	config.no_valids = 3
	config.valids = ['energy_estim', 'epoch_time', 'no_in_traj']

	# physics model parameters
	config.lattice_model = 'ising'

	### dimensionality
	config.dim = 1
	config.L = 24

	### magnetisation and transverse field
	config.g = 1.0
	config.J = 1.0

	### simulation time
	config.T = 60
	config.t_vector_increment = 1000 # see how trajectories are generated

	# sampling param
	config.sample_step = 'gumble' # gumble or max step

	return config