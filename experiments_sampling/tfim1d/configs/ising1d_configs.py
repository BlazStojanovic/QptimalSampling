""" default hyperparam config for 1d TFIM """

import ml_collections

def get_defaults(): # this is the full experimental one for TFIM 1d
	config = ml_collections.ConfigDict()

	# learninig hyperparameters
	### rates
	config.learning_rate = 0.01
	config.b1 = 0.9
	config.b2 = 0.999

	### type
	config.optimizer = 'adam'

	### training details
	config.batch_type = 'construct' # can be permute, split, construct
	config.batch_Tvar = False
	config.batch_size = 20
	config.batch_Na = 4
	config.batch_Neven = 2

	config.num_epochs = 400
	config.chpt_freq = 5 # checkpoint frequency

	### architecture params
	config.training_mode = 'adaptive' # only applies for permutations and split
	config.loss_type = 'endpoint'
	config.architecture = 'pcnn'
	config.out_channels = 1
	config.hid_channels = 5
	config.kernel_size = 3
	config.layers = 3

	### validation
	config.no_valids = 3
	config.valids = ['energy_estim', 'epoch_time', 'no_in_traj']

	# physics model parameters
	config.lattice_model = 'ising'

	### dimensionality
	config.dim = 1
	config.L = 0

	### magnetisation and transverse field
	config.g = 1.0
	config.J = 1.0

	### simulation time
	config.T = 1
	config.t_vector_increment = 10000 # see how trajectories are generated

	# sampling param
	config.sample_step = 'gumble' # gumble or max step

	return config


def get_structure():
	config = ml_collections.ConfigDict()

	# learninig hyperparameters
	### rates
	config.learning_rate = 0.01
	config.b1 = 0.9
	config.b2 = 0.999

	### type
	config.optimizer = 'adam'

	### training details
	config.batch_size = 40
	config.num_epochs = 90
	config.chpt_freq = 10 # checkpoint frequency

	### architecture params
	config.training_mode = 'adaptive'
	config.loss_type = 'endpoint'
	config.architecture = 'pcnn'
	config.out_channels = 1
	config.hid_channels = 10
	config.kernel_size = 3
	config.layers = 6

	### validation
	config.no_valids = 3
	config.valids = ['energy_estim', 'epoch_time', 'no_in_traj']

	# physics model parameters
	config.lattice_model = 'ising'

	### dimensionality
	config.dim = 1
	config.L = 0

	### magnetisation and transverse field
	config.g = 1.0
	config.J = 1.0

	### simulation time
	config.T = 10
	config.t_vector_increment = 50000 # see how trajectories are generated

	# sampling param
	config.sample_step = 'gumble' # gumble or max step

	return config