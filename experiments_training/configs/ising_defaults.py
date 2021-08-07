""" default hyperparam config for 2d TFIM """

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
	config.batch_size = 32
	config.num_epochs = 10
	config.chpt_freq = 3 # checkpoint frequency

	### architecture params
	config.loss_type = 'endpoint'
	config.architecture = 'pcnn'
	config.out_channels = 1
	config.hid_channels = 2
	config.kernel_size = (3,3)
	config.layers = 3

	### validation
	config.no_valids = 3
	config.valids = ['energy_estim', 'epoch_time', 'no_in_traj']

	# physics model parameters
	config.lattice_model = 'ising'

	### dimensionality
	config.dim = 2
	config.L = 3

	### magnetisation and transverse field
	config.g = 1.0
	config.J = 1.0

	### simulation time
	config.T = 20
	config.t_vector_increment = 1000 # see how trajectories are generated

	# sampling param
	config.sample_step = 'gumble' # gumble or max step

	return config

def get_batch_vs_loss():
	config = ml_collections.ConfigDict()

	# learninig hyperparameters
	### rates
	config.learning_rate = 0.01
	config.b1 = 0.9
	config.b2 = 0.999

	### type
	config.optimizer = 'adam'

	### training details
	config.batch_size = 0
	config.num_epochs = 400
	config.chpt_freq = 3 # checkpoint frequency

	### architecture params
	config.loss_type = 'endpoint'
	config.architecture = 'pcnn'
	config.out_channels = 1
	config.hid_channels = 5
	config.kernel_size = (3,3)
	config.layers = 3

	### validation
	config.no_valids = 3
	config.valids = ['energy_estim', 'epoch_time', 'no_in_traj']

	# physics model parameters
	config.lattice_model = 'ising'

	### dimensionality
	config.dim = 2
	config.L = 3

	### magnetisation and transverse field
	config.g = 1.0
	config.J = 1.0

	### simulation time
	config.T = 75
	config.t_vector_increment = 2000 # see how trajectories are generated

	# sampling param
	config.sample_step = 'gumble' # gumble or max step

	return config

def get_time_vs_loss():
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
	config.num_epochs = 400
	config.chpt_freq = 3 # checkpoint frequency

	### architecture params
	config.loss_type = 'endpoint'
	config.architecture = 'pcnn'
	config.out_channels = 1
	config.hid_channels = 5
	config.kernel_size = (3,3)
	config.layers = 3

	### validation
	config.no_valids = 3
	config.valids = ['energy_estim', 'epoch_time', 'no_in_traj']

	# physics model parameters
	config.lattice_model = 'ising'

	### dimensionality
	config.dim = 2
	config.L = 3

	### magnetisation and transverse field
	config.g = 1.0
	config.J = 1.0

	### simulation time
	config.T = 0
	config.t_vector_increment = 120 # see how trajectories are generated

	# sampling param
	config.sample_step = 'gumble' # gumble or max step

	return config

def get_lay_vs_loss():
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
	config.num_epochs = 400
	config.chpt_freq = 3 # checkpoint frequency

	### architecture params
	config.loss_type = 'endpoint'
	config.architecture = 'pcnn'
	config.out_channels = 1
	config.hid_channels = 5
	config.kernel_size = (3,3)
	config.layers = 0

	### validation
	config.no_valids = 3
	config.valids = ['energy_estim', 'epoch_time', 'no_in_traj']

	# physics model parameters
	config.lattice_model = 'ising'

	### dimensionality
	config.dim = 2
	config.L = 3

	### magnetisation and transverse field
	config.g = 1.0
	config.J = 1.0

	### simulation time
	config.T = 50
	config.t_vector_increment = 1000 # see how trajectories are generated

	# sampling param
	config.sample_step = 'gumble' # gumble or max step

	return config

def get_width_vs_loss():
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
	config.num_epochs = 400
	config.chpt_freq = 3 # checkpoint frequency

	### architecture params
	config.loss_type = 'endpoint'
	config.architecture = 'pcnn'
	config.out_channels = 1
	config.hid_channels = 0
	config.kernel_size = (3,3)
	config.layers = 3

	### validation
	config.no_valids = 3
	config.valids = ['energy_estim', 'epoch_time', 'no_in_traj']

	# physics model parameters
	config.lattice_model = 'ising'

	### dimensionality
	config.dim = 2
	config.L = 3

	### magnetisation and transverse field
	config.g = 1.0
	config.J = 1.0

	### simulation time
	config.T = 50
	config.t_vector_increment = 1000 # see how trajectories are generated

	# sampling param
	config.sample_step = 'gumble' # gumble or max step

	return config

def get_TB():
	config = ml_collections.ConfigDict()

	# learninig hyperparameters
	### rates
	config.learning_rate = 0.01
	config.b1 = 0.9
	config.b2 = 0.999

	### type
	config.optimizer = 'adam'

	### training details
	config.batch_size = 0
	config.num_epochs = 90
	config.chpt_freq = 3 # checkpoint frequency

	### architecture params
	config.loss_type = 'endpoint'
	config.architecture = 'pcnn'
	config.out_channels = 1
	config.hid_channels = 3
	config.kernel_size = (3,3)
	config.layers = 3

	### validation
	config.no_valids = 3
	config.valids = ['energy_estim', 'epoch_time', 'no_in_traj']

	# physics model parameters
	config.lattice_model = 'ising'

	### dimensionality
	config.dim = 2
	config.L = 3

	### magnetisation and transverse field
	config.g = 1.0
	config.J = 1.0

	### simulation time
	config.T = 0
	config.t_vector_increment = 200 # see how trajectories are generated

	# sampling param
	config.sample_step = 'gumble' # gumble or max step

	return config

def get_WL():
	config = ml_collections.ConfigDict()

	# learninig hyperparameters
	### rates
	config.learning_rate = 0.01
	config.b1 = 0.9
	config.b2 = 0.999

	### type
	config.optimizer = 'adam'

	### training details
	config.batch_size = 32
	config.num_epochs = 90
	config.chpt_freq = 3 # checkpoint frequency

	### architecture params
	config.loss_type = 'endpoint'
	config.architecture = 'pcnn'
	config.out_channels = 1
	config.hid_channels = 0
	config.kernel_size = (3,3)
	config.layers = 0

	### validation
	config.no_valids = 3
	config.valids = ['energy_estim', 'epoch_time', 'no_in_traj']

	# physics model parameters
	config.lattice_model = 'ising'

	### dimensionality
	config.dim = 2
	config.L = 3

	### magnetisation and transverse field
	config.g = 1.0
	config.J = 1.0

	### simulation time
	config.T = 30
	config.t_vector_increment = 800

	# sampling param
	config.sample_step = 'gumble' # gumble or max step

	return config