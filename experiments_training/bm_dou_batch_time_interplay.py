import sys
sys.path.append('../')

import Trainer as tr

import configs.ising_defaults as iconf
import numpy as np

if __name__ == '__main__':
	
	N_bs =  (np.arange(0, 11)*10 + 2).astype(int) # range 2 - 100
	Ts =  (np.arange(0, 11)*10 + 2).astype(int) # range 2 - 100

	for Nb in N_bs:
		for t in Ts:
			# IO details
			conf = iconf.get_TB()
			conf.batch_size = int(Nb)
			conf.T = int(t)
			conf.t_vector_increment = int(t*conf.t_vector_increment//10) # see how trajectories are generated
			experiment_name = "TB_interplay"
			out = "Nb{}T{}".format(int(Nb), int(t))

			# construct experiment
			ising_ex1 = tr.Trainer(experiment_name=experiment_name,
					 config=conf,
					 output_dir=out)

			ising_ex1.setup_experiment()
			out = ising_ex1.train_rates(prngn=999)