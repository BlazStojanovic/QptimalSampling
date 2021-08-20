import sys
sys.path.append('../')

import Trainer as tr
import Sampler as sa

import configs.ising_defaults as iconf

if __name__ == '__main__':
	
	Ts = [10, 50, 100, 150, 200]

	for t in Ts:
		# IO details
		conf = iconf.get_time_vs_loss()
		conf.T = t
		conf.t_vector_increment = int(t*conf.t_vector_increment//10) # see how trajectories are generated
		conf.batch_type = 'permute'
		conf.training_mode = 'adaptive'
		print(conf.t_vector_increment)

		experiment_name = "time_vs_loss"
		out = "T{}".format(int(t))

		# construct experiment
		ising_ex1 = tr.Trainer(experiment_name=experiment_name,
				 config=conf,
				 output_dir=out)

		ising_ex1.setup_experiment()
		out = ising_ex1.train_rates()