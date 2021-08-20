import sys
sys.path.append('../')

import Trainer as tr
import Sampler as sa

import configs.ising_defaults as iconf

if __name__ == '__main__':
	
	hiddens = [3, 9, 15]
	# hiddens = [9, 15]

	for h in hiddens:
		# IO details
		conf = iconf.get_width_vs_loss()
		conf.hid_channels = h

		conf.batch_type = 'permute'
		conf.training_mode = 'adaptive'

		experiment_name = "width_vs_loss"
		out = "wid{}".format(int(h))

		# construct experiment
		ising_ex1 = tr.Trainer(experiment_name=experiment_name,
				 config=conf,
				 output_dir=out)

		ising_ex1.setup_experiment()
		out = ising_ex1.train_rates(prngn=3*h)