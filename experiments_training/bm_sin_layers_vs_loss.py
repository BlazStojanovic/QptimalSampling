import sys
sys.path.append('../')

import Trainer as tr
import Sampler as sa

import configs.ising_defaults as iconf

if __name__ == '__main__':
	
	# layers = [3, 9, 15, 30]
	layers = [9]

	for l in layers:
		# IO details
		conf = iconf.get_lay_vs_loss()
		conf.layers = l

		experiment_name = "layer_vs_loss"
		out = "lay{}".format(int(l))

		# construct experiment
		ising_ex1 = tr.Trainer(experiment_name=experiment_name,
				 config=conf,
				 output_dir=out)

		ising_ex1.setup_experiment()
		out = ising_ex1.train_rates(prngn=99)