import sys
sys.path.append('../')

import Trainer as tr
import Sampler as sa

import configs.ising_defaults as iconf

if __name__ == '__main__':
	
	N_bs = [100, 130]

	for nbc in N_bs:
		# IO details
		conf = iconf.get_batch_vs_loss()
		conf.batch_size = nbc

		experiment_name = "batch_vs_loss"
		out = "nb{}".format(int(nbc))

		# construct experiment
		ising_ex1 = tr.Trainer(experiment_name=experiment_name,
				 config=conf,
				 output_dir=out)

		ising_ex1.setup_experiment()
		out = ising_ex1.train_rates()
