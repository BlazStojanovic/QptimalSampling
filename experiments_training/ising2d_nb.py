import sys
sys.path.append('../')

import Trainer as tr
import Sampler as sa

import configs.ising_defaults as iconf

if __name__ == '__main__':
	
	for nbc in [16, 32, 64, 96]:
		# IO details
		conf = iconf.get_ising_2d()
		conf.batch_size = nbc
		out = "nb{}".format(int(nbc))

		# construct experiment
		ising_ex1 = tr.Trainer(experiment_name="ising_2d_meeting1_fig",
				 config=conf,
				 output_dir=out)

		ising_ex1.setup_experiment()
		out = ising_ex1.train_rates()
