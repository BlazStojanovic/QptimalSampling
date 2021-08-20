"""
Showing that the rates blow up, sample smaller and smaller timescales for this loss
"""

import sys
sys.path.append('../../')

import ml_collections
import Trainer as tr
import configs.ising2d_configs as iconf
import numpy as np

if __name__ == '__main__':

		# seed = [0, 11, 22, 33, 44]
		seed = [22]

		for i in seed:
			# IO details
			print("seed: i")

			conf = iconf.get_defaults()
			out = "{}".format(i)
			conf.batch_type = 'permute'
			conf.training_mode = 'adaptive'

			conf.L = 3
			conf.num_epochs = 30

			# construct experiment
			ising_ex1 = tr.Trainer(experiment_name="rate_blowup",
					 config=conf,
					 output_dir=out)

			ising_ex1.setup_experiment()
			out = ising_ex1.train_rates(prngn=i)
