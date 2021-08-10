"""
Simple test for 1D TFIM
"""

import sys
sys.path.append('../../')

import Trainer as tr

import configs.ising1d_configs as iconf
import numpy as np

import gc

if __name__ == '__main__':
		
		ratios = np.linspace(0.5, 3.5, 10)
		# ratios = [1.83333333, 2.16666667, 2.5, 2.83333333, 3.16666667, 3.5]
		# ratios = [1]

		for i, r in enumerate(ratios):
			# IO details
			conf = iconf.get_defaults()
			out = "gjrat{}".format(int(r*100))

			conf.J = r
			N = conf.L

			# construct experiment
			ising_ex1 = tr.Trainer(experiment_name="tfim1d_gjrat_single{}".format(N),
					 config=conf,
					 output_dir=out)

			ising_ex1.setup_experiment()
			out = ising_ex1.train_rates(prngn=2)
			gc.collect()

		np.save("../data/tfim1d_gjrat_single12/ratios.npy", ratios)