"""
Simple test for 1D TFIM
"""

import sys
sys.path.append('../../')

import Trainer as tr
import Sampler as sa

import configs.ising1d_configs as iconf

if __name__ == '__main__':
		# IO details
		conf = iconf.get_defaults()
		out = "g1j1"

		# construct experiment
		ising_ex1 = tr.Trainer(experiment_name="tfim1d_test",
				 config=conf,
				 output_dir=out)

		ising_ex1.setup_experiment()
		out = ising_ex1.train_rates()