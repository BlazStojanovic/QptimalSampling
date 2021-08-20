"""
Simple test for 1D TFIM
"""

import sys
sys.path.append('../../')

import ml_collections
import Trainer as tr

import configs.ising1d_configs as iconf
import numpy as np

import gc

def run_for_L(L): # todo add flexibility
	# ratios = np.linspace(0.1, 1.9, 11)
	ratios = [0.1, 1.9]
	print(ratios)

	for i, r in enumerate(ratios):
		# IO details
		conf = iconf.get_defaults()
		out = "gjrat{}".format(int(r*100))

		conf.g = r
		conf.L = L
		conf.J = 1.0

		# construct experiment
		ising_ex1 = tr.Trainer(experiment_name="tfim1d_gjrat_single{}".format(L),
				 config=conf,
				 output_dir=out)

		ising_ex1.setup_experiment()
		out = ising_ex1.train_rates(prngn=2)
		gc.collect()

		np.save("../data/tfim1d_gjrat_single{}/ratios.npy".format(L), ratios)

if __name__ == '__main__':

	# run_for_L(3)
	run_for_L(6)
	# run_for_L(12)
