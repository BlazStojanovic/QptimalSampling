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

def run_for_L(L, i): # todo add flexibility
	# IO details
	conf = iconf.get_structure()
	out = "run-{}".format(int(i))
	conf.batch_type = 'permute'
	conf.training_mode = 'adaptive'
		
	conf.L = L

	# construct experiment
	ising_ex1 = tr.Trainer(experiment_name="tfim1d_structure-L{}".format(L),
			 config=conf,
			 output_dir=out)

	ising_ex1.setup_experiment()
	out = ising_ex1.train_rates(prngn=i)
	gc.collect()

if __name__ == '__main__':

	run_for_L(6, 0)
	run_for_L(6, 1)
