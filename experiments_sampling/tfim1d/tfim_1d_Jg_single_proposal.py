"""
Testing new batch generation proposals
"""

import sys
sys.path.append('../../')

import ml_collections
import Trainer as tr

import configs.ising1d_configs as iconf
import numpy as np

import gc

def single_run(L, g): 
	conf = iconf.get_defaults()
	out = "g{}".format(int(g*100))

	conf.batch_type = 'construct'
	conf.batch_Tvar = False

	conf.batch_size = 15
	conf.num_epochs = 100

	conf.T = 1
	conf.g = g
	conf.L = L
	conf.J = 1.0

	# construct experiment
	ising_ex1 = tr.Trainer(experiment_name="batch_prop_single{}".format(L),
			 config=conf,
			 output_dir=out)

	ising_ex1.setup_experiment()
	out = ising_ex1.train_rates(prngn=99)

if __name__ == '__main__':
	single_run(6, g=0.2)
	single_run(6, g=0.4)
	single_run(6, g=0.6)
	single_run(6, g=0.8)
	single_run(6, g=1.0)
	single_run(6, g=1.2)
	single_run(6, g=1.4)
	single_run(6, g=1.6)
	single_run(6, g=1.8)
