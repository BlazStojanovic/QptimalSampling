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

def single_run_time(L, g, t, appx, tmod): 
	
	for T in t:
		conf = iconf.get_defaults()
		out = "g{}T{}".format(int(g*100), T)

		conf.batch_type = 'construct'
		conf.batch_Tvar = tmod

		conf.batch_size = 15
		conf.num_epochs = 250

		conf.T = T
		conf.g = g
		conf.L = L
		conf.J = 1.0

		# constructss experiment
		ising_ex1 = tr.Trainer(experiment_name="batch_prop_troub{}".format(L),
				 config=conf,
				 output_dir=out)

		ising_ex1.setup_experiment()
		out = ising_ex1.train_rates(prngn=T)

if __name__ == '__main__':
	t = [1, 2, 4]
	# t = [5, 10, 20]
	# t = [40, 60, 80]
	
	# ni se narejeno
	# single_run_time(6, 0.6, t, appx='F', tmod=False)
	# single_run_time(6, 1.0, t, appx='F', tmod=False)
	# single_run_time(6, 1.4, t, appx='F', tmod=False)


	single_run_time(6, 0.6, t, appx='T', tmod=False)
	# single_run_time(6, 1.0, t, appx='T', tmod=True)
	# single_run_time(6, 1.4, t, appx='T', tmod=True)