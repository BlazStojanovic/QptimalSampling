import sys
sys.path.append('../')

import Trainer as tr

import configs.ising_defaults as iconf
import numpy as np


if __name__ == '__main__':
	
	widths =  (np.arange(10, 11)*2 + 1).astype(int)
	layers =  (np.arange(10, 11) + 2).astype(int)

	for w in widths:
		for l in layers:
			# IO details
			conf = iconf.get_WL()
			conf.hid_channels = int(w)
			conf.layers = int(l)

			print("width: {}, layers {}".format(w, l))

			experiment_name = "WL_interplay"
			out = "W{}L{}".format(int(w), int(l))

			# construct experiment
			ising_ex1 = tr.Trainer(experiment_name=experiment_name,
					 config=conf,
					 output_dir=out)

			ising_ex1.setup_experiment()
			out = ising_ex1.train_rates(prngn=999)