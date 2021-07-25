"""
Main file for running the Ising model example. 
"""

import train_rates
import configs.defaults

import numpy as np

import os
import json

from flax import serialization

def store_rates(config, storedir, out):
	dirs = [ f.name for f in os.scandir(storedir) if f.is_dir() ]
	path = storedir+"1/" # first experiment dir

	# create new dir with appropriate run number
	if not dirs:
		print("Empty storage directory, creating first experiment dir /1/")
		try:
			os.mkdir(path)
		except OSError:
			print ("Creation of the directory %s failed" % path)
	else:
		dnum = 1 + max(list(map(int, dirs)))
		path = storedir+"{}/".format(dnum)
		print("Saving into expermient dir /{}/".format(dnum))
		try:
			os.mkdir(path)
		except OSError:
			print ("Creation of the directory %s failed" % path)

	# store
	with open(path+'/config.json', 'w') as fp:
		json.dump(config.to_dict(), fp)
	
	state, loss, eest = out

	# save the loss 
	np.save(path+'/'+config.loss_type+".npy", loss)
	np.save(path+'/'+config.loss_type+"_eest.npy", eest)
	
	params = state.params

	bytes_output = serialization.to_bytes(params)


	f = open(path+'/params.txt', 'wb')
	f.write(bytes_output)
	f.close()


if __name__ == '__main__':
	
	# set config and 
	config = configs.defaults.get_config()
	workdir = "tmp/ising/"
	storedir = "ising_rates/"
	
	out = train_rates.train(config, storedir, workdir)
	
	store_rates(config, storedir, out)