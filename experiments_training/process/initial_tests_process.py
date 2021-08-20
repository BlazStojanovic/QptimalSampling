"""
preprocess for initial training figure, 
meanining sampling the loss and finding how volatile it is, 
estimating execution times and memory usage

"""

import sys
sys.path.append('../../')
sys.path.append('../')

import Trainer as tr
import time

import numpy as np
import jax.numpy as jnp
import jax

import configs.ising_defaults as iconf

def loss_avg_var(key, load_path, no_sampl, t, b):
	lls = np.zeros((no_sampl,))
	conf = iconf.get_TB()
	
	# set time and batch size
	conf.batch_size = 50
	# conf.T = int(t)
	conf.batch_type = 'permute'
	conf.training_mode = 'adaptive'
	conf.T = 50
	conf.t_vector_increment = 7500

	out = "Nb{}T{}/final/".format(int(b), int(t))
	training_losses = np.load(load_path + out + 'loss.npy')
	# setup experiment
	bench = tr.Trainer(experiment_name="temp",
			 config=conf,
			 output_dir=out)

	bench.setup_experiment()
	dir_path = load_path + out
	print(dir_path)
	bench.load_from_chp(dir_path)

	params, model, tx, state, loss_, valids_, epoch_start = bench.init_training(prngn=111)

	for i in range(no_sampl):
		start_time = time.time()
		S0 = bench.initialise_lattice(key)

		# get trajectory
		times, flips, it = bench.get_trajectory(model, params, key, S0)
		times, flips = times[:it, :], flips[:it, :]

		Ts, Fs = bench.get_batch(key, times, flips)

		key, subkey = jax.random.split(key)
		trajectories =  bench.flip_to_trajectory(S0, jnp.shape(Ts)[1], Fs)

		lls[i], eest_ = bench.lossf(model, params, trajectories, Ts, Fs)		
		looptime = time.time() - start_time
		print("t-{}, B-{}: i = {}, took: {}".format(t, b, i, looptime))

	print("Last training step loss was, {}".format(training_losses[-1]))
	print("10 more samples ", lls)

	return np.average(lls), np.var(lls, ddof=1), lls

def NB_avg_var_loss(load_path, no_sampl):
	# batch vs time
	path_prefix = '../data/TB_interplay/'
	B =  (np.arange(0, 11)*10 + 2).astype(int) # range 2 - 102
	T =  (np.arange(0, 11)*10 + 2).astype(int) # range 2 - 102

	# storage
	NbT_avg_loss = np.zeros((11, 11))
	NbT_var_loss = np.zeros((11, 11))
	NbT_full = np.zeros((11, 11, no_sampl))

	for i, b in enumerate(B):
		for j, t in enumerate(T):
			key = jax.random.PRNGKey(0)
			av, va, allofthem = loss_avg_var(key, load_path, no_sampl, t, b)
			NbT_avg_loss[i, j] = av
			NbT_var_loss[i, j] = va
			NbT_full[i, j, :] = allofthem

			np.save('../data/initial_tests_fig/NbT_avg_loss.npy', NbT_avg_loss)
			np.save('../data/initial_tests_fig/NbT_var_loss.npy', NbT_var_loss)
			np.save('../data/initial_tests_fig/NbT_full.npy', NbT_full)

def wl_loss_avg_var(key, load_path, no_sampl, w, l):
	lls = np.zeros((no_sampl,))
	conf = iconf.get_WL()
	
	# set time and batch size
	conf.batch_size = 50
	conf.T = 50
	conf.t_vector_increment = 7500

	conf.hid_channels = int(w)
	conf.layers = int(l)

	try:
		out = "W{}L{}/final/".format(int(w), int(l))
		training_losses = np.load(load_path + out + 'loss.npy')
		inp = load_path + out
		bench = tr.Trainer(experiment_name="temp", config=conf, output_dir=out)
		dir_path = load_path + out
		bench.load_from_chp(dir_path)

	except FileNotFoundError: # execution halted due to infinite gradients or some other reason
		out = "W{}L{}/checkpoints/".format(int(w), int(l))
		training_losses = np.load(load_path + out + 'loss.npy')
		inp = load_path + out
		bench = tr.Trainer(experiment_name="temp", config=conf, output_dir=out)
		dir_path = load_path + out
		bench.load_from_chp(dir_path)
	
	bench.setup_experiment()
	print(dir_path)
	params, model, tx, state, loss_, valids_, epoch_start = bench.init_training(prngn=1)

	for i in range(no_sampl):
		start_time = time.time()
		S0 = bench.initialise_lattice(key)

		# get trajectory
		times, flips, it = bench.get_trajectory(model, params, key, S0)
		times, flips = times[:it, :], flips[:it, :]

		Ts, Fs = bench.get_batch(key, times, flips)

		key, subkey = jax.random.split(key)
		trajectories =  bench.flip_to_trajectory(S0, jnp.shape(Ts)[1], Fs)

		lls[i], eest_ = bench.lossf(model, params, trajectories, Ts, Fs)		
		looptime = time.time() - start_time
		print("W-{}, L-{}: i = {}, it = {}, took: {}".format(w, l, i, it, looptime))

	# print("Last training step loss was, {}".format(training_losses[-1]))
	# print("10 more samples ", lls)

	return np.average(lls), np.var(lls, ddof=1), lls

def WL_avg_var_loss(load_path, no_sampl):
	# batch vs time
	path_prefix = '../data/WL_interplay/'
	widths =  (np.arange(11)*2 + 1).astype(int)
	layers =  (np.arange(11) + 2).astype(int)

	# storage
	WL_avg_loss = np.zeros((11, 11))
	WL_var_loss = np.zeros((11, 11))
	WL_full = np.zeros((11, 11, no_sampl))

	for i, w in enumerate(widths):
		for j, l in enumerate(layers):
			key = jax.random.PRNGKey(0)
			av, va, allofthem = wl_loss_avg_var(key, load_path, no_sampl, w, l)
			WL_avg_loss[i, j] = av
			WL_var_loss[i, j] = va
			WL_full[i, j, :] = allofthem

			np.save('../data/initial_tests_fig/WL_avg_loss.npy', WL_avg_loss)
			np.save('../data/initial_tests_fig/WL_var_loss.npy', WL_var_loss)
			np.save('../data/initial_tests_fig/WL_full.npy', WL_full)

def wl_times_size(load_path):
	WL_timeapprox = np.zeros((11, 11))
	WL_sizeapprox = np.zeros((11, 11))
	WL_final_loss = np.zeros((11, 11))
	
	path_prefix = '../data/WL_interplay/'
	widths =  (np.arange(11)*2 + 1).astype(int)
	layers =  (np.arange(11) + 2).astype(int)

	for i, w in enumerate(widths):
		for j, l in enumerate(layers):
			try:
				inp = load_path + "W{}L{}/final/valids.npy".format(int(w), int(l))
				valids = np.load(inp)
				fl = np.load(load_path + "W{}L{}/checkpoints/loss.npy".format(int(w), int(l)))
				fl = np.trim_zeros(fl)
				times = np.average(np.trim_zeros(valids[:, 1]))
				storage = np.average(np.trim_zeros(valids[:, 2]))
			except FileNotFoundError: # execution halted due to infinite gradients or some other reason
				inp = load_path + "W{}L{}/checkpoints/valids.npy".format(int(w), int(l))
				valids = np.load(inp)
				fl = np.load(load_path + "W{}L{}/checkpoints/loss.npy".format(int(w), int(l)))
				fl = np.trim_zeros(fl)
				times = np.average(np.trim_zeros(valids[:, 1]))
				storage = np.average(np.trim_zeros(valids[:, 2]))

			WL_timeapprox[i, j] = times
			WL_sizeapprox[i, j] = storage
			WL_final_loss[i, j] = np.average(fl[-4:-1])

	np.save('../data/initial_tests_fig/WL_fl.npy', WL_final_loss)
	np.save('../data/initial_tests_fig/WL_timeapprox.npy', WL_timeapprox)
	np.save('../data/initial_tests_fig/WL_sizeapprox.npy', WL_sizeapprox)

	return WL_final_loss

def nb_times_size(load_path):
	NbT_timeapprox = np.zeros((11, 11))
	NbT_sizeapprox = np.zeros((11, 11))
	# NbT_sizeapprox_var = np.zeros((11, 11))
	NbT_final_loss = np.zeros((11, 11))
	
	B =  (np.arange(0, 11)*10 + 2).astype(int) # range 2 - 102
	T =  (np.arange(0, 11)*10 + 2).astype(int) # range 2 - 102

	for i, b in enumerate(B):
		for j, t in enumerate(T):
			inp = load_path + "Nb{}T{}/final/valids.npy".format(int(b), int(t))
			valids = np.load(inp)
			fl = np.load(load_path + "Nb{}T{}/checkpoints/loss.npy".format(int(b), int(t)))

			times = np.average(valids[:, 1])
			storage = np.average(valids[:, 2])
			# storage_var = np.var(valids[:, 2])
			NbT_timeapprox[i, j] = times
			NbT_sizeapprox[i, j] = storage
			NbT_final_loss[i, j] = np.average(fl[-4:-1])


	np.save('../data/initial_tests_fig/NbT_fl.npy', NbT_final_loss)
	np.save('../data/initial_tests_fig/NbT_timeapprox.npy', NbT_timeapprox)
	np.save('../data/initial_tests_fig/NbT_sizeapprox.npy', NbT_sizeapprox)
	# np.save('../data/initial_tests_fig/NbT_sizeapproxvar.npy', NbT_sizeapprox_var)

	return NbT_final_loss

if __name__ == '__main__':
	tb_load_path = '../data/TB_interplay/'
	wl_load_path = '../data/WL_interplay/'
	no_sampl = 10

	wl_times_size(wl_load_path)
	nb_times_size(tb_load_path)

	NB_avg_var_loss(tb_load_path, no_sampl)	
	WL_avg_var_loss(wl_load_path, no_sampl)
