import matplotlib.pyplot as plt
import numpy as np

import matplotlib
from matplotlib import colors
from matplotlib import rc
from mpl_toolkits.axes_grid1 import make_axes_locatable

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{libertine} \usepackage{amsfonts}'

# hparams searched
widths =  (np.arange(11)*2 + 1).astype(int)
layers =  (np.arange(11) + 2).astype(int)
b =  (np.arange(0, 11)*10 + 2).astype(int)
t =  (np.arange(0, 11)*10 + 2).astype(int)
tf = t.astype(float)

time_nb = np.load('../data/initial_tests_fig/NbT_timeapprox.npy')
time_wl = np.load('../data/initial_tests_fig/WL_timeapprox.npy')
iter_nb = np.load('../data/initial_tests_fig/NbT_sizeapprox.npy')
iter_wl = np.load('../data/initial_tests_fig/WL_sizeapprox.npy')
avg_nb = np.load('../data/initial_tests_fig/NbT_avg_loss.npy')
avg_wl = np.load('../data/initial_tests_fig/WL_avg_loss.npy')
var_nb = np.load('../data/initial_tests_fig/NbT_var_loss.npy')
var_wl = np.load('../data/initial_tests_fig/WL_var_loss.npy')
nb_full = np.load('../data/initial_tests_fig/NbT_full.npy')
wl_full = np.load('../data/initial_tests_fig/WL_full.npy')
fl_nb = np.load('../data/initial_tests_fig/NbT_fl.npy')
fl_wl = np.load('../data/initial_tests_fig/WL_fl.npy')

def plot_grid():
	matplotlib.rcParams.update({'font.size': 16})

	fig, ax = plt.subplots(2, 3)
	fig.set_size_inches(16, 10)

	# fl plots
	Z = fl_nb*np.reciprocal(tf)*50 # magic numbers: 50 -> time of testing
	a00 = ax[0, 0].imshow(Z, cmap='Blues', norm=colors.LogNorm(vmin=(Z).min(), vmax=(Z).max()))
	divider = make_axes_locatable(ax[0, 0])
	cax = divider.append_axes('right', size='5%', pad=0.05)
	fig.colorbar(a00, cax=cax, orientation='vertical')

	Z = fl_wl*np.reciprocal(30.0)*50 # magic numbers: 30 -> time when training, 50 -> time of testing
	a10 = ax[1, 0].imshow(Z, cmap='Blues', norm=colors.LogNorm(vmin=(Z).min(), vmax=(Z).max())) 
	divider = make_axes_locatable(ax[1, 0])
	cax = divider.append_axes('right', size='5%', pad=0.05)
	fig.colorbar(a10, cax=cax, orientation='vertical')

	# avg plots
	a01 = ax[0, 1].imshow(avg_nb, cmap='Blues', norm=colors.LogNorm(vmin=(avg_nb).min(), vmax=(avg_nb).max()))
	divider = make_axes_locatable(ax[0, 1])
	cax = divider.append_axes('right', size='5%', pad=0.05)
	fig.colorbar(a01, cax=cax, orientation='vertical')

	a11 = ax[1, 1].imshow(avg_wl, cmap='Blues', norm=colors.LogNorm(vmin=(avg_wl).min(), vmax=(avg_wl).max()))
	divider = make_axes_locatable(ax[1, 1])
	cax = divider.append_axes('right', size='5%', pad=0.05)
	fig.colorbar(a11, cax=cax, orientation='vertical')

	# var plots
	a02 = ax[0, 2].imshow(var_nb, cmap='Blues', norm=colors.LogNorm(vmin=(var_nb).min(), vmax=(var_nb).max()))
	divider = make_axes_locatable(ax[0, 2])
	cax = divider.append_axes('right', size='5%', pad=0.05)
	fig.colorbar(a02, cax=cax, orientation='vertical')

	a12 = ax[1, 2].imshow(var_wl, cmap='Blues', norm=colors.LogNorm(vmin=(var_wl).min(), vmax=(var_wl).max()))
	divider = make_axes_locatable(ax[1, 2])
	cax = divider.append_axes('right', size='5%', pad=0.05)
	fig.colorbar(a12, cax=cax, orientation='vertical')

	ax[0, 0].set_xticks(np.arange(11))
	ax[0, 0].set_xticklabels(t)
	ax[0, 0].set_yticks(np.arange(11))
	ax[0, 0].set_yticklabels(b)
	ax[0, 1].set_xticks(np.arange(11))
	ax[0, 1].set_xticklabels(t)
	ax[0, 1].set_yticks(np.arange(11))
	ax[0, 1].set_yticklabels(b)
	ax[0, 2].set_xticks(np.arange(11))
	ax[0, 2].set_xticklabels(t)
	ax[0, 2].set_yticks(np.arange(11))
	ax[0, 2].set_yticklabels(b)

	ax[1, 0].set_xticks(np.arange(11))
	ax[1, 0].set_xticklabels(layers)
	ax[1, 0].set_yticks(np.arange(11))
	ax[1, 0].set_yticklabels(widths)
	ax[1, 1].set_xticks(np.arange(11))
	ax[1, 1].set_xticklabels(layers)
	ax[1, 1].set_yticks(np.arange(11))
	ax[1, 1].set_yticklabels(widths)
	ax[1, 2].set_xticks(np.arange(11))
	ax[1, 2].set_xticklabels(layers)
	ax[1, 2].set_yticks(np.arange(11))
	ax[1, 2].set_yticklabels(widths)

	ax[0, 0].set_ylabel('$N_b$')
	ax[0, 0].set_xlabel('$T$')
	ax[0, 1].set_xlabel('$T$')
	ax[0, 2].set_xlabel('$T$')

	ax[1, 0].set_ylabel('$N_w$')
	ax[1, 0].set_xlabel('$N_l$')
	ax[1, 1].set_xlabel('$N_l$')
	ax[1, 2].set_xlabel('$N_l$')

	ax[0, 0].set_title('Training loss, $N_b$ vs $T$')
	ax[1, 0].set_title('Training loss, $N_w$ vs $N_l$')
	ax[0, 1].set_title('Average test loss, $N_b$ vs $T$')
	ax[1, 1].set_title('Average test loss, $N_w$ vs $N_l$')
	ax[0, 2].set_title('Variance of test loss, $N_b$ vs $T$')
	ax[1, 2].set_title('Variance of test loss, $N_w$ vs $N_l$')

	# axis sharing
	ax[0, 0].sharey(ax[0, 1])
	ax[0, 1].sharey(ax[0, 2])
	ax[1, 0].sharey(ax[1, 1])
	ax[1, 1].sharey(ax[1, 2])

	plt.setp(ax[0, 0].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
	plt.setp(ax[0, 1].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
	plt.setp(ax[0, 2].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
	plt.setp(ax[1, 0].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
	plt.setp(ax[1, 1].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
	plt.setp(ax[1, 2].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

	plt.subplots_adjust(wspace=0.3, hspace=0.35)
	plt.savefig("../../../Thesis/Chapter5/Figs/Raster/avg_var_loss.png", bbox_inches='tight')
	plt.savefig("../figures/avg_var_loss.png", bbox_inches='tight')
	plt.show()


def plot_times_size():
	matplotlib.rcParams.update({'font.size': 22})

	# get times - NbT 
	times = time_nb
	Nbnorm_times = times*np.reciprocal(times[0, :])
	Tnorm_times = times.T*np.reciprocal(times[:, 0])

	avg_nb_times = np.average(Nbnorm_times, axis=1)
	nb_times_err = np.sqrt(np.var(Nbnorm_times, axis=1, ddof=1)/11)

	avg_T_times = np.average(Tnorm_times, axis=1)
	T_times_err = np.sqrt(np.var(Tnorm_times, axis=1, ddof=1)/11)

	# get times - WL
	times = time_wl
	Wnorm_times = times*np.reciprocal(times[0, :])
	Lnorm_times = times.T*np.reciprocal(times[:, 0])

	avg_W_times = np.average(Wnorm_times, axis=1)
	W_times_err = np.sqrt(np.var(Wnorm_times, axis=1)/11)

	avg_L_times = np.average(Lnorm_times, axis=1)
	L_times_err = np.sqrt(np.var(Lnorm_times, axis=1)/11)

	fig, ax = plt.subplots(2, 2)
	fig.set_size_inches(16, 10)

	# time plots
	ax[0, 0].set_ylabel("epoch time [a.u.]")
	ax[0, 0].set_xlabel("parameter size")
	ax[0, 0].plot(b[:-2], avg_nb_times[:-2], 'o--', color='blue', linewidth=2, label='$N_b$')
	ax[0, 0].fill_between(b[:-2], avg_nb_times[:-2] - nb_times_err[:-2], avg_nb_times[:-2] + nb_times_err[:-2], alpha = 0.2, color='blue')
	ax[0, 0].plot(t[:-2], avg_T_times[:-2], 'o--', color='red', linewidth=2, label='$T$')
	ax[0, 0].fill_between(b[:-2], avg_T_times[:-2] - T_times_err[:-2], avg_T_times[:-2] + T_times_err[:-2], alpha = 0.2, color='red')
	ax[0, 0].legend()

	ax[0, 1].set_ylabel("epoch time [a.u.]")
	ax[0, 1].set_xlabel("parameter size")
	ax[0, 1].plot(widths, avg_W_times, 'o--', color='blue', linewidth=2, label='$N_w$')
	ax[0, 1].fill_between(widths, avg_W_times - W_times_err, avg_W_times + W_times_err, alpha = 0.2, color='blue')
	ax[0, 1].plot(layers, avg_L_times, 'o--', color='red', linewidth=2, label='$N_l$')
	ax[0, 1].fill_between(layers, avg_L_times - L_times_err, avg_L_times + L_times_err, alpha = 0.2, color='red')
	ax[0, 1].legend()
	

	# get sizes - WL 
	iters = iter_nb
	size_b = b/b[0]
	size_t = iter_nb[0]/iter_nb[0, 0]

	iters = iter_wl
	Wnorm_iters = iters*np.reciprocal(iters[0, :])
	Lnorm_iters = iters.T*np.reciprocal(iters[:, 0])

	avg_W_iters = np.average(Wnorm_iters, axis=1)
	W_iters_err = np.sqrt(np.var(Wnorm_iters, axis=1)/11)

	avg_L_iters = np.average(Lnorm_iters, axis=1)
	L_iters_err = np.sqrt(np.var(Lnorm_iters, axis=1)/11)

	# time plots
	ax[1, 0].set_ylabel("batch size [a.u.]")
	ax[1, 0].set_xlabel("parameter size")
	ax[1, 0].plot(b[:-2], size_b[:-2], 'o--', color='blue', linewidth=2, label='$N_b$')
	ax[1, 0].plot(t[:-2], size_t[:-2], 'o--', color='red', linewidth=2, label='$T$')
	ax[1, 0].legend()

	ax[1, 1].set_ylabel("batch size [a.u.]")
	ax[1, 1].set_xlabel("parameter size")
	ax[1, 1].plot(widths, avg_W_iters, 'o--', color='blue', linewidth=2, label='$N_w$')
	ax[1, 1].fill_between(widths, avg_W_iters - W_iters_err, avg_W_iters + W_iters_err, alpha = 0.2, color='blue')
	ax[1, 1].plot(layers, avg_L_iters, 'o--', color='red', linewidth=2, label='$N_l$')
	ax[1, 1].fill_between(layers, avg_L_iters - L_iters_err, avg_L_iters + L_iters_err, alpha = 0.2, color='red')
	ax[1, 1].legend()

	# spacing
	plt.subplots_adjust(wspace=0.3, hspace=0.25)

	# axis sharing
	ax[0, 0].sharey(ax[0, 1])
	ax[0, 0].sharex(ax[1, 0])
	ax[1, 1].sharex(ax[0, 1])

	plt.savefig("../../../Thesis/Chapter5/Figs/Raster/initial_time_space.png", bbox_inches='tight')
	plt.savefig("../figures/initial_time_space.png", bbox_inches='tight')
	plt.show()

if __name__ == '__main__':
	plot_grid()
	plot_times_size()