import netket as nk
import numpy as np
import matplotlib.pyplot as plt

import json

def get_obs(j, L=3):
	g = nk.graph.Hypercube(length=L, n_dim=2, pbc=True)

	# Hilbert space of spins on the graph
	hi = nk.hilbert.Spin(s=1/2, N=g.n_nodes)

	# Ising spin hamiltonian
	ha = nk.operator.Ising(hilbert=hi, graph=g, h=1.0, J=-j)

	# RBM Spin Machine
	ma = nk.models.RBM(alpha=1, use_visible_bias=True, dtype=float)

	# Metropolis Local Sampling
	sa = nk.sampler.MetropolisLocal(hi, n_chains=16, reset_chains=True)

	# Optimizer
	op = nk.optimizer.Sgd(learning_rate=0.1)

	# Variational monte carlo driver
	gs = nk.VMC(ha, op, sa, ma, n_samples=1500)

	# Create a JSON output file, and overwrite if file exists
	logger = nk.logging.JsonLog("../data/tfim2d_netket/tfim2d_fs_L{}-{}.json".format(L, int(j*100)), "w")

	sigz = nk.operator.LocalOperator(hi)
	sigx = nk.operator.LocalOperator(hi)

	for i in range(L*L):
		sigz += nk.operator.spin.sigmaz(hi, i)/L/L
		sigx += nk.operator.spin.sigmax(hi, i)/L/L

	obs = {'sigz': sigz, 'sigx': sigx}

	# Run the optimization
	gs.run(n_iter=700, out=logger, obs=obs)

def tfim2d_comp_run(L, N):
	ratios = np.linspace(0.0, 0.7, N)
	for j in ratios:
		get_obs(j, L=L)

def tfim2d_comp_save(L, N):
	ratios = np.linspace(0.0, 0.7, N)
	outs = np.zeros((7, N))
	outs[0, :] = ratios

	for it, j in enumerate(ratios):
		f = "../data/tfim2d_netket/tfim2d_fs_L{}-{}.json.log".format(L, int(j*100))
		data=json.load(open(f))
			
		iters = data['Energy']['iters']
		energy = data['Energy']['Mean']
		sigma = data['Energy']['Sigma']
		sigz = data['sigz']['Mean']
		sigzvar = data['sigz']['Sigma']
		sigx = data['sigx']['Mean']
		sigxvar = data['sigx']['Sigma']

		plt.plot(energy, 'r')
		plt.show()
		plt.plot(iters, sigz)
		plt.show()

		# print(sigz)
		# print(len(energy[-31:-1]))
		outs[1, it] = sum(energy[-101:-1])/100
		# outs[2, it] = sigma[-1]
		# print(sum(sigz[-10:-1])/10)
		outs[3, it] = sum(sigz[-101:-1])/100
		# outs[4, it] = np.average(sigzvar[-10:-1])
		outs[5, it] = sum(sigx[-101:-1])/100
		# outs[6, it] = np.average(sigxvar[-10:-1])

	np.save('../data/tfim2d_netket/fs-{}.npy'.format(L), outs)


if __name__ == '__main__':
	
	# L = [3, 6, 9]
	# L = [3, 6]
	L = [6]
	N = 15

	for l in L:
		# tfim2d_comp_run(l, N)
		tfim2d_comp_save(l, N)
