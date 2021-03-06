import netket as nk
import numpy as np
import matplotlib.pyplot as plt

import json

def get_obs(j, L=12):
	g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)

	# Hilbert space of spins on the graph
	hi = nk.hilbert.Spin(s=1/2, N=g.n_nodes)

	# Ising spin hamiltonian
	ha = nk.operator.Ising(hilbert=hi, graph=g, h=j, J=-1.0)

	# RBM Spin Machine
	ma = nk.models.RBM(alpha=1, use_visible_bias=True, dtype=float)

	# Metropolis Local Sampling
	sa = nk.sampler.MetropolisLocal(hi, n_chains=16, reset_chains=False)

	# Optimizer
	op = nk.optimizer.Sgd(learning_rate=0.1)

	# Variational monte carlo driver
	gs = nk.VMC(ha, op, sa, ma, n_samples=2000)

	# Create a JSON output file, and overwrite if file exists
	logger = nk.logging.JsonLog("../data/tfim1d_netket/tfim1d_fs_L{}-{}.json".format(L, int(j*100)), "w")

	sigz = nk.operator.LocalOperator(hi)
	sigx = nk.operator.LocalOperator(hi)

	for i in range(L):
		sigz += nk.operator.spin.sigmaz(hi, i)/L
		sigx += nk.operator.spin.sigmax(hi, i)/L

	obs = {'sigz': sigz, 'sigx': sigx}

	# Run the optimization
	gs.run(n_iter=1000, out=logger, obs=obs)

def tfim1d_comp_run(L, N):
	ratios = np.linspace(0.3, 1.7, N)
	for j in ratios:
		get_obs(j, L=L)

def tfim1d_comp_save(L, N):
	ratios = np.linspace(0.3, 1.7, N)
	outs = np.zeros((7, N))
	outs[0, :] = ratios

	for it, j in enumerate(ratios):
		f = "../data/tfim1d_netket/tfim1d_fs_L{}-{}.json.log".format(L, int(j*100))
		data=json.load(open(f))
			
		iters = data['Energy']['iters']
		energy = data['Energy']['Mean']
		sigma = data['Energy']['Sigma']
		# todo add others
		sigz = data['sigz']['Mean']
		sigzvar = data['sigz']['Sigma']
		sigx = data['sigx']['Mean']
		sigxvar = data['sigx']['Sigma']

		# plt.plot(energy, 'r')
		# plt.show()
		# plt.plot(iters, sigz)
		# plt.show()

		# print(sigz)

		# print(len(energy[-31:-1]))
		outs[1, it] = sum(energy[-31:-1])/30
		# outs[2, it] = sigma[-1]
		# print(sum(sigz[-10:-1])/10)
		outs[3, it] = sum(sigz[-31:-1])/30
		# outs[4, it] = np.average(sigzvar[-10:-1])
		outs[5, it] = sum(sigx[-31:-1])/30
		# outs[6, it] = np.average(sigxvar[-10:-1])

	np.save('../data/tfim1d_netket/fs-{}.npy'.format(L), outs)


if __name__ == '__main__':
	
	L = [3, 6, 12]
	N = 20

	for l in L:
		tfim1d_comp_run(l, N)
		tfim1d_comp_save(l, N)
