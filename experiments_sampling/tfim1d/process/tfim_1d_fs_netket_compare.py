import netket as nk
import numpy as np
import matplotlib.pyplot as plt

import json

def get_obs(j, h=1, length=12):
	g = nk.graph.Hypercube(length=length, n_dim=1, pbc=True)

	# Hilbert space of spins on the graph
	hi = nk.hilbert.Spin(s=1/2, N=g.n_nodes)

	# Ising spin hamiltonian at the critical point
	ha = nk.operator.Ising(hilbert=hi, graph=g, h=j, J=-1)

	# RBM Spin Machine
	ma = nk.models.RBM(alpha=1, use_visible_bias=True, dtype=float)

	# Metropolis Local Sampling
	sa = nk.sampler.MetropolisLocal(hi, n_chains=16)

	# The variational state
	vs = nk.variational.MCState(sa, ma, n_samples=1000, n_discard_per_chain=100)
	vs.init_parameters(nk.nn.initializers.normal(stddev=0.01), seed=1234)

	# Optimizer
	op = nk.optimizer.Sgd(learning_rate=0.1)

	# Stochastic Reconfiguration
	sr = nk.optimizer.SR(diag_shift=0.1)

	# Variational monte carlo driver
	gs = nk.VMC(ha, op, variational_state=vs, preconditioner=sr)

	# Create a JSON output file, and overwrite if file exists
	logger = nk.logging.JsonLog("../data/tfim1d_netket/tfim1d_fs_{}.json".format(int(j*100)), "w")

	sigz = nk.operator.LocalOperator(hi)

	for i in range(length):
		sigz += 2*nk.operator.spin.sigmaz(hi, i)-1

	sigz = sigz/length

	# Run the optimization
	gs.run(n_iter=1000, out=logger, obs={'sigz': sigz})

if __name__ == '__main__':
	L = 12
	ratios = np.linspace(0.0, 2, 10)
	outs = np.zeros((7, 10))	
	outs[0, :] = ratios

	for j in ratios:
		get_obs(j)

	for it, j in enumerate(ratios):
		f = "../data/tfim1d_netket/tfim1d_fs_{}.json.log".format(int(j*100))
		data=json.load(open(f))
			
		iters = data['Energy']['iters']
		energy = data['Energy']['Mean']
		sigma = data['Energy']['Sigma']
		# todo add others
		sigz = data['sigz']['Mean']
		sigzvar = data['sigz']['Sigma']

		outs[1, it] = energy[-1]
		outs[2, it] = sigma[-1]
		outs[3, it] = sigz[-1]
		outs[4, it] = sigzvar[-1]

	np.save('../data/tfim1d_netket/fs-{}.npy'.format(L), outs)

