import netket as nk

# 2D Lattice
g = nk.graph.Hypercube(length=3, n_dim=2, pbc=True)

# Hilbert space of spins on the graph
hi = nk.hilbert.Spin(s=1/2, N=g.n_nodes)

# Ising spin hamiltonian at the critical point
ha = nk.operator.Ising(hilbert=hi, graph=g, h=1.0, J=-1.0)

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
logger = nk.logging.JsonLog("test", "w")

# Run the optimization
gs.run(n_iter=1000, out=logger)


