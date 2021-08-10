
import matplotlib.pyplot as plt
import numpy as np

import matplotlib
from matplotlib import rc
matplotlib.rcParams.update({'font.size': 30})
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{libertine} \usepackage{amsfonts}'

import sys
sys.path.append('../')

import tfim1d.tfim_1d_analytical

jg = np.linspace(0, 4, 500)
mz = tfim1d.tfim_1d_analytical.m_z(jg)

fig = plt.figure(figsize=(8, 8))
plt.plot(jg, mz, 'k--', label=r'analytical, $N = \infty$')
plt.axvline(x=2, linewidth=1, color='k')
plt.ylabel(r"$\langle \hat \sigma_z \rangle$")
plt.xlabel(r"$J/g$")

plt.legend()
plt.savefig("../figures/tfim1d_mz_finite_scaling.pdf", bbox_inches='tight')
plt.savefig("../../../Thesis/Chapter5/Figs/Vector/tfim1d_mz_finite_scaling.pdf", bbox_inches='tight')
plt.show()