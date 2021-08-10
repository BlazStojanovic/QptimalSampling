
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

jg2 = np.linspace(2, 4, 500)
mx2 = tfim1d.tfim_1d_analytical.m_x(jg2)
jg1 = np.linspace(0, 2, 500)
mx1 = np.zeros(500)

fig = plt.figure(figsize=(8, 8))
plt.plot(jg1, mx1, 'k--')
plt.plot(jg2, mx2, 'k--', label=r'analytical, $N = \infty$')
plt.axvline(x=2, linewidth=1, color='k')
plt.ylabel(r"$\langle \sigma_x \rangle$")
plt.xlabel(r"$J/g$")

plt.legend()
plt.savefig("../figures/tfim1d_mx_finite_scaling.pdf", bbox_inches='tight')
plt.savefig("../../../Thesis/Chapter5/Figs/Vector/tfim1d_mx_finite_scaling.pdf", bbox_inches='tight')
plt.show()