
import matplotlib.pyplot as plt
import numpy as np

import matplotlib
from matplotlib import rc
matplotlib.rcParams.update({'font.size': 22})
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{libertine} \usepackage{amsfonts}'

import sys
sys.path.append('../')

import tfim1d.tfim_1d_analytical

jg = np.linspace(0, 4, 500)
en = tfim1d.tfim_1d_analytical.energy_per_site(jg)

fig = plt.figure(figsize=(8, 8))
plt.plot(jg, en, 'k--', label=r'analytical, $N = \infty$')
plt.axvline(x=2, linewidth=1, color='k')
plt.ylabel(r"$E_0/N$")
plt.xlabel(r"$J/g$")

plt.legend()
plt.savefig("../figures/tfim1d_en_finite_scaling.pdf", bbox_inches='tight')
plt.savefig("../../../Thesis/Chapter5/Figs/Vector/tfim1d_en_finite_scaling.pdf", bbox_inches='tight')
plt.show()