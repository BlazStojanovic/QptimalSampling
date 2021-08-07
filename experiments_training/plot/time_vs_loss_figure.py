import matplotlib.pyplot as plt
import numpy as np

import matplotlib
from matplotlib import rc
matplotlib.rcParams.update({'font.size': 22})
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{libertine} \usepackage{amsfonts}'


v10 = np.load("../data/time_vs_loss/T10/final/valids.npy")
v50 = np.load("../data/time_vs_loss/T50/final/valids.npy")
v100 = np.load("../data/time_vs_loss/T100/final/valids.npy")
# v150 = np.load("../data/time_vs_loss/T150/final/valids.npy")
# v200 = np.load("../data/time_vs_loss/T200/final/valids.npy")

l10 = np.load("../data/time_vs_loss/T10/final/loss.npy")
l50 = np.load("../data/time_vs_loss/T50/final/loss.npy")
l100 = np.load("../data/time_vs_loss/T100/final/loss.npy")
# l150 = np.load("../data/time_vs_loss/T150/final/loss.npy")
# l200 = np.load("../data/time_vs_loss/T200/final/loss.npy")

Evmc = -19.13084

fig = plt.figure(figsize=(12, 6))
plt.plot(np.arange(len(l10)) + 1, l10, 'o--', color='blue', linewidth='1', label="T=10")
plt.plot(np.arange(len(l50)) + 1, l50, 'o--', color='red', linewidth='1', label="T=50")
plt.plot(np.arange(len(l100)) + 1, l100, 'o--', color='green', linewidth='1', label="T=100")
# plt.plot(np.arange(len(l150)) + 1, l150, 'o--', color='purple', linewidth='1', label="T=150")
# plt.plot(np.arange(len(l200)) + 1, l200, 'o--', color='purple', linewidth='1', label="T=200")
plt.yscale('log')
# plt.xscale('log')
plt.legend()
plt.ylabel(r'${\mathrm{Var}}_{\mathbb{P}_v}\left[ \log \frac{\mathrm{d} \mathbb{P}_v}{\mathrm{d} \mathbb{P}_{\mathrm{FK}}} \right]$')
plt.xlabel('epoch no.')
plt.savefig("../figures/tvl_single.pdf", bbox_inches='tight')
plt.savefig("../../../Thesis/Chapter5/Figs/Vector/tvl_single.pdf", bbox_inches='tight')
plt.show()
