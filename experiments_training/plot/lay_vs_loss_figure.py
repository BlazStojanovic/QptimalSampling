import matplotlib.pyplot as plt
import numpy as np

import matplotlib
from matplotlib import rc
matplotlib.rcParams.update({'font.size': 22})
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{libertine} \usepackage{amsfonts}'


v3 = np.load("../data/layer_vs_loss/lay3/final/valids.npy")
v9 = np.load("../data/layer_vs_loss/lay9/final/valids.npy")
v15 = np.load("../data/layer_vs_loss/lay15/final/valids.npy")

l3 = np.load("../data/layer_vs_loss/lay3/final/loss.npy")
l9 = np.load("../data/layer_vs_loss/lay9/final/loss.npy")
l15 = np.load("../data/layer_vs_loss/lay15/final/loss.npy")

Evmc = -19.13084
print(v3)


fig = plt.figure(figsize=(12, 6))
plt.plot(np.arange(len(l3)) + 1, l3, 'o--', color='blue', linewidth='1', label="$N_l=3$")
plt.plot(np.arange(len(l9)) + 1, l9, 'o--', color='green', linewidth='1', label="$N_l=9$")
plt.plot(np.arange(len(l15)) + 1, l15, 'o--', color='red', linewidth='1', label="$N_l=15$")

plt.yscale('log')
# plt.xscale('log')
plt.legend()
plt.ylabel(r'${\mathrm{Var}}_{\mathbb{P}_v}\left[ \log \frac{\mathrm{d} \mathbb{P}_v}{\mathrm{d} \mathbb{P}_{\mathrm{FK}}} \right]$')
plt.xlabel('epoch no.')
plt.savefig("../figures/lvl_single.pdf", bbox_inches='tight')
plt.savefig("../../../Thesis/Chapter5/Figs/Vector/lvl_single.pdf", bbox_inches='tight')
plt.show()
