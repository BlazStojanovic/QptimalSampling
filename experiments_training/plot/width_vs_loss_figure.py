import matplotlib.pyplot as plt
import numpy as np

import matplotlib
from matplotlib import rc
matplotlib.rcParams.update({'font.size': 22})
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{libertine} \usepackage{amsfonts}'


v3 = np.load("../data/width_vs_loss/wid3/final/valids.npy")
v9 = np.load("../data/width_vs_loss/wid9/final/valids.npy")
v15 = np.load("../data/width_vs_loss/wid15/checkpoints/valids.npy")

l3 = np.load("../data/width_vs_loss/wid3/final/loss.npy")
l9 = np.load("../data/width_vs_loss/wid9/final/loss.npy")
l15 = np.load("../data/width_vs_loss/wid15/checkpoints/loss.npy")

print(v3, v9, v15)
Evmc = -19.13084

fig = plt.figure(figsize=(12, 6))
plt.plot(np.arange(len(l3)) + 1, l3, 'h--', color='blue', linewidth='1', label="k=3")
plt.plot(np.arange(len(l9)) + 1, l9, 'o--', color='green', linewidth='1', label="k=9")
plt.plot(np.arange(len(l15[:-31])) + 1, l15[:-31], 's--', color='red', linewidth='1', label="k=15")

plt.yscale('log')
# plt.xscale('log')
plt.legend()
plt.ylabel(r'${\mathrm{Var}}_{\mathbb{P}_v}\left[ \log \frac{\mathrm{d} \mathbb{P}_v}{\mathrm{d} \mathbb{P}_{\mathrm{FK}}} \right]$')
plt.xlabel('epoch no.')
plt.savefig("../figures/wvl_single.pdf", bbox_inches='tight')
plt.savefig("../../../Thesis/Chapter5/Figs/Vector/wvl_single.pdf", bbox_inches='tight')
plt.show()
