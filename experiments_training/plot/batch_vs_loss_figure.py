
import matplotlib.pyplot as plt
import numpy as np

import matplotlib
from matplotlib import rc
matplotlib.rcParams.update({'font.size': 22})
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{libertine} \usepackage{amsfonts}'


v10 = np.load("../data/batch_vs_loss/nb10/final/valids.npy")
v40 = np.load("../data/batch_vs_loss/nb40/final/valids.npy")
v70 = np.load("../data/batch_vs_loss/nb70/final/valids.npy")
# v100 = np.load("../data/batch_vs_loss/nb64l20/final/valids.npy")
# v130 = np.load("../data/batch_vs_loss/nb96/final/valids.npy")

l10 = np.load("../data/batch_vs_loss/nb10/final/loss.npy")
l40 = np.load("../data/batch_vs_loss/nb40/final/loss.npy")
l70 = np.load("../data/batch_vs_loss/nb70/final/loss.npy")
# l100 = np.load("../data/batch_vs_loss/nb64l20/final/loss.npy")
# l130 = np.load("../data/batch_vs_loss/nb96/final/loss.npy")

Evmc = -19.13084

fig = plt.figure(figsize=(12, 6))
plt.plot(np.arange(len(l10)) + 1, l10, 'o--', color='blue', linewidth='1', label="Nb=10")
plt.plot(np.arange(len(l40)) + 1, l40, 'o--', color='red', linewidth='1', label="Nb=40")
plt.plot(np.arange(len(l70)) + 1, l70, 'o--', color='green', linewidth='1', label="Nb=70")
# plt.plot(np.arange(len(l100)) + 1, l100, 'o--', color='purple', linewidth='1', label="Nb=100")
# plt.plot(np.arange(len(l130)) + 1, l130, 'o--', color='purple', linewidth='1', label="Nb=130")
plt.yscale('log')
# plt.xscale('log')
plt.legend()
plt.ylabel(r'${\mathrm{Var}}_{\mathbb{P}_v}\left[ \log \frac{\mathrm{d} \mathbb{P}_v}{\mathrm{d} \mathbb{P}_{\mathrm{FK}}} \right]$')
plt.xlabel('epoch no.')
plt.savefig("../figures/bvl_single.pdf", bbox_inches='tight')
plt.show()
