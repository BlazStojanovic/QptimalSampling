import matplotlib.pyplot as plt
import numpy as np

import matplotlib
from matplotlib import rc
matplotlib.rcParams.update({'font.size': 22})
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{libertine} \usepackage{amsfonts}'

valids16 = np.load("../data/ising_2d_meeting1_fig/nb16/final/valids.npy")
valids32 = np.load("../data/ising_2d_meeting1_fig/nb32/final/valids.npy")
valids64 = np.load("../data/ising_2d_meeting1_fig/nb64/final/valids.npy")
valids64l20 = np.load("../data/ising_2d_meeting1_fig/nb64l20/final/valids.npy")
valids96 = np.load("../data/ising_2d_meeting1_fig/nb96/checkpoints/valids.npy")

l16 = np.load("../data/ising_2d_meeting1_fig/nb16/final/loss.npy")
l32 = np.load("../data/ising_2d_meeting1_fig/nb32/final/loss.npy")
l64 = np.load("../data/ising_2d_meeting1_fig/nb64/final/loss.npy")
l64l20 = np.load("../data/ising_2d_meeting1_fig/nb64l20/final/loss.npy")
l96 = np.load("../data/ising_2d_meeting1_fig/nb96/checkpoints/loss.npy")

# Evmc = -19.13084

# fig = plt.figure(figsize=(12, 6))
# plt.plot(np.arange(len(E8)), E8, color='blue', linewidth='2', label="Nb=8")
# plt.plot(np.arange(len(E32)), E16, color='red', linewidth='2', label="Nb=16")
# plt.plot(np.arange(len(E32)), E32, color='green', linewidth='2', label="Nb=32")
# plt.plot([0, 100], [Evmc, Evmc], "--k")
# plt.legend()
# plt.show()

fig = plt.figure(figsize=(12, 6))
plt.plot(np.arange(len(l10)), l16, 'o--', color='blue', linewidth='1', label="Nb=16")
plt.plot(np.arange(len(l32)), l32, 'o--', color='red', linewidth='1', label="Nb=32")
plt.plot(np.arange(len(l64)), l64, 'o--', color='green', linewidth='1', label="Nb=64")
plt.plot(np.arange(len(l96)), l96, 'o--', color='purple', linewidth='1', label="Nb=96")
plt.yscale('log')
# plt.xscale('log')
plt.legend()
plt.ylabel(r'${\mathrm{Var}}_{\mathbb{P}_v}\left[ \log \frac{\mathrm{d} \mathbb{P}_v}{\mathrm{d} \mathbb{P}_{\mathrm{FK}}} \right]$')
plt.xlabel('epoch no.')
plt.savefig("../figures/meetin1.pdf", bbox_inches='tight')
plt.show()