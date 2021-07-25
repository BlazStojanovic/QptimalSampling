import matplotlib.pyplot as plt
import numpy as np

import matplotlib
from matplotlib import rc
matplotlib.rcParams.update({'font.size': 22})
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{libertine}'

E8 = np.load("../../Ising/ising_rates/61/endpointloss_eest.npy")
E16 = np.load("../../Ising/ising_rates/70/endpointloss_eest.npy")
E32 = np.load("../../Ising/ising_rates/79/endpointloss_eest.npy")

print(E8)

l8 = np.load("../../Ising/ising_rates/61/endpointloss.npy")
l16 = np.load("../../Ising/ising_rates/70/endpointloss.npy")
l32 = np.load("../../Ising/ising_rates/79/endpointloss.npy")

Evmc = -19.13084

# fig = plt.figure(figsize=(12, 6))
# plt.plot(np.arange(len(E8)), E8, color='blue', linewidth='2', label="Nb=8")
# plt.plot(np.arange(len(E32)), E16, color='red', linewidth='2', label="Nb=16")
# plt.plot(np.arange(len(E32)), E32, color='green', linewidth='2', label="Nb=32")
# plt.plot([0, 100], [Evmc, Evmc], "--k")
# plt.legend()
# plt.show()

fig = plt.figure(figsize=(12, 6))
plt.plot(np.arange(len(l8)), l8, 'o--', color='blue', linewidth='2', label="Nb=8")
plt.plot(np.arange(len(l32)), l16, 'o--', color='red', linewidth='2', label="Nb=16")
plt.plot(np.arange(len(l32)), l32, 'o--', color='green', linewidth='2', label="Nb=32")
plt.yscale('log')
plt.legend()
plt.savefig("../figures/meeting1.pdf", bbox_inches='tight')
plt.show()
