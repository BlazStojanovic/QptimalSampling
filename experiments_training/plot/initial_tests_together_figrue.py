import matplotlib.pyplot as plt
import numpy as np

import matplotlib
from matplotlib import rc
matplotlib.rcParams.update({'font.size': 22})
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{libertine} \usepackage{amsfonts}'

fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
fig.set_size_inches(16, 10)

ax[0, 0].set_title("Batch size $N_b$ vs. loss")
ax[0, 1].set_title("Time $T$ vs. loss")
ax[1, 0].set_title("No. filters $N_w$ vs. loss")
ax[1, 1].set_title("No. layers $N_l$ vs. loss")

v3 = np.load("../data/width_vs_loss/wid3/final/valids.npy")
v9 = np.load("../data/width_vs_loss/wid9/final/valids.npy")
v15 = np.load("../data/width_vs_loss/wid15/checkpoints/valids.npy")

l3 = np.load("../data/width_vs_loss/wid3/final/loss.npy")
l9 = np.load("../data/width_vs_loss/wid9/final/loss.npy")
l15 = np.load("../data/width_vs_loss/wid15/checkpoints/loss.npy")

ax[1 ,0].plot(np.arange(len(l3)) + 1, l3, '.--', color='blue', linewidth='1', label="$N_w=3$")
ax[1 ,0].plot(np.arange(len(l9)) + 1, l9, '.--', color='purple', linewidth='1', label="$N_w=9$")
ax[1 ,0].plot(np.arange(len(l15[:-31])) + 1, l15[:-31], '.--', color='red', linewidth='1', label="$N_w=15$")

ax[1 ,0].set_yscale('log')
ax[1 ,0].legend()
ax[1 ,0].set_ylabel(r'${\mathrm{Var}}_{\mathbb{P}_v}\left[ \log \frac{\mathrm{d} \mathbb{P}_v}{\mathrm{d} \mathbb{P}_{\mathrm{FK}}} \right]$')
ax[1 ,0].set_xlabel('epoch no.')

v3 = np.load("../data/layer_vs_loss/lay3/final/valids.npy")
v9 = np.load("../data/layer_vs_loss/lay9/final/valids.npy")
v15 = np.load("../data/layer_vs_loss/lay15/final/valids.npy")

l3 = np.load("../data/layer_vs_loss/lay3/final/loss.npy")
l9 = np.load("../data/layer_vs_loss/lay9/final/loss.npy")
l15 = np.load("../data/layer_vs_loss/lay15/final/loss.npy")

ax[1 ,1].plot(np.arange(len(l3)) + 1, l3, '.--', color='blue', linewidth='1', label="$N_l=3$")
ax[1 ,1].plot(np.arange(len(l9)) + 1, l9, '.--', color='purple', linewidth='1', label="$N_l=9$")
ax[1 ,1].plot(np.arange(len(l15)) + 1, l15, '.--', color='red', linewidth='1', label="$N_l=15$")

ax[1 ,1].set_yscale('log')
ax[1 ,1].legend()
ax[1 ,1].set_xlabel('epoch no.')

v10 = np.load("../data/batch_vs_loss/nb10/final/valids.npy")
v40 = np.load("../data/batch_vs_loss/nb40/final/valids.npy")
v70 = np.load("../data/batch_vs_loss/nb70/final/valids.npy")

l10 = np.load("../data/batch_vs_loss/nb10/final/loss.npy")
l40 = np.load("../data/batch_vs_loss/nb40/final/loss.npy")
l70 = np.load("../data/batch_vs_loss/nb70/final/loss.npy")

ax[0 ,0].plot(np.arange(len(l10)) + 1, l10, '.--', color='blue', linewidth='1', label="$N_b=10$")
ax[0 ,0].plot(np.arange(len(l40)) + 1, l40, '.--', color='red', linewidth='1', label="$N_b=40$")
ax[0 ,0].plot(np.arange(len(l70)) + 1, l70, '.--', color='purple', linewidth='1', label="$N_b=70$")
ax[0 ,0].set_yscale('log')
ax[0 ,0].legend()
ax[0 ,0].set_ylabel(r'${\mathrm{Var}}_{\mathbb{P}_v}\left[ \log \frac{\mathrm{d} \mathbb{P}_v}{\mathrm{d} \mathbb{P}_{\mathrm{FK}}} \right]$')

v10 = np.load("../data/time_vs_loss/T10/final/valids.npy")
v50 = np.load("../data/time_vs_loss/T50/final/valids.npy")
v100 = np.load("../data/time_vs_loss/T100/final/valids.npy")

l10 = np.load("../data/time_vs_loss/T10/final/loss.npy")
l50 = np.load("../data/time_vs_loss/T50/final/loss.npy")
l100 = np.load("../data/time_vs_loss/T100/final/loss.npy")

ax[0 ,1].plot(np.arange(len(l10)) + 1, l10, '.--', color='blue', linewidth='1', label="$T=10$")
ax[0 ,1].plot(np.arange(len(l50)) + 1, l50, '.--', color='red', linewidth='1', label="$T=50$")
ax[0 ,1].plot(np.arange(len(l100)) + 1, l100, '.--', color='purple', linewidth='1', label="$T=100$")
ax[0 ,1].set_yscale('log')
ax[0 ,1].legend()

plt.subplots_adjust(wspace=0.05, hspace=0.2)
plt.savefig("../figures/init_test_learning.pdf", bbox_inches='tight')
plt.savefig("../../../Thesis/Chapter5/Figs/Vector/init_test_learning.pdf", bbox_inches='tight')
plt.show()
