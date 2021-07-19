import numpy as np
import matplotlib.pyplot as plt
import json

plt.ion()

# 4x4 lattice
# exact=-3.21550807082536*16

# 5x5 lattice
exact = -80.13310152422413


while True:
    plt.clf()
    plt.ylabel("Energy")
    plt.xlabel("Iteration #")

    data = json.load(open("test.log"))
    iters = data["Energy"]["iters"]
    energy = data["Energy"]["Mean"]
    sigma = data["Energy"]["Sigma"]
    evar = data["Energy"]["Variance"]

    nres = len(iters)
    cut = 60
    if nres > cut:

        fitx = iters[-cut:-1]
        fity = energy[-cut:-1]
        z = np.polyfit(fitx, fity, deg=0)
        p = np.poly1d(z)

        plt.xlim([nres - cut, nres])
        maxval = np.max(energy[-cut:-1])
        plt.ylim([exact - (np.abs(exact) * 0.01), maxval + np.abs(maxval) * 0.01])
        error = (z[0] - exact) / -exact
        plt.gca().text(
            0.95,
            0.8,
            "Relative Error : " + "{:.2e}".format(error),
            verticalalignment="bottom",
            horizontalalignment="right",
            color="green",
            fontsize=15,
            transform=plt.gca().transAxes,
        )

        plt.plot(fitx, p(fitx))

    plt.errorbar(iters, energy, yerr=sigma, color="red")
    plt.axhline(y=exact, xmin=0, xmax=iters[-1], linewidth=2, color="k", label="Exact")

    plt.legend(frameon=False)
    plt.pause(1)
    # plt.draw()

plt.ioff()
plt.show()
