import numpy as np
import matplotlib.pyplot as plt


L = 4
A = np.random.randint(0, high=2, size=(L, L))*2-1
print(A)

au = np.roll(A, 1, axis=0)
ar = np.roll(A, 1, axis=1)

msh = au*ar

fig, ax = plt.subplots(2, 1)
ax[0].imshow(au)
ax[1].imshow(msh)
plt.show()