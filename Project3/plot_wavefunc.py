import numpy as np
import matplotlib.pyplot as plt

alpha = 2

r = np.linspace(1,2.5,100)
psi = np.exp(-2*alpha*r)

fig, ax = plt.subplots()

ax.plot(r, psi, color="orange")
ax.set_xlabel(r'$r_1+r_2$', fontsize=13)
ax.set_ylabel(r'exp$(-2\alpha(r_1+r_2))$', fontsize=13)
ax.set_title(r'Exponential decay of $|\Psi|^2$')
fig.subplots_adjust(left=0.14)
plt.savefig("./Figures/wavefunc.png")
plt.show()
