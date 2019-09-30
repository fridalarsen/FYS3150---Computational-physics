import numpy as np
import matplotlib.pyplot as plt
from eigenvalue_problem import EigenvalueProblem

def lambda_(n):
    return (4*n + 3)

rho_max = [1, 5, 10, 25, 50, 75]
N = [1e1, 2.5e1, 5e1, 1e2, 3e2, 5e2]

lambda_arma = np.zeros((len(N), len(rho_max)))
lambda_jc = np.zeros((len(N), len(rho_max)))

f = EigenvalueProblem()

for i in range(len(rho_max)):
    for j in range(len(N)):
        rho = np.linspace(0, rho_max[i], N[j] + 2)
        rho = rho[1:-1]
        h = rho[1]-rho[0]

        V = rho**2
        d = np.ones(len(rho)) * (2/(h**2) + V)
        e = np.ones(len(rho)) * (-1/h**2)

        T = np.diag(d) + np.diag(e[:-1], k=-1) + np.diag(e[:-1], k=1)

        eig_arma, vec_arma = f(T, "Armadillo")
        eig_jc, vec_jc = f(T, "Jacobi_cpp", 5e3)

        lambda_arma[j,i] = np.max(np.abs(eig_arma - \
                           lambda_(np.arange(len(eig_arma)))))
        lambda_jc[j,i] = np.max(np.abs(eig_jc - lambda_(np.arange(len(eig_jc)))))

fig, ax = plt.subplots()
im = ax.imshow(np.log10(lambda_arma))
fig.colorbar(im)
ax.set_xticklabels([""]+[str(r) for r in rho_max])
ax.set_yticklabels([""]+[str(n) for n in N])
ax.set_xlabel(r'$\rho_{max}$', fontsize=13)
ax.set_ylabel('N', fontsize=13)
ax.set_title('Logarithmic max error - Armadillo')
fig.subplots_adjust(left=0.16, right=0.96)
plt.savefig("Figures/error_colorbar_qdots1.png")
plt.show()

fig, ax = plt.subplots()
im = ax.imshow(np.log10(lambda_jc))
fig.colorbar(im)
ax.set_xticklabels([""]+[str(r) for r in rho_max])
ax.set_yticklabels([""]+[str(n) for n in N])
ax.set_xlabel(r'$\rho_{max}$', fontsize=13)
ax.set_ylabel('N', fontsize=13)
ax.set_title('Logarithmic max error - Jacobi method')
fig.subplots_adjust(left=0.16, right=0.96)
plt.savefig("Figures/error_colorbar_qdots2.png")
plt.show()

# visualizing eigenvectors
rho = np.linspace(0, 5, 1e2 + 2)
rho = rho[1:-1]
h = rho[1]-rho[0]

V = rho**2
d = np.ones(len(rho)) * (2/(h**2) + V)
e = np.ones(len(rho)) * (-1/h**2)

T = np.diag(d) + np.diag(e[:-1], k=-1) + np.diag(e[:-1], k=1)

eig_jc, vec_jc = f(T, "Jacobi_cpp", 1e4)
plt.plot(rho, vec_jc[0], label=f'$\lambda=${eig_jc[0]:.4f}')
plt.plot(rho, vec_jc[1], label=f'$\lambda=${eig_jc[1]:.4f}')
plt.plot(rho, vec_jc[2], label=f'$\lambda=${eig_jc[2]:.4f}')
plt.legend()
plt.xlabel(r'$\rho$', fontsize=13)
plt.ylabel(r'$u(\rho)$', fontsize=13)
plt.title("Eigenvectors of one electron in harmonic oscillator potential")
plt.savefig("Figures/eigvec_qdots1.png")
plt.show()
