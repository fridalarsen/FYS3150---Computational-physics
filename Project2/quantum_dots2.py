import numpy as np
import matplotlib.pyplot as plt
from eigenvalue_problem import EigenvalueProblem

omega_r = [0.01, 0.5, 1., 5.]
rho = np.linspace(0, 5, 1e2 + 2)
rho = rho[1:-1]
h = rho[1]-rho[0]

for i in range(len(omega_r)):
    V = (omega_r[i]**2) * (rho**2) + 1/rho
    d = np.ones(len(rho)) * (2/(h**2) + V)
    e = np.ones(len(rho)) * (-1/h**2)

    T = np.diag(d) + np.diag(e[:-1], k=-1) + np.diag(e[:-1], k=1)
    f = EigenvalueProblem()
    val_j, vec_j = f(T, "Jacobi_cpp", 1e4)
    val_arma, vec_arma = f(T, "Armadillo")

    print(f'omega_r={omega_r[i]}')
    print('Armadillo:', val_arma[:5])
    print('Jacobi:', val_j[:5])

    plt.plot(rho, vec_j[0], label=f'$\lambda$={val_j[0]:.4f}')
    plt.plot(rho, vec_j[1], label=f'$\lambda$={val_j[1]:.4f}')
    plt.plot(rho, vec_j[2], label=f'$\lambda$={val_j[2]:.4f}')
    plt.title(f'Eigenvectors of 2 electrons in a H.O. potential, $\omega_r=${omega_r[i]}')
    plt.xlabel(r'$\rho$', fontsize=13)
    plt.ylabel(r'$\psi(\rho)$', fontsize=13)
    plt.legend()
    plt.savefig(f'Figures/eigvec_qdots2_{i}')
    plt.show()
