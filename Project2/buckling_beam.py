import numpy as np
import matplotlib.pyplot as plt
from eigenvalue_problem import EigenvalueProblem

rotations = np.array([5e1, 1e2, 2e2, 3e2, 5e2, 7e2, 1e3, 1.5e3])

np_diff = np.zeros(len(rotations))
jc1_diff = np.zeros(len(rotations))
jc2_diff = np.zeros(len(rotations))
arma_diff = np.zeros(len(rotations))

rho = np.linspace(0, 1, 1e2 + 2)
h = rho[1] - rho[0]
rho = rho[1:-1]

d = np.ones(len(rho)) * (2/(h**2))
a = np.ones(len(rho)) * (-1/(h**2))

T = np.diag(d) + np.diag(a[:-1], k=1) + np.diag(a[:-1], k=-1)

# calculate analytic eigenvalues
j = np.linspace(1, len(rho), len(rho))
eig_a = d + 2*a*np.cos((j*np.pi)/(len(rho)+1))

f = EigenvalueProblem()
for i in range(len(rotations)):
    # calculate eigenvalues numerically
    eig_np, vec_np = f(T, "NumPy")
    eig_jc1, vec_jc1 = f(T, "Jacobi_py", rotations[i])
    eig_jc2, vec_jc2 = f(T, "Jacobi_cpp", rotations[i])
    eig_arma, vec_arma = f(T, "Armadillo")

    np_diff[i] = np.max(np.abs(eig_np - eig_a))/np.max(eig_a)
    jc1_diff[i] = np.max(np.abs(eig_jc1 - eig_a))/np.max(eig_a)
    jc2_diff[i] = np.max(np.abs(eig_jc2 - eig_a))/np.max(eig_a)
    arma_diff[i] = np.max(np.abs(eig_arma - eig_a))/np.max(eig_a)

plt.plot(rotations, jc2_diff, 'o', label='Jacobi algorithm')
plt.plot(rotations, arma_diff, label='Armadillo')
plt.legend()
plt.title("Convergence of Jacobi method")
plt.xlabel('Number of rotations')
plt.ylabel('Relative max error')
plt.savefig('Figures/relative_max_error.png')
plt.show()

# determine gamma
F = 1.
L = 1.

eig_np, vec_np = f(T, "NumPy")
eig_jc1, vec_jc1 = f(T, "Jacobi_py", 10e3)
eig_jc2, vec_jc2 = f(T, "Jacobi_cpp", 10e3)
eig_arma, vec_arma = f(T, "Armadillo")

gamma_np = -(F*L**2)/eig_np
gamma_jc1 = -(F*L**2)/eig_jc1
gamma_jc2 = -(F*L**2)/eig_jc2
gamma_arma = -(F*L**2)/eig_arma

print(gamma_np[0])
print(gamma_jc1[0])
print(gamma_jc2[0])
print(gamma_arma[0])

# visualizing eigenvectors
plt.plot(rho, vec_jc2[0], color="cyan", label="Jacobi algorithm")
plt.plot(rho, vec_jc2[1], color="cyan")
plt.plot(rho, vec_jc2[2], color="cyan")
plt.plot(rho, -vec_arma[0], color="purple", label="Armadillo")
plt.plot(rho, vec_arma[1], color="purple")
plt.plot(rho, vec_arma[2], color="purple")
plt.legend()
plt.xlabel(r'$\rho$')
plt.ylabel('Vertical displacement')
plt.title("Eigenvectors of a buckling beam")
plt.savefig("Figures/eigenvectors_buckling_beam")
plt.show()
