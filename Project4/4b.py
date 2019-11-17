import numpy as np
import matplotlib.pyplot as plt
from file_to_np import file_to_array
import os

n = 7
os.system("g++ -std=c++11 -o 4b.x 4b.cpp -O3")
os.system(f"./4b.x {n}")

k_B = 1.
T = 1.
beta = 1/(k_B*T)
J = 1.

# Analytic expression
E_mean_a = (-8*J*np.exp(beta*8*J) + 8*J*np.exp(-beta*8*J)) / (np.exp(beta*8*J)
            + np.exp(-beta*8*J) + 6)
M_mean_abs_a = (4*np.exp(8*beta*J) + 2) / (np.exp(8*beta*J) + np.exp(-8*beta*J)
                + 6)
C_V_a = (1./(k_B*(T**2)))*((64*(J**2)*np.exp(beta*8*J)
         + 64*(J**2)*np.exp(-beta*8*J)) / (np.exp(beta*8*J)
         + np.exp(-beta*8*J) + 6) - E_mean_a**2)
chi_a = beta*((8*np.exp(-beta*8*J) + 8*np.exp(beta*8*J) + 4) /
        (np.exp(8*beta*J) + np.exp(8*beta*J) + 6))

# Numerical results
E_mean = file_to_array("Results/E_mean_4b.dat")
M_mean = file_to_array("Results/M_mean_4b.dat")
C_V = file_to_array("Results/C_V_4b.dat")
chi = file_to_array("Results/chi_4b.dat")/4.

MC_cycles = np.logspace(1, n, n)

# Plots for comparison
plt.plot(MC_cycles, E_mean, label="Numerical", color="red")
plt.axhline(y=E_mean_a, label="Analytic", color="green")
plt.xscale("log")
plt.xlabel("Number of MC-cycles", fontsize=12)
plt.ylabel(r'$\langle E\rangle$', fontsize=12)
plt.title("Mean energy of 2x2 lattice", fontsize=14)
plt.legend()
plt.subplots_adjust(left=0.15)
plt.savefig("./Figures/E_mean_4b.png")
plt.show()

plt.plot(MC_cycles, M_mean, label="Numerical", color="red")
plt.axhline(y=M_mean_abs_a, label="Analytic", color="green")
plt.xscale("log")
plt.xlabel("Number of MC-cycles", fontsize=12)
plt.ylabel(r'$\langle |\mathcal{M}| \rangle$', fontsize=12)
plt.title("Mean magnetization of 2x2 lattice", fontsize=14)
plt.legend()
plt.savefig("./Figures/M_mean_4b.png")
plt.show()

plt.plot(MC_cycles, C_V, label="Numerical", color="red")
plt.axhline(y=C_V_a, label="Analytic", color="green")
plt.xscale("log")
plt.xlabel("Number of MC-cycles", fontsize=12)
plt.ylabel(r'$C_V$', fontsize=12)
plt.title("Specific heat of 2x2 lattice", fontsize=14)
plt.legend()
plt.savefig("./Figures/C_V_4b.png")
plt.show()

plt.plot(MC_cycles, chi, label="Numerical", color="red")
plt.axhline(y=chi_a, label="Analytic", color="green")
plt.xscale("log")
plt.xlabel("Number of MC-cycles", fontsize=12)
plt.ylabel(r'$\chi$', fontsize=12)
plt.title("Susceptibility of 2x2 lattice", fontsize=14)
plt.legend()
plt.savefig("./Figures/chi_4b.png")
plt.show()
