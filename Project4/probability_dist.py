import numpy as np
import matplotlib.pyplot as plt
from file_to_np import file_to_array
import os

os.system("g++ -std=c++11 -o 4d.x 4d.cpp ising_model.cpp -O3")
os.system("./4d.x")

E1 = file_to_array("Results/4d_T1_E.dat")
E2 = file_to_array("Results/4d_T2_E.dat")

var1 = np.var(E1)
var2 = np.var(E2)

plt.hist(E1, bins=20, label=f'$\sigma_E^2$ = {var1:.2f}', color="firebrick")
plt.title("Distribution of energies, T=1.0", fontsize=14)
plt.xlabel("E", fontsize=12)
plt.ylabel("Number of configurations", fontsize=12)
plt.legend()
plt.savefig("Figures/4d_histogram_T1.png")
plt.show()

plt.hist(E2, bins=20, label=f'$\sigma_E^2$ = {var2:.2f}', color="firebrick")
plt.title("Distribution of energies, T=2.4", fontsize=14)
plt.xlabel("E", fontsize=12)
plt.ylabel("Number of configurations", fontsize=12)
plt.legend()
plt.savefig("Figures/4d_histogram_T2.png")
plt.show()
