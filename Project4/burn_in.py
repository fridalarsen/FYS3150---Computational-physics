import numpy as np
import matplotlib.pyplot as plt
from file_to_np import file_to_array
import os

os.system("g++ -std=c++11 -o 4c.x 4c.cpp ising_model.cpp -O3")
os.system("./4c.x")

M1 = file_to_array("Results/burn_in_M_4c_1.dat")
E1 = file_to_array("Results/burn_in_E_4c_1.dat")
acceptance1 = file_to_array("Results/4c_acceptance_1.dat")

M2 = file_to_array("Results/burn_in_M_4c_2.dat")
E2 = file_to_array("Results/burn_in_E_4c_2.dat")
acceptance2 = file_to_array("Results/4c_acceptance_2.dat")

M3 = file_to_array("Results/burn_in_M_4c_3.dat")
E3 = file_to_array("Results/burn_in_E_4c_3.dat")
acceptance3 = file_to_array("Results/4c_acceptance_3.dat")

M4 = file_to_array("Results/burn_in_M_4c_4.dat")
E4 = file_to_array("Results/burn_in_E_4c_4.dat")
acceptance4 = file_to_array("Results/4c_acceptance_4.dat")

plt.plot(M1, label="T=1.0", color="green")
plt.xlabel("Number of MC-cycles")
plt.ylabel("M")
plt.title("Time evolution of M from organized grid")
plt.legend()
plt.savefig("Figures/4c_M1.png")
plt.show()

plt.plot(E1, label="T=1.0", color="green")
plt.xlabel("Number of MC-cycles")
plt.ylabel("E")
plt.title("Time evolution of E from organized grid")
plt.legend()
plt.savefig("Figures/4c_E1.png")
plt.show()

plt.plot(M2, label="T=1.0", color="green")
plt.xlabel("Number of MC-cycles")
plt.ylabel("M")
plt.title("Time evolution of M from randomized grid")
plt.legend()
plt.savefig("Figures/4c_M2.png")
plt.show()

plt.plot(E2, label="T=1.0", color="green")
plt.xlabel("Number of MC-cycles")
plt.ylabel("E")
plt.title("Time evolution of E from randomized grid")
plt.legend()
plt.savefig("Figures/4c_E2.png")
plt.show()

plt.plot(M3, label="T=2.4", color="green")
plt.xlabel("Number of MC-cycles")
plt.ylabel("M")
plt.title("Time evolution of M from organized grid")
plt.legend()
plt.savefig("Figures/4c_M3.png")
plt.show()

plt.plot(E3, label="T=2.4", color="green")
plt.xlabel("Number of MC-cycles")
plt.ylabel("E")
plt.title("Time evolution of E from organized grid")
plt.legend()
plt.savefig("Figures/4c_E3.png")
plt.show()

plt.plot(M4, label="T=2.4", color="green")
plt.xlabel("Number of MC-cycles")
plt.ylabel("M")
plt.title("Time evolution of M from randomized grid")
plt.legend()
plt.savefig("Figures/4c_M4.png")
plt.show()

plt.plot(E4, label="T=2.4", color="green")
plt.xlabel("Number of MC-cycles")
plt.ylabel("E")
plt.title("Time evolution of E from randomized grid")
plt.legend()
plt.savefig("Figures/4c_E4.png")
plt.show()

plt.plot(acceptance1[0:150], label="T=1.0, organized", color="darkgreen")
plt.plot(acceptance2[0:150], label="T=1.0, randomized", color="crimson")
plt.plot(acceptance3[0:150], label="T=2.4, organzied", color="rebeccapurple")
plt.plot(acceptance4[0:150], label="T=2.4, randomized", color="gold")
plt.xlabel("Number of MC-cycles")
plt.ylabel("Number of accepted spin flips")
plt.legend()
plt.title("Accepted configurations")
plt.savefig("Figures/4c_accepted_config.png")
plt.show()
