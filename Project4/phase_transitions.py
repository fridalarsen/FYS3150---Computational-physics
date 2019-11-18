import numpy as np
import matplotlib.pyplot as plt
from file_to_np import file_to_array

L = [40, 60, 80, 100]
T = np.linspace(2.0, 2.3, 7)

for i in range(len(L)):
    E_mean = file_to_array(f'Results/4e_E_L{i}.dat')
    M_mean = file_to_array(f'Results/4e_M_L{i}.dat')
    C_V = file_to_array(f'Results/4e_Cv_L{i}.dat')
    chi = file_to_array(f'Results/4e_chi_L{i}.dat')

    data = [E_mean, M_mean, C_V, chi]
    ylabels = [r'$\langle E\rangle$', r'$\langle |M|\rangle$', r'$C_V$',
               r'$\chi$']
    titles = [f'Mean energy during phase transition, L={L[i]}',
              f'Mean absolute magnetization during phase transition, L={L[i]}',
              f'Specific heat during phase transition, L={L[i]}',
              f'Susceptibility during phase transition, L={L[i]}']

    for j in range(len(data)):
        plt.plot(T, data[j], color="indigo")
        plt.xlabel("T [kT/J]", fontsize=12)
        plt.ylabel(ylabels[j], fontsize=12)
        plt.subplots_adjust(left=0.15, right=0.92)
        plt.title(titles[j], fontsize=14)
        plt.savefig(f'Figures/4e_L{L[i]}_{j}.png')
        plt.show()

# the zoomed-in plots for L=100:
T = np.linspace(2.2, 2.3, 30)

E_mean = file_to_array(f'Results/4e_E_L100_zoom.dat')
M_mean = file_to_array(f'Results/4e_M_L100_zoom.dat')
C_V = file_to_array(f'Results/4e_Cv_L100_zoom.dat')
chi = file_to_array(f'Results/4e_chi_L100_zoom.dat')

data = [E_mean, M_mean, C_V, chi]
ylabels = [r'$\langle E\rangle$', r'$\langle |M|\rangle$', r'$C_V$',
           r'$\chi$']
titles = ['Mean energy during phase transition, L=100',
          'Mean absolute magnetization during phase transition, L=100',
          'Specific heat during phase transition, L=100',
          'Susceptibility during phase transition, L=100']

for i in range(len(data)):
    plt.plot(T, data[i], color="indigo")
    plt.xlabel("T [kT/J]", fontsize=12)
    plt.ylabel(ylabels[i], fontsize=12)
    plt.subplots_adjust(left=0.15, right=0.92)
    plt.title(titles[i], fontsize=14)
    plt.savefig(f'Figures/4e_L100_zoom_{i}.png')
    plt.show()
