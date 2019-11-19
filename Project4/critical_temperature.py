import numpy as np
import matplotlib.pyplot as plt
from file_to_np import file_to_array

L = np.array([40, 60, 80, 100])
T = np.linspace(2.24, 2.3, 30)

T_C_CV = np.zeros(len(L))
T_C_chi = np.zeros(len(L))

for i in range(len(L)):
    C_V = file_to_array(f'Results/4e_Cv_L{L[i]}.dat')
    chi = file_to_array(f'Results/4e_chi_L{L[i]}.dat')

    i_C_V = np.argmax(C_V)
    i_chi = np.argmax(chi)

    T_C_CV[i] = T[i_C_V]
    T_C_chi[i] = T[i_chi]

T_C_average = (T_C_CV + T_C_chi)/2.

print("T_C_CV:", T_C_CV)
print("T_C_chi:", T_C_chi)
print("T_C_average:", T_C_average)

x = 1/L
y = T_C_average
n = len(x)

# least squares
D = np.sum(x**2) - (1/n)*((np.sum(x))**2)
E = np.sum(x*y) - (1/n)*np.sum(x)*np.sum(y)
F = np.sum(y**2) - (1/n)*(np.sum(x)**2)

mean_x = (1/n)*np.sum(x)
mean_y = (1/n)*np.sum(y)

# coefficients
m = E/D
c = mean_y - m*mean_x

# errors
d = y - m*x - c

delta_m = np.sqrt((1/(n-2))*((np.sum(d**2))/D))
delta_c = np.sqrt((1/(n-2))*(D/n + mean_x**2)*((np.sum(d**2))/D))

print(f'T_C={c} +- {delta_c}')

plt.scatter(1/L, T_C_average)
plt.plot(x, m*x + c)
plt.xlabel(r'$\frac{1}{L}$', fontsize=12)
plt.ylabel(r'$T_C$', fontsize=12)
plt.title("Critical temperature", fontsize=14)
plt.subplots_adjust(bottom=0.12)
plt.savefig("Figures/critical_temp.png")
plt.show()

"""
Sample run:

[terminal]$ python3 critical_temperature.py
T_C_CV: [2.28551724 2.28344828 2.28137931 2.27103448]
T_C_chi: [2.24206897 2.25034483 2.25655172 2.26275862]
T_C_average: [2.2637931  2.26689655 2.26896552 2.26689655]
T_C=2.2709832433181005 +- 0.0021476465750923635
"""
