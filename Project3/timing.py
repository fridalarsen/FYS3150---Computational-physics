import numpy as np
import matplotlib.pyplot as plt
from project3 import Project3

gauss_N = [5, 10, 15, 20, 25, 30, 35, 40]
exact = 5*(np.pi**2)/(16**2)
P3 = Project3()

with open("results_3a.dat", "w", 1) as f:
    f.write("n - output - error - time\n")
    for n in gauss_N:
        print(f"3a, working on N = {n}.")
        for t in range(5):
            output_a, time_a = P3("3a", lamb=2.2, N=n, tol=1e-8)
            error_a = abs(exact - output_a)
            f.write(f'{n} - {output_a} - {error_a} - {time_a}\n')

with open("results_3b.dat", "w", 1) as f:
    f.write("n - output - error - time\n")
    for n in gauss_N:
        print(f"3b, working on N = {n}.")
        for t in range(5):
            output_b, time_b = P3("3b", N=n, tol=1e-8)
            error_b = abs(exact - output_b)
            f.write(f'{n} - {output_b} - {error_b} - {time_b}\n')

mc_N = [10, 100, 1000, 10000, 100000, 500000, 1000000]

with open("results_3c.dat", "w", 1) as f:
    f.write("n - output - var - error - time\n")
    for n in mc_N:
        print(f"3c, working on N = {n}.")
        for t in range(5):
            output, var, time = P3("3c", N=n, a=-2.2, b=2.2)
            error = abs(exact - output)
            f.write(f"{n} - {output} - {var} - {error} - {time}\n")

with open("results_3d.dat", "w", 1) as f:
    f.write("n - output - var - error - time\n")
    for n in mc_N:
        print(f"3d, working on N = {n}.")
        for t in range(5):
            output, var, time = P3("3d", N=n)
            error = abs(exact - output)
            f.write(f"{n} - {output} - {var} - {error} - {time}\n")

with open("results_3e.dat", "w", 1) as f:
    f.write("n - output - var - error - time\n")
    for n in mc_N:
        print(f"3e, working on N = {n}.")
        for t in range(5):
            output, var, time = P3("3e", N=n/2, npar=2)
            error = abs(exact - output)
            f.write(f"{n} - {output} - {var} - {error} - {time}\n")
