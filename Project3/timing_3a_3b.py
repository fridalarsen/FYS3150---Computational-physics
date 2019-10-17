import numpy as np
import matplotlib.pyplot as plt
from project3 import Project3

N = [5, 10, 15, 20, 25, 30, 35, 40]
exact = 5*(np.pi**2)/(16**2)
P3 = Project3()

with open("results_3a.dat", "w", 1) as f:
    with open("results_3b.dat", "w", 1) as g:
        f.write("n - output - error - time \n")
        g.write("n - output - error - time \n")
        for n in N:
            print(f"Currently working on N = {n}.")
            for t in range(5):
                output_a, time_a = P3("3a", lamb=2.2, N=n, tol=1e-8)
                error_a = abs(exact - output_a)
                f.write(f'{n} - {output_a} - {error_a} - {time_a}\n')

                output_b, time_b = P3("3b", N=n, tol=1e-8)
                error_b = abs(exact - output_b)
                g.write(f'{n} - {output_b} - {error_b} - {time_b}\n')
