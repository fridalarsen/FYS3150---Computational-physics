import numpy as np
import matplotlib.pyplot as plt
from tridiag_matrix_algorithm import run_tma
from special_tridiag_matrix_algorithm import special_tma


n = [int(m) for m in [1e1, 1e2, 1e3, 1e4, 1e5, 1e6]]
m = 5

# Original tridiagonal matrix algorithm
times_otma = np.zeros([len(n), 2])
for i in range(len(n)):
    t = np.zeros(m)
    for j in range(m):
        x, sol, algorithm, time = run_tma(n[i])
        t[j] = time
    times_otma[i, 0] = t.mean()
    times_otma[i, 1] = t.std()

# Special tridiagonal matrix algorithm
times_stma = np.zeros([len(n), 2])
for i in range(len(n)):
    t = np.zeros(m)
    for j in range(m):
        x = np.linspace(0, 1, n[i]+2)
        f = 100*np.exp(-10*x)
        alg, time = special_tma(f[1:-1], x[1:-1])
        t[j] = time
    times_stma[i, 0] = t.mean()
    times_stma[i, 1] = t.std()

plt.errorbar(n, times_otma[:,0], yerr = times_otma[:,1], c='red',
             fmt="none", capsize=5, label="Original algorithm")
plt.errorbar(n, times_stma[:,0], yerr = times_stma[:,1], c='blue',
             fmt="none", capsize=5, label="Special algorithm")
plt.scatter(n, times_otma[:,0], c='red', s=10)
plt.scatter(n, times_stma[:,0], c='blue', s=10)
plt.xscale("log")
plt.yscale("log")
plt.title("Algorithm run times")
plt.xlabel('log $n$')
plt.ylabel("log time [s]")
plt.legend()
plt.savefig("Figures/time_plot.png")
plt.show()
