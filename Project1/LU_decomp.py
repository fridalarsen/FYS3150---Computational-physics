import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import time


n = [int(m) for m in [1e1, 1e2, 1e3, 1e4]]
m = 5

times_LU_decomp = np.zeros([len(n),m])
for i in range(len(n)):
    t = np.zeros(m)
    for j in range(m):
        a = np.ones(n[i]-1) * (-1)
        b = np.ones(n[i]) * (2)
        A = np.diag(b)

        h = 1/(n[i]+1)
        x = np.linspace(0, 1, n[i]+2)
        f = h**2*100*np.exp(-10*x)
        u_sol = 1 - (1-np.exp(-10))*x - np.exp(-10*x)

        u_lu = np.zeros(n[i]+2)

        for k in range(n[i]-1):
            A[k+1, k] = a[k]
            A[k, k+1] = a[k]

        start = time.process_time()
        LU, P = scipy.linalg.lu_factor(A)
        u_lu[1:-1] = scipy.linalg.lu_solve((LU, P), f[1:-1])
        end = time.process_time()

        t[j] = end - start
    times_LU_decomp[i, 0] = t.mean()
    times_LU_decomp[i, 1] = t.std()

    plt.plot(x, u_lu, label="LU-decomp")
    plt.plot(x, u_sol, label="Closed-form")
    plt.title("Approximation by LU-decomposition, n={}".format(n[i]))
    plt.xlabel('$x$')
    plt.ylabel('$u(x)$')
    plt.legend()
    plt.show()

plt.errorbar(n, times_LU_decomp[:,0], yerr = times_LU_decomp[:,1], c='red',
             fmt="none", capsize=5)
plt.scatter(n, times_LU_decomp[:,0], c='red', s=10)
plt.xscale("log")
plt.yscale("log")
plt.title("Algorithm run times - LU decomposition")
plt.xlabel('log $n$')
plt.ylabel("log time [s]")
plt.savefig("Figures/time_plot_LU.png")
plt.show()
