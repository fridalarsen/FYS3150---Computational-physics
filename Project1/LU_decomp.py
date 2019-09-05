import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import time


n = [int(m) for m in [1e1, 1e2, 1e3]]
t = np.zeros(len(n))

for i in range(len(n)):
    a = np.ones(n[i]-1) * (-1)
    b = np.ones(n[i]) * (2)
    A = np.diag(b)

    for j in range(n[i]-1):
        A[j+1, j] = a[j]
        A[j, j+1] = a[j]

    start = time.process_time()
    LU, P = scipy.linalg.lu_factor(A)

    h = 1/(n[i]+1)
    x = np.linspace(0, 1, n[i]+2)
    f = h**2*100*np.exp(-10*x)
    u_sol = 1 - (1-np.exp(-10))*x - np.exp(-10*x)

    u_lu = np.zeros(n[i]+2)
    u_lu[1:-1] = scipy.linalg.lu_solve((LU, P), f[1:-1])
    end = time.process_time()
    t[i] = end - start

    plt.plot(x, u_lu, label="LU-decomp")
    plt.plot(x, u_sol, label="Closed-form")
    plt.title("Approximation by LU-decomposition, n={}".format(n[i]))
    plt.xlabel('$x$')
    plt.ylabel('$u(x)$')
    plt.legend()
    plt.show()

print(n)
print(t)
