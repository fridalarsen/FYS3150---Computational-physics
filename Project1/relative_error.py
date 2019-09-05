import numpy as np
from special_tridiag_matrix_algorithm import special_tma


def relative_error(v, u, n):
    eps = np.zeros(n)
    h = 1/(n+1)

    for i in range(n):
        diff   = v[i] - u[i]
        error  = np.log10(np.abs(diff / u[i]))
        eps[i] = error

    return np.max(eps), h


n = [int(m) for m in [1e1, 1e2, 1e3, 1e4, 1e5, 1e6]]

relative_errors = np.zeros(len(n))
h = np.zeros(len(n))
for i in range(len(n)):
    x = np.linspace(0, 1, n[i]+2)
    f = 100*np.exp(-10*x)

    sol = 1 - (1-np.exp(-10))*x - np.exp(-10*x)
    algo, runtime = special_tma(f[1:-1], x[1:-1])

    relative_errors[i], h[i] = relative_error(algo[1:-1], sol[1:-1], n[i])

print(relative_errors)
print(h)
