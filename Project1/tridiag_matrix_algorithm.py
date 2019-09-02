import numpy as np
import matplotlib.pyplot as plt

def tridiagonal_matrix_algorithm(a, b, c, b_tilde):
    n       = len(b)
    u       = np.zeros(n)
    c_prime = np.zeros(n-1)
    d_prime = np.zeros(n)

    c_prime[0]  = c[0] / b[0]
    d_prime[0]  = b_tilde[0] / b[0]

    for i in range(1, n-1):
        c_prime[i] = c[i]/(b[i]-(a[i-1]*c_prime[i-1]))

    for j in range(1, n):
        d_prime[j] = (b_tilde[j]-(a[j-1]*d_prime[j-1])) \
                     / (b[j]-(a[j-1]*c_prime[j-1]))

    u[n-1] = d_prime[n-1]
    for k in range(n-2, -1, -1):
        u[k] = d_prime[k] - (c_prime[k]*u[k+1])

    return u

def run_tma(n):
    x  = np.linspace(0, 1, n+2)
    h  = x[1] - x[0]

    u_sol = 1 - (1-np.exp(-10))*x - np.exp(-10*x)

    b_tilde = (h**2)*100*np.exp(-10*x)
    a = np.ones(n-1)*(-1)
    b = np.ones(n)*(2)
    c = np.ones(n-1)*(-1)

    u_algorithm = np.zeros(n+2)
    u_algorithm[1:n+1] = tridiagonal_matrix_algorithm(a, b, c, b_tilde)

    return x, u_sol, u_algorithm

if __name__ == '__main__':

    n_ = [10, 100, 1000]

    for i in n_:
        x, sol, algorithm = run_tma(i)
        plt.plot(x, sol, label="Closed-form solution")
        plt.plot(x, algorithm, label="Algorithm solution")
        plt.legend()
        plt.title("Approximation by tridiagonal matrix algorithm, n={}".format(i))
        plt.savefig("t_m_a_n_{}.png".format(i))
        plt.show()
















# y0
