import numpy as np
import matplotlib.pyplot as plt

def special_tma(f, n):
    """
    Tridiagonal matrix algorithm for the special case where the upper and lower
    diagonal are the same, and the arrays only consist of the same value.

    Arguments:
        f (array) : RHS vector.
        n (int) : Grid size

    Returns:
        Solution u of the problem Au = f in the specialized case.

    """
    u       = np.zeros(n)
    f_tilde = np.zeros(n)
    b_tilde = np.array([(i+1)/i for i in range(1,n+1)])

    h = x[1] - x[0]
    b_squig = h**2*f

    f_tilde[0] = b_squig[0]
    for i in range(1,n):
        i_ = i/(i+1)
        f_tilde[i] = b_squig[i] + i_*f_tilde[i-1]

    u[n-1] = f_tilde[n-1]/b_tilde[n-1]
    for i in range(n-2, -1, -1):
        i_ = (i+1)/(i+2)
        u[i] = i_*(f_tilde[i]+u[i+1])

    return np.concatenate([np.zeros(1), u, np.zeros(1)])

if __name__ == '__main__':
    n_ = [10, 100, 1000]

    for i in n_:

        x = np.linspace(0, 1, i+2)
        f = 100*np.exp(-10*x)
        sol  = 1 - (1-np.exp(-10))*x - np.exp(-10*x)

        alg_2 = special_tma(f, i)

        plt.plot(x, sol, label="Closed-form solution")
        plt.plot(x, alg_2, label="Specialized algorithm solution")
        plt.legend()
        plt.xlabel('$x$')
        plt.ylabel('$u(x)$')
        plt.title("Approximation by special tridiagonal matrix algorithm, n={}".format(i))
        plt.savefig("Figures/stma_n{}.png".format(i))
        plt.show()
