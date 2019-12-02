import numpy as np
import matplotlib.pyplot as plt

class SIR:
    """
    Class for modeling an infectious disease.
    """
    def __init__(self, N, a=1.0, b=1.0, c=1.0):
        """
        Arguments:
            N (int): Population size
            a (float, optional): Rate of transmission, defaults to 1
            b (float, optional): Rate of recovery, defaults to 1
            c (float, optional): Rate of immunity loss, defaults to 1
        """
        self.N = N

        self.set_parameters(a, b, c)

    def set_parameters(self, a, b, c):
        """
        Function for setting the rate parameters of the model.
        Arguments:
            a (float): Rate of transmission
            b (float): Rate of recovery
            c (float): Rate of immunity loss
        """
        self.a = a
        self.b = b
        self.c = c

    def S_deriv(self, S, I, R):
        """
        Function for finding the derivative of S.
        Arguments:
            S (float): Number of susceptible individuals
            I (float): Number of infected individuals
            R (float): Number of recovered individuals
        Returns:
            S_deriv (float): The derivative of S
        """
        S_deriv = self.c*(float(self.N) - S - I) - (self.a*S*I)/float(self.N)

        return S_deriv

    def I_deriv(self, S, I):
        """
        Function for finding the derivative of I.
        Arguments:
            S (float): Number of susceptible individuals
            I (float): Number of infected individuals
        Returns:
            I_deriv (float): The derivative of I
        """
        I_deriv = (self.a*S*I)/float(self.N) - self.b*I

        return I_deriv

    def solve(self, S0, I0, n, t1, t2):
        """
        Function for solving the SIR-model using RK4.
        Arguments:
            I0 (int): Initial number of infected individuals
            S0 (int): Initial number of susceptible individuals
            n (int): Number of time-steps to perform
            t1 (float): Time starting point
            t2 (float): Time end point
        Returns:
            S (array): Evolution of number of susceptible individuals
            I (array): Evolution of number of infected individuals
            R (array): Evolution of number of recovered individuals
            t (array): Time array
        """
        h = (t2 - t1)/float(n)

        S = np.zeros(n)
        I = np.zeros(n)
        R = np.zeros(n)
        t = np.zeros(n)

        # set initial conditions
        S[0] = S0
        I[0] = I0

        # solve
        for i in range(1, n):
            k1_S = h*self.S_deriv(S[i-1], I[i-1], R[i-1])
            k2_S = h*self.S_deriv(S[i-1] + k1_S/2., I[i-1] + k1_S/2.,
                                  R[i-1] + k1_S/2.)
            k3_S = h*self.S_deriv(S[i-1] + k2_S/2., I[i-1] + k2_S/2.,
                                  R[i-1] + k2_S/2.)
            k4_S = h*self.S_deriv(S[i-1] + k3_S, I[i-1] + k3_S, R[i-1] + k3_S)

            k1_I = h*self.I_deriv(S[i-1], I[i-1])
            k2_I = h*self.I_deriv(S[i-1] + k1_I/2., I[i-1] + k1_I/2.)
            k3_I = h*self.I_deriv(S[i-1] + k2_I/2., I[i-1] + k2_I/2.)
            k4_I = h*self.I_deriv(S[i-1] + k3_I, I[i-1] + k3_I)

            S[i] = S[i-1] + (1/6.)*(k1_S + 2*k2_S + 2*k3_S + k4_S)
            I[i] = I[i-1] + (1/6.)*(k1_I + 2*k2_I + 2*k3_I + k4_I)
            R[i] = self.N - S[i] - I[i]
            t[i] = t[i-1] + h

        return S, I, R, t

if __name__ == "__main__":
    a = 4.0
    b = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0}
    c = 0.5

    model = SIR(N=400)
    for key, b_val in b.items():
        model.set_parameters(a, b_val, c)
        S, I, R, t = model.solve(300, 100, 1000, 0.0, 15.0)

        plt.plot(t, S, label="Susceptible", color="crimson")
        plt.plot(t, I, label="Infected", color="forestgreen")
        plt.plot(t, R, label="Recovered", color="gold")
        plt.xlabel("Time", fontsize=12)
        plt.ylabel("Number of individuals", fontsize=12)
        plt.title(f'Disease evolution in population {key}', fontsize=15)
        plt.legend()
        plt.savefig(f'Figures/RK4_population_{key}')
        plt.show()
