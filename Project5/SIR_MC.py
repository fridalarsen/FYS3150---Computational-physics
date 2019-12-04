import numpy as np
import matplotlib.pyplot as plt
from SIR1 import SIR

class SIR_MC(SIR):
    def solve_MC(self, S0, I0, n, t1):
        """
        Function for solving the SIR-model using Monte Carlo.
        Arguments:
            S0 (int): Initial number of susceptible individuals
            I0 (int): Initial number of infected individuals
            n (int): Number of time-steps/Monte Carlo cycles to perform
            t1 (float): Time starting point
        Returns:
            S (array): Evolution of number of susceptible individuals
            I (array): Evolution of number of infected individuals
            R (array): Evolution of number of recovered individuals
            t (array): Time array
        """
        # prepare arrays
        S = np.zeros(n)
        S[0] = S0
        I = np.zeros(n)
        I[0] = I0
        R = np.zeros(n)
        t = np.zeros(n)
        t[0] = t1

        # calculate time step
        dt_SI = 4./(self.a*self.N)
        dt_IR = 1./(self.b*self.N)
        dt_RS = 1./(self.c*self.N)

        dt = np.min(np.array([dt_SI, dt_IR, dt_RS]))

        for i in range(1, n):
            # calculate transition probabilities
            P_SI = (self.a*S[i-1]*I[i-1]*dt)/float(self.N)
            P_IR = self.b*I[i-1]*dt
            P_RS = self.c*R[i-1]*dt

            j = np.random.randint(0,3)
            r = np.random.random()

            # determine move / no move
            if j == 0 and r < P_SI and S[i-1] != 0:
                S[i] = S[i-1] - 1
                I[i] = I[i-1] + 1
                R[i] = R[i-1]
            elif j == 1 and r < P_IR and I[i-1] != 0:
                I[i] = I[i-1] - 1
                R[i] = R[i-1] + 1
                S[i] = S[i-1]
            elif j == 2 and r < P_RS and R[i-1] != 0:
                R[i] = R[i-1] - 1
                S[i] = S[i-1] + 1
                I[i] = I[i-1]
            else:
                S[i] = S[i-1]
                I[i] = I[i-1]
                R[i] = R[i-1]

            t[i] = t[i-1] + dt
        return S, I, R, t

if __name__ == "__main__":
    a = 4.0
    b = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0}
    c = 0.5

    model = SIR_MC(400)
    for key, b_val in b.items():
        n = 5000*b_val
        model.set_parameters(a, b_val, c)
        S, I, R, t = model.solve_MC(300, 100, int(n), 0.0)

        plt.plot(t, S, label="Susceptible", color="crimson")
        plt.plot(t, I, label="Infected", color="forestgreen")
        plt.plot(t, R, label="Recovered", color="gold")
        plt.xlabel("Time", fontsize=12)
        plt.ylabel("Number of individuals", fontsize=12)
        plt.title(f'Disease evolution in population {key}', fontsize=15)
        plt.legend()
        plt.savefig(f'Figures/MC_population_{key}.png')
        plt.show()
