import numpy as np
import matplotlib.pyplot as plt

class SIR_model:
    """
    Class for modeling an infectious disease.
    """
    def __init__(self, N0, vital_dynamics=False, seasonal_variation=False,
                 vaccination=False):
        """
        Arguments:
            N (int): Initial size of population
            vital_dynamics (bool): Whether to include population aspects or not,
                                   defaults to False
            seasonal_variation (bool): Whether to have a time-dependent rate
                                       of transmission or not, defaults to False
            vaccination (bool): Whether to include vaccination or not, defaults
                                to False.
        """
        self.N = N0

        self.vital_dynamics = vital_dynamics
        self.seasonal_variation = seasonal_variation
        self.vaccination = vaccination

        # bools for keeping track of what parameters have been set by user
        self.vr = False
        self.vp = False
        self.vacr = False

    def set_disease_parameters(self, a=1.0, b=1.0, c=1.0):
        """
        Function for specifying the disease parameters.
        Arguments:
            a (float, optional): Rate of transmission, defaults to 1
            b (float, optional): Rate of recovery, defaults to 1
            c (float, optional): Rate of immunity loss, defaults to 1
        """
        if self.seasonal_variation == False:
            self.a = a

        self.b = b
        self.c = c

    def vital_rates(self, e=1.0, d=1.0, d1=1.0):
        """
        Function for specifying population details.
        Arguments:
            e (float): Birth rate, defaults to 1
            d (float): (Natural) death rate, defaults to 1
            d1 (float): Death rate due to disease, defaults to 1
        """
        self.e = e
        self.d = d
        self.d1 = d1

        self.vr = True

    def variational_param(self, A, omega, a0):
        """
        Function for setting the parameters of .
        Arguments:
            A (float): Maximum deviation from a0
            omega (float): Frequency of oscillation
            a0 (float): Average transmission rate
        """
        self.A = A
        self.omega = omega
        self.a0 = a0

        self.vp = True

    def a_var(self, t):
        """
        Function for calculating transmission rate.
        Arguments:
            t (float): Time
        Returns:
            a (float): The transmission rate at time t
        """
        a = self.A*np.cos(self.omega*t) + self.a0

        return a

    def vaccination_rate(self, f):
        """
        Function for specifying the rate of vaccination.
        Arguments:
            f (float or function): Rate of vaccination
        """
        self.f = f

        self.vacr = True

    def S_deriv(self, S, I, R, t):
        """
        Function for finding the derivative of S.
        Arguments:
            S (float): Number of susceptible individuals
            I (float): Number of infected individuals
            R (float): Number of recovered individuals
            t (float): Current time-value
        Returns:
            S_deriv (float): The derivative of S
        """
        S_deriv = self.c*R - (self.a*S*I)/float(self.N)

        # check for model extensions
        if self.vital_dynamics and not self.vaccination:
            S_deriv += -self.d*S + self.e*self.N
        elif self.vaccination and not self.vital_dynamics:
            if isinstance(self.f, float):
                S_deriv -= self.f
            else:
                S_deriv -= self.f(t)
        elif self.vaccination and self.vital_dynamics:
            if isinstance(self.f, float):
                S_deriv += -self.d*S + self.e*self.N - self.f
            else:
                S_deriv += -self.d*S + self.e*self.N - self.f(t)

        return S_deriv

    def I_deriv(self, S, I, t):
        """
        Function for finding the derivative of I.
        Arguments:
            S (float): Number of susceptible individuals
            I (float): Number of infected individuals
            t (float): Current time-value
        Returns:
            I_deriv (float): The derivative of I
        """
        # check for seasonal variation
        if self.seasonal_variation:
            I_deriv = (self.a_var(t)*S*I)/float(self.N) - self.b*I
        else:
            I_deriv = (self.a*S*I)/float(self.N) - self.b*I

        # check for vital dynamics
        if self.vital_dynamics:
            I_deriv += -self.b*I - self.d*I - self.d1*I

        return I_deriv

    def R_deriv(self, S, I, R, t):
        """
        Function for finding the derivative of R.
        Arguments:
            S (float): Number of susceptible individuals
            I (float): Number of infected individuals
            R (float): Number of recovered individuals
            t (float): Current time-value
        Returns:
            R_deriv (float): The derivative of R
        """
        R_deriv = self.b*I - self.c*R

        # check for model extensions
        if self.vital_dynamics and not self.vaccination:
            R_deriv -= self.d*R
        elif self.vaccination and not self.vital_dynamics:
            if isinstance(self.f, float):
                R_deriv += self.f
            else:
                R_deriv += self.f(t)
        elif self.vital_dynamics and self.vaccination:
            if isinstance(self.f, float):
                R_deriv += self.f - self.d*R
            else:
                R_deriv += self.f(t) - self.d*R

        return R_deriv

    def solve_RK4(self, S0, I0, n, t1, t2):
        """
        Function for solving the SIR-model using RK4.
        Arguments:
            S0 (int): Initial number of infected individuals
            I0 (int): Initial number of susceptible individuals
            n (int): Number of time-steps to perform
            t1 (float): Time starting point
            t2 (float): Time end point
        Returns:
            S (array): Evolution of number of susceptible individuals
            I (array): Evolution of number of infected individuals
            R (array): Evolution of number of recovered individuals
            t (array): Time array
        """
        if self.vital_dynamics and self.vr:
            raise AttributeError("Rates for population change have not been\
                                  set.")
        if self.seasonal_variation and self.vp:
            raise AttributeError("Parameters for seasonal variation have\
                                  not been set.")
        if self.vaccination and self.vacr:
            raise AttributeError("Vaccination rate has not been set.")

        # set time-step
        h = (t2 - t1)/float(n)

        # prepare arrays
        S = np.zeros(n)
        I = np.zeros(n)
        R = np.zeros(n)
        t = np.zeros(n)

        # set initial conditions
        S[0] = S0
        I[0] = I0
        t[0] = t1

        # solve
        for i in range(1, n):
            k1_S = h*self.S_deriv(S[i-1], I[i-1], R[i-1], t[i-1])
            k2_S = h*self.S_deriv(S[i-1] + k1_S/2., I[i-1] + k1_S/2.,
                                  R[i-1] + k1_S/2., t[i-1])
            k3_S = h*self.S_deriv(S[i-1] + k2_S/2., I[i-1] + k2_S/2.,
                                  R[i-1] + k2_S/2., t[i-1])
            k4_S = h*self.S_deriv(S[i-1] + k3_S, I[i-1] + k3_S, R[i-1] + k3_S,
                                  t[i-1])

            k1_I = h*self.I_deriv(S[i-1], I[i-1], t[i-1])
            k2_I = h*self.I_deriv(S[i-1] + k1_I/2., I[i-1] + k1_I/2., t[i-1])
            k3_I = h*self.I_deriv(S[i-1] + k2_I/2., I[i-1] + k2_I/2., t[i-1])
            k4_I = h*self.I_deriv(S[i-1] + k3_I, I[i-1] + k3_I, t[i-1])

            k1_R = h*self.R_deriv(S[i-1], I[i-1], R[i-1], t[i-1])
            k2_R = h*self.R_deriv(S[i-1] + k1_S/2., I[i-1] + k1_S/2.,
                                  R[i-1] + k1_S/2., t[i-1])
            k3_R = h*self.R_deriv(S[i-1] + k2_S/2., I[i-1] + k2_S/2.,
                                  R[i-1] + k2_S/2., t[i-1])
            k4_R = h*self.R_deriv(S[i-1] + k3_S, I[i-1] + k3_S, R[i-1] + k3_S,
                                  t[i-1])

            S[i] = S[i-1] + (1/6.)*(k1_S + 2*k2_S + 2*k3_S + k4_S)
            I[i] = I[i-1] + (1/6.)*(k1_I + 2*k2_I + 2*k3_I + k4_I)
            R[i] = R[i-1] + (1/6.)*(k1_R + 2*k2_R + 2*k3_R + k4_R)
            t[i] = t[i-1] + h

        return S, I, R, t

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
        if self.vital_dynamics and self.vr:
            raise AttributeError("Rates for population change have not been\
                                  set.")
        if self.seasonal_variation and self.vp:
            raise AttributeError("Parameters for seasonal variation have\
                                  not been set.")
        if self.vaccination and self.vacr:
            raise AttributeError("Vaccination rate has not been set.")

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
            if seasonal_variation:
                P_SI = (self.a_var(t)*S[i-1]*I[i-1]*dt)/float(self.N)
            else:
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

            # account for population change
            if self.vital_dynamics:
                # determine probabilities
                P_b = self.e*self.N
                P_d_S = self.d*S[i]
                P_d_I = (self.d + self.d1)*I[i]
                P_d_R = self.d*R[i]

                r1 = np.random.random()
                # determine birth / no birth
                if r1 < P_b:
                    self.N += 1
                    S[i] += 1

                # determine deaths in each group
                if np.random.random() < P_d_S:
                    self.N -= 1
                    S[i] -= 1
                elif np.random.random() < P_d_I:
                    self.N -= 1
                    I[i] -= 1
                elif np.random.random() < P_d_R:
                    self.N -= 1
                    R[i] -= 1

            t[i] = t[i-1] + dt
        return S, I, R, t
