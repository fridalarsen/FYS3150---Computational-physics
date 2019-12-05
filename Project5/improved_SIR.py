import numpy as np
import matplotlib.pyplot as plt

class SIR_model:
    """
    Class for modeling an infectious disease.
    """
    def __init__(self, N0, vital_dynamics=False, seasonal_variation=False,
                 vaccination=False):):
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
        if seasonal_variation == False:
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
            A (float):
            omega (float):
            t (float):
            a0 (float):
        """
        self.A = A
        self.omega = omega
        self.a0 = a0

        self.vp = True

    def a_var(self, t):
        """
        Function for calculating transmission rate.
        Arguments:
            t (float): time
        Returns:
            a (float):
        """
        a = self.A*np.cos(self.omega*t) + self.a0

        return a

    def vaccination_rate(self, f=0.0):
        """
        Function for specifying the rate of vaccination.
        Arguments:
            f (float or function, optional): Rate of vaccination, defaults to 0
        """
        self.f = f

        self.vacr = True

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
        if vital_dynamics = True:

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

    def solve_RK4(self):
        """
        """
        if self.vital_dynamics == True and self.vr == False:
            raise AttributeError("Rates for population change have not been\
                                  set.")
        if self.seasonal_variation == True and self.vp = False:
            raise AttributeError("Parameters for seasonal variation have\
                                  not been set.")
        if self.vaccination == True and self.vacr == False:
            raise AttributeError("Vaccination rate has not been set.")

    def solve_MC(self):
        """
        """
        if self.vital_dynamics == True and self.vr == False:
            raise AttributeError("Rates for population change have not been\
                                  set.")
        if self.seasonal_variation == True and self.vp = False:
            raise AttributeError("Parameters for seasonal variation have\
                                  not been set.")
        if self.vaccination == True and self.vacr == False:
            raise AttributeError("Vaccination rate has not been set.")
