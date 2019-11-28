import numpy as np
import matplotlib.pyplot as plt
from RK4 import RK4_step

class SIR:
    def __init__(self, N, I0, S0, a, b, c):
        """
        Function for modeling an infectious disease.
        Arguments:
            N (int): Population size
            I0 (int): Initial number of infected individuals
            S0 (int): Initial number of susceptible individuals
            a (float): Rate of transmission
            b (float): Rate of recovery
            c (float): Rate of immunity loss
        """
        self.N = N
        self.I0 = I0
        self.S0 = S0
        self.a = a
        self.b = b
        self.c = c

    def S_deriv(self, S, R, I):
        """
        Function for finding the derivative of S.
        Returns:
            S_deriv (float): The derivative of S
        """
        S_deriv = self.c*R - (self.a*S*I)/float(self.N)

        return S_deriv

    def I_deriv(self, S, I):
        """
        Function for finding the derivative of I.
        Returns:
            I_deriv (float): The derivative of I
        """
        I_deriv = (self.a*S*I)/float(self.N) - self.b*I

        return I_deriv

    def R_deriv(self, I, R):
        """
        Function for finding the derivative of R.
        Returns:
            R_deriv (float): The derivative of R
        """
        R_deriv = self.b*I - self.c*R

        return R_deriv
