import numpy as np
import matplotlib.pyplot as plt

class SIR_model:
    """
    Class for modeling an infectious disease.
    """
    def __init__(self, N0):
        """
        Arguments:
            N (int): Initial size of population
        """
        self.N = N0

    def set_disease_parameters(self, a=1.0, b=1.0, c=1.0):
        """
        Function for specifying the disease parameters.
        Arguments:
            a (float, optional): Rate of transmission, defaults to 1
            b (float, optional): Rate of recovery, defaults to 1
            c (float, optional): Rate of immunity loss, defaults to 1
        """
        self.a = a
        self.b = b
        self.c = c
