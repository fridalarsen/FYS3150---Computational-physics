import numpy as np
import matplotlib.pyplot as plt
import ctypes
import os
from numpy.ctypeslib import ndpointer
import time

class Project3:
    """
    Class for solving project 3.
    """
    def __init__(self):
        """
        Setup c++ functions.
        """
        os.system("g++ -std=c++11 -o project3_.o -c -O3 project3_.cpp")
        os.system("g++ -shared -fPIC -o project3_.so project3_.o")

        cpp_lib = ctypes.cdll.LoadLibrary(os.path.abspath("./project3_.so"))

        self.solve3a = cpp_lib.Solve3a
        self.solve3a.restype = ctypes.c_double
        self.solve3a.argtypes = [ctypes.c_double, ctypes.c_int, ctypes.c_double]

        self.solve3b = cpp_lib.Solve3b
        self.solve3b.restype = ctypes.c_double
        self.solve3b.argtypes = [ctypes.c_int, ctypes.c_double]

    def __call__(self, task, **kwargs):
        """
        Compute integral for a given task.

        Arguments:
            task (str): What task is to be solved.
            **kwargs: See task functions.

        Returns:
            I (float): Computed integral value.
            t (float): Time taken to compute integral.
        """
        tasks = ["3a", "3b"]
        if task not in tasks:
            raise ValueError(f"Task {task} is invalid. Options: {tasks}")

        if task == "3a":
            solve3X = self.run_solve3a
        elif task == "3b":
            solve3X = self.run_solve3b

        start = time.perf_counter()
        I = solve3X(**kwargs)
        end = time.perf_counter()
        t = end-start
        return I, t

    def run_solve3a(self, lamb, N, tol=1e-8):
        """
        Function for computing the integral with Gauss-Legendre quadrature.

        Arguments:
            lamb (float): Limits of integration.
            N (int): Number of integration points.
            tol (float): Tolerance for Newton's method for finding zeros of
                         Legendre polynomials.

        Returns:
            I (float): Computed integral value.
        """
        I = self.solve3a(float(lamb), int(N), float(tol))
        return I

    def run_solve3b(self, N, tol=1e-8):
        """
        Function for computing the integral with Gauss-Legendre and Gauss-
        Laguerre quadrature.

        Arguments:
            N (int): Number of integration points.
            tol (float): Tolerance to avoid division by zero.

        Returns:
            I (float): Computed integral value.
        """
        I = self.solve3b(int(N), float(tol))
        return I


if __name__ == "__main__":
    P3 = Project3()
    I_a, t_a = P3("3a", lamb=2.2, N=15, tol=1e-8)
    I_b, t_b = P3("3b", N=15, tol=1e-8)

    print(f"exact = {5*(np.pi**2)/(16**2)}")
    print(f"3a    = {I_a}")
    print(f"3b    = {I_b}")
