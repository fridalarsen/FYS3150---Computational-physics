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
        os.system("mpicxx MC_parallelize.cpp -o MC_parallelize.x")

        cpp_lib = ctypes.cdll.LoadLibrary(os.path.abspath("./project3_.so"))

        self.solve3a = cpp_lib.Solve3a
        self.solve3a.restype = ctypes.c_double
        self.solve3a.argtypes = [ctypes.c_double, ctypes.c_int, ctypes.c_double]

        self.solve3b = cpp_lib.Solve3b
        self.solve3b.restype = ctypes.c_double
        self.solve3b.argtypes = [ctypes.c_int, ctypes.c_double]

        self.solve3c = cpp_lib.Solve3c
        self.solve3c.restype = None
        self.solve3c.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_double),
                                 ctypes.POINTER(ctypes.c_double),
                                 ctypes.c_double, ctypes.c_double]

        self.solve3d = cpp_lib.Solve3d
        self.solve3d.restype = None
        self.solve3d.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_double),
                                 ctypes.POINTER(ctypes.c_double), ctypes.c_int]

    def __call__(self, task, **kwargs):
        """
        Compute integral for a given task.

        Arguments:
            task (str): What task is to be solved.
            **kwargs: See task functions.

        Returns:
            I (float): Computed integral value.
            t (float): Time taken to compute integral.
            var (float): Variance of computed integral. (Only returned for the
                         Monte Carlo methods)
        """
        tasks = ["3a", "3b", "3c", "3d", "3e"]
        if task not in tasks:
            raise ValueError(f"Task not correctly chosen. Options: {tasks}")

        if task == "3a":
            solve3X = self.run_solve3a
        elif task == "3b":
            solve3X = self.run_solve3b
        elif task == "3c":
            solve3X = self.run_solve3c
        elif task == "3d":
            solve3X = self.run_solve3d
        elif task == "3e":
            solve3X = self.run_solve3e

        if task == "3a" or task == "3b":
            I, t = solve3X(**kwargs)
            return I, t
        else:
            I, var, t = solve3X(**kwargs)
            return I, var, t

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
            t (float): Time taken to compute integral.
        """
        start = time.perf_counter()
        I = self.solve3a(float(lamb), int(N), float(tol))
        end = time.perf_counter()
        t = end-start
        return I, t

    def run_solve3b(self, N, tol=1e-8):
        """
        Function for computing the integral with Gauss-Legendre and Gauss-
        Laguerre quadrature.

        Arguments:
            N (int): Number of integration points.
            tol (float): Tolerance to avoid division by zero.

        Returns:
            I (float): Computed integral value.
            t (float): Time taken to compute integral.
        """
        start = time.perf_counter()
        I = self.solve3b(int(N), float(tol))
        end = time.perf_counter()
        t = end-start
        return I, t

    def run_solve3c(self, N, a, b):
        """
        Function for computing the integral with brute force Monte Carlo.

        Argumetns:
            N (int): Number of integration points.
            a (float): Lower integration limit.
            b (float): Upper integration limit.

        Returns:
            I (float): Computed integral value.
            var (float): Variance of computed integral.
            t (float): Time taken to compute integral.
        """
        I = ctypes.c_double()
        var = ctypes.c_double()

        start = time.perf_counter()
        self.solve3c(int(N), ctypes.byref(I), ctypes.byref(var), float(a),
                     float(b))
        end = time.perf_counter()
        t = end-start

        I = I.value
        var = var.value
        return I, var, t

    def run_solve3d(self, N, seed=None):
        """
        Function for computing the integral with Monte Carlo using importance
        sampling and spherical coordinates.

        Argumetns:
            N (int): Number of integration points.
            seed (int, optional): Seed for random number generator.

        Returns:
            I (float): Computed integral value.
            var (float): Variance of computed integral.
            t (float): Time taken to compute integral.
        """
        if seed == None:
            seed = np.random.randint(100)

        I = ctypes.c_double()
        var = ctypes.c_double()

        start = time.perf_counter()
        self.solve3d(int(N), ctypes.byref(I), ctypes.byref(var), int(seed))
        end = time.perf_counter()
        t = end-start

        I = I.value
        var = var.value
        return I, var, t

    def run_solve3e(self, N, npar):
        """
        Function for computing the integral with Monte Carlo using importance
        sampling and spherical coordinates.

        Argumetns:
            N (int): Number of integration points.
            npar (int): Number of parallel processes.

        Returns:
            I (float): Computed integral value.
            var (float): Variance of computed integral.
            t (float): Time taken to compute integral.
        """
        if os.path.exists("results_3e.txt"):
            os.remove("results_3e.txt")

        start = time.perf_counter()
        os.system(f"mpiexec -n {npar} ./MC_parallelize.x {N}")
        end = time.perf_counter()
        t = end-start

        with open("results_3e.txt", "r") as results:
            I, var = [float(line) for line in results.readlines()]

        return I, var, t

if __name__ == "__main__":
    P3 = Project3()
    I_a, t_a = P3("3a", lamb=2.2, N=15, tol=1e-8)
    I_b, t_b = P3("3b", N=15, tol=1e-8)
    I_c, var_c, t_c = P3("3c", N=500000, a=-2.2, b=2.2)
    I_d, var_d, t_d = P3("3d", N=200000)
    I_e, var_e, t_e = P3("3e", N=100000, npar=2)

    print(f"exact = {5*(np.pi**2)/(16**2)}")
    print(f"3a    = {I_a}")
    print(f"3b    = {I_b}")
    print(f"3c    = {I_c}")
    print(f"3d    = {I_d}")
    print(f"3e    = {I_e}")
