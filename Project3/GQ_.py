import numpy as np
import matplotlib.pyplot as plt
import ctypes
import os
from numpy.ctypeslib import ndpointer

class NumInt:
    """
    Class for integrating a function numerically.
    """
    def __init__(self):
        """
        Setup c++ functions.
        """
        os.system("g++ -std=c++11 -c GQ.cpp -o GQ.o")
        os.system("g++ -shared -fPIC GQ.o -o GQ.so")

        cpp_lib = ctypes.cdll.LoadLibrary(os.path.abspath("./GQ.so"))

        doublepp = np.ctypeslib.ndpointer(dtype=np.uintp)

        self.GaussLegendre = cpp_lib.GaussLegendre
        self.GaussLegendre.restype = None
        self.GaussLegendre.argtypes = [ctypes.c_double, ctypes.c_double,
                                       doublepp, doublepp, ctypes.c_int,
                                       ctypes.c_double]

        self.GaussLaguerre = cpp_lib.GaussLaguerre
        self.GaussLaguerre.restype = None
        self.GaussLaguerre.argtypes = [doublepp, doublepp, ctypes.c_int,
                                       ctypes.c_double]

    @staticmethod
    def npy2cpp(x):
        """
        Function for converting a numpy array into a c++ readable array.
        """
        xpp = (x.ctypes.data + np.arange(x.shape[0]) *\
               x.strides[0]).astype(np.uintp)
        return xpp

    @staticmethod
    def estimate_integral(f, x, w, vectorized=True):
        """
        Function for calculating integral as a sum of function values and
        weights.

        Arguments:
            f (function): Function to be integrated.
            x (array): Points where function is to be evaluated.
            w (array): Weights corresponding to each evaluation point.
            vectorized (bool): True if f supports vectorized computations.

        Returns:
            I (float): Estimated value of integral.

        """
        if vectorized == True:
            y = f(x)
        else:
            y = np.zeros(len(x))
            for i in range(len(x)):
                y[i] = f(x[i])


        I = np.sum(w*y)
        return I


    def __call__(self, f, N, method="GaussLegendre", vectorized=True,
                 **kwargs):
        """
        Arguments:
            f (function): Function to be integrated.
            N (int): Number of integration points.
            method (str): Method with which to integrate.
                          Options: GaussLegendre, GaussLaguerre

        GaussLegendre **kwargs:
            a (float): Lower integration limit.
            b (float): Upper integration limit.
            tol (float): Tolerance for Newton's method.

        GaussLaguerre **kwargs:
            alf (float): Weight function exponent.
        """
        methods = ["GaussLegendre", "GaussLaguerre"]

        if method not in methods:
            raise ValueError(f"Please choose one of the following methods:\
                             {methods}")
        try:
            N = int(N)
        except:
            print("Invalid/missing N, using default N=10")
            N = 10

        if method == "GaussLegendre":
            try:
                tol = float(kwargs["tol"])
            except:
                print("Invalid/missing tolerance, using default tol=1e-5")
                tol = 1e-5
            try:
                a = float(kwargs["a"])
                b = float(kwargs["b"])
            except:
                print("Invalid/missing interval limits. Using default a=0, b=1")
                a = 0
                b = 1

            x, w = self.Gauss_Legendre(a, b, N, tol)

        elif method == "GaussLaguerre":
            try:
                alf = float(kwargs["alf"])
            except:
                raise ValueError("Missing alpha in weight function.")

            x, w = self.Gauss_Laguerre(N, alf)

        I = NumInt.estimate_integral(f, x, w, vectorized)
        return I

    def Gauss_Legendre(self, a, b, N, tol):
        """
        Function for computing mesh points and weights using the Gauss-
        Legendre quadrature.
        """
        x = np.zeros((N,1))
        w = np.zeros((N,1))
        self.GaussLegendre(a, b, NumInt.npy2cpp(x), NumInt.npy2cpp(w),
                           N, tol)
        x = x.flatten()
        w = w.flatten()
        return x, w

    def Gauss_Laguerre(self, N, alf):
        """
        Function for computing mesh points and weights using the Gauss-
        Laguerre quadrature.
        """
        x = np.zeros((N+1,1))
        w = np.zeros((N+1,1))
        self.GaussLaguerre(NumInt.npy2cpp(x), NumInt.npy2cpp(w), N, alf)
        x = x.flatten()[:N]
        w = w.flatten()[:N]
        return x, w
