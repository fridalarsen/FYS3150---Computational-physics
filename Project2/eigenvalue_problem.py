import numpy as np
import ctypes
import os
from numpy.ctypeslib import ndpointer

class EigenvalueProblem:
    """
    Class for finding the eigenvalues of a matrix.
    """
    def __init__(self):
        """
        Setup c++ functions.
        """
        os.system("g++ -c -o jacobi_algorithm.o jacobi_algorithm.cpp")
        os.system("g++ -shared -fPIC jacobi_algorithm.o -o" \
                  " jacobi_algorithm.so -l armadillo")

        cpp_lib = ctypes.cdll.LoadLibrary(os.path.abspath \
                  ("./jacobi_algorithm.so"))

        doublepp = np.ctypeslib.ndpointer(dtype=np.uintp)

        self.ArmadilloEigenpairs = cpp_lib.ArmadilloEigenpairs
        self.ArmadilloEigenpairs.restype = None
        self.ArmadilloEigenpairs.argtypes = [doublepp,
                                             doublepp,
                                             doublepp,
                                             ctypes.c_int]

        self.JacobiEigenpairs2 = cpp_lib.JacobiEigenpairs2
        self.JacobiEigenpairs2.restype = None
        self.JacobiEigenpairs2.argtypes = [doublepp,
                                           ctypes.c_double,
                                           doublepp,
                                           doublepp,
                                           ctypes.c_int]

    def __call__(self, A, method, maxit=100):
        """
        Arguments:
            A (matrix): Matrix whose eigenvalues are to be determined.
            method (str): Method to be used for finding eigenvalues.
                          Options: NumPy, Jacobi_py, Jacobi_cpp, Armadillo
            maxit (float, optional): Maximum number of iterations.

        Returns:
            EigenValues (array): Eigenvalues of A determined by specified
                                 method, sorted from smallest to largest.
            EigenVectors (array): List of eigenvectors as row vectors, sorted
                                  corresponding to the eigenvalues.
        """
        def arr2pparr(x):
            xpp = (x.ctypes.data + np.arange(x.shape[0]) *\
                   x.strides[0]).astype(np.uintp)
            return xpp

        A = A.astype(np.float_)

        methods = ["NumPy", "Jacobi_py", "Jacobi_cpp", "Armadillo"]
        if method not in methods:
            raise ValueError(f"Please choose one of the following methods:\
                             {methods}")

        if method == "NumPy":
            EigenValues, EigenVectors = self.NumpyEigenpairs(A)

        elif method == "Jacobi_py":
            EigenValues, EigenVectors = self.JacobiEigenpairs1(A, maxit)

        elif method == "Jacobi_cpp":
            N = A.shape[0]
            EigenValues = np.zeros((N,1))
            EigenVectors = np.eye((N))
            self.JacobiEigenpairs2(arr2pparr(A), maxit, arr2pparr(EigenVectors),
                                   arr2pparr(EigenValues), N)
            EigenValues = EigenValues.flatten()
            idx = np.argsort(EigenValues)
            EigenValues = EigenValues[idx]
            EigenVectors = EigenVectors[:, idx]

        elif method == "Armadillo":
            N = A.shape[0]
            EigenVectors = np.eye(N)
            EigenValues = np.zeros((N,1))
            self.ArmadilloEigenpairs(arr2pparr(A), arr2pparr(EigenVectors),
                                     arr2pparr(EigenValues), N)

        return EigenValues.flatten(), EigenVectors.T

    def NumpyEigenpairs(self, A):
        """
        Function implementing NumPy's functions for diagonalizing a matrix and
        finding eigenvalues and eigenvectors.

        Arguments:
            A (matrix): Matrix whose eigenvalues are to be found.

        Returns:
            EigenValues (array): List of eigenvalues sorted from smallest to
                                 largest.
            EigenVectors (array): All eigenvectors as row vectors, sorted
                                   corresponding to the eigenvalues.
        """
        EigenValues, EigenVectors = np.linalg.eig(A)
        idx = np.argsort(EigenValues)

        EigenValues = EigenValues[idx]
        EigenVectors = EigenVectors[:,idx]

        return EigenValues, EigenVectors

    def JacobiEigenpairs1(self, A, maxit):
        """
        Function implementing Jacobi's method for diagonalizing a symmetric
        matrix and finding its eigenvalues and eigenvectors.

        Arguments:
            A (matrix): Matrix whose eigenvalues are to be found. Must be
                        symmetric.
            maxit (float): Maximum number of iterations to perform.

        Returns:
            EigenValues (array): List of eigenvalues sorted from smallest to
                                 largest.
            EigenVectors (array): All eigenvectors as row vectors, sorted
                                   corresponding to eigenvalues.
        """
        tol = 1e-12
        iterations = 0
        N = A.shape[0]
        max_offdiag = 0.

        # find maximum off-diagonal element of the initial matrix
        for i in range(N):
            for j in range(i+1, N):
                if abs(A[i,j]) > max_offdiag:
                    max_offdiag = abs(A[i,j])

        EigenVectors = np.eye(N, dtype=np.float_)

        while max_offdiag > tol and iterations <= maxit:
            # find the maximum off-diagonal element
            max_offdiag = 0
            k = 0
            l = 0
            for i in range(N):
                for j in range(i+1, N):
                    if abs(A[i,j]) > max_offdiag:
                        max_offdiag = abs(A[i,j])
                        k = i
                        l = j

            if not (k == 0 and l == 0):
                # prepare transform
                if A[k, l] == 0:
                    c = 1.
                    s = 0.
                else:
                    tau = (A[l,l]-A[k,k]) / (2*A[k,l])

                    if tau >= 0:
                        t = 1/(tau + np.sqrt(1 + tau**2))
                    else:
                        t = -1/(-tau + np.sqrt(1 + tau**2))

                    c = 1/np.sqrt(1+t*t)
                    s = c*t

                # similarity transform
                a_kk = A[k,k]
                a_ll = A[l,l]

                A[k,k] = a_kk*(c**2) - 2*A[k,l]*c*s + a_ll*(s**2)
                A[l,l] = a_ll*(c**2) + 2*A[k,l]*c*s + a_kk*(s**2)
                A[k,l] = 0.
                A[l,k] = 0.

                for i in range(N):
                    if i != k and i != l:
                        a_ik = A[i,k]
                        a_il = A[i,l]
                        A[i,k] = a_ik*c - a_il*s
                        A[k,i] = A[i,k]
                        A[i,l] = a_il*c + a_ik*s
                        A[l,i] = A[i,l]

                    # update eigenvectors
                    ev_ik = EigenVectors[i,k]
                    ev_il = EigenVectors[i,l]

                    EigenVectors[i,k] = c*ev_ik - s*ev_il
                    EigenVectors[i,l] = c*ev_il + s*ev_ik

            iterations += 1

        EigenValues = np.diag(A)
        idx = np.argsort(EigenValues)

        EigenValues = EigenValues[idx]
        EigenVectors = EigenVectors[:, idx]

        return EigenValues, EigenVectors

if __name__ == "__main__":
    f = EigenvalueProblem()
    A = np.array([[0,1,-2],[1,3,0],[-2,0,5]])

    numpy_val, numpy_vec = f(A, "NumPy")
    arma_val, arma_vec = f(A, "Armadillo")
    jacobi_cpp_val, jacobi_cpp_vec = f(A, "Jacobi_cpp", 200)
    jacobi_py_val, jacobi_py_vec = f(A, "Jacobi_py", 200)

    print(numpy_vec)
    print(numpy_val)
    print("-------------------")
    print(arma_vec)
    print(arma_val)
    print("-------------------")
    print(jacobi_cpp_vec)
    print(jacobi_cpp_val)
    print("-------------------")
    print(jacobi_py_vec)
    print(jacobi_py_val)
