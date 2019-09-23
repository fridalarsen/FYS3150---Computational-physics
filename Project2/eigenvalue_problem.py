import numpy as np

class EigenvalueProblem:
    """
    Class for finding the eigenvalues of a matrix.
    """
    def __call__(self, A, method, maxit=100):
        """
        Arguments:
            A (matrix): Matrix whose eigenvalues are to be determined.
            method (str): Method to be used for finding eigenvalues.
                          Options: NumPy, Jacobi
            maxit (float, optional): Maximum number of iterations.

        Returns:
            EigenValues (array): Eigenvalues of A determined by specified method.
        """
        A = A.astype(np.float_)

        methods = ["NumPy", "Jacobi"]
        if method not in methods:
            raise ValueError(f"Please choose one of the following methods:\
                             {methods}")

        if method == "NumPy":
            EigenValues = self.NumpyEigenvalues(A)

        elif method == "Jacobi":
            EigenValues = self.JacobiEigenvalues(A, maxit)

        return EigenValues


    def NumpyEigenvalues(self, A):
        """
        Function implementing NumPy's functions for diagonalizing a matrix and
        finding eigenvalues.

        Arguments:
            A (matrix): Matrix whose eigenvalues are to be found.

        Returns:
            EigenValues (array): List of eigenvalues sorted from smallest to
                                 largest.
        """
        EigenValues, EigenVectors = np.linalg.eig(A)

        EigenValues = EigenValues[np.argsort(EigenValues)]

        return EigenValues

    def JacobiEigenvalues(self, A, maxit):
        """
        Function implementing Jacobi's method for diagonalizing a symmetric
        matrix and finding its eigenvalues.

        Arguments:
            A (matrix): Matrix whose eigenvalues are to be found. Must be
                        symmetric.
            maxit (float): Maximum number of iterations to perform.

        Returns:
            EigenValues (array): List of eigenvalues sorted from smallest to
                                 largest.
        """
        tol = 1e-8
        iterations = 0
        N = A.shape[0]
        max_offdiag = 0.

        # find maximum off-diagonal element of the initial matrix
        for i in range(N):
            for j in range(i+1, N):
                if abs(A[i,j]) > max_offdiag:
                    max_offdiag = abs(A[i,j])

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

                iterations += 1

        EigenValues = np.diag(A)
        EigenValues = EigenValues[np.argsort(EigenValues)]
        return EigenValues


if __name__ == "__main__":
    f = EigenvalueProblem()
    A = np.array([[0,1,-2],[1,3,0],[-2,0,5]])

    A_jacobi = f(A, "Jacobi")
    A_numpy = f(A, "NumPy")

    print(A_jacobi)
    print(A_numpy)
