import numpy as np

class EigenvalueProblem:
    """
    Class for finding the eigenvalues of a matrix.
    """

    def __init__(self, A):
        """
        Arguments:
            A (matrix): Symmetric matrix whose eigenvalues are to be found.
        """
        self.A = A
        self.N = len(A[0])

    def NumpyEigenvalues(self):
        """
        Function implementing NumPy's functions for diagonalizing a matrix and
        finding eigenvalues.

        Returns:
            EigenValues (array): List of eigenvalues.
        """
        EigenValues, EigenVectors = np.linalg.eig(self.A)

        return EigenValues 




if __name__ == "__main__":
    lol = 1
