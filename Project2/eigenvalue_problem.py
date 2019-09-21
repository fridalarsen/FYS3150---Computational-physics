import numpy as np

class EigenvalueProblem:
    """
    Class for finding the eigenvalues of a matrix.
    """
    def __call__(self, A):
        """
        Arguments:
            A (matrix): Matrix whose eigenvalues are to be determined.
            method (str): Method to be used for finding eigenvalues.
                          Options: NumPy

        Returns:
            eig (array): Eigenvalues of A determined by specified method.
        """
        methods = ["NumPy"]
        if method not in methods:
            raise ValueError(f"Please choose one of the following methods:\
                             {methods}")

        if method == "NumPy":
            return self.NumpyEigenvalues(A)


    def NumpyEigenvalues(self, A):
        """
        Function implementing NumPy's functions for diagonalizing a matrix and
        finding eigenvalues.

        Returns:
            EigenValues (array): List of eigenvalues sorted from smallest to
                                 largest.
        """
        EigenValues, EigenVectors = np.linalg.eig(A)

        EigenValues = EigenValues[np.argsort(EigenValues)]

        return EigenValues




if __name__ == "__main__":
    lol = 1
