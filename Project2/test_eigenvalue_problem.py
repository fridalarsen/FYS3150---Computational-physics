import numpy as np
from eigenvalue_problem import EigenvalueProblem

def test_NumpyEigenvalues_diagonal():
    """
    Function for testing that NumpyEigenvalues returns the analytic eigenvalues
    for a diagonal matrix.
    """
    A = np.diag([1, 2, 3])
    Eig = EigenvalueProblem()

    calc_eig = Eig(A, "NumPy")
    exp_eig = [1, 2, 3]
    tol = 1e-5

    assert abs(calc_eig[0] - exp_eig[0]) < tol
    assert abs(calc_eig[1] - exp_eig[1]) < tol
    assert abs(calc_eig[2] - exp_eig[2]) < tol

def test_JacobiEigenvalues_diagonal():
    """
    Function for testing that JacobiEigenvalues returns the analytic eigenvalues
    for a diagonal matrix.
    """
    A = np.diag([1, 2, 3])
    Eig = EigenvalueProblem()

    calc_eig = Eig(A, "Jacobi")
    exp_eig = [1, 2, 3]
    tol = 1e-5

    assert abs(calc_eig[0] - exp_eig[0]) < tol
    assert abs(calc_eig[1] - exp_eig[1]) < tol
    assert abs(calc_eig[2] - exp_eig[2]) < tol

def test_NumpyEigenvalues():
    """
    Function for testing that NumpyEigenvalues returns the analytic eigenvalues
    for a non-diagonal matrix.
    """
    A = np.array([[6, -2, -1], [-2, 6, -1], [-1, -1, 5]])
    Eig = EigenvalueProblem()

    calc_eig = Eig(A, "NumPy")
    exp_eig = [3, 6, 8]
    tol = 1e-5

    assert abs(calc_eig[0] - exp_eig[0]) < tol
    assert abs(calc_eig[1] - exp_eig[1]) < tol
    assert abs(calc_eig[2] - exp_eig[2]) < tol

def test_JacobiEigenvalues():
    """
    Function for testing that JacobiEigenvalues returns the analytic eigenvalues
    for a non-diagonal symmetric matrix.
    """
    A = np.array([[6, -2, -1], [-2, 6, -1], [-1, -1, 5]])
    Eig = EigenvalueProblem()

    calc_eig = Eig(A, "Jacobi")
    exp_eig = [3, 6, 8]
    tol = 1e-5
    print(calc_eig)

    assert abs(calc_eig[0] - exp_eig[0]) < tol
    assert abs(calc_eig[1] - exp_eig[1]) < tol
    assert abs(calc_eig[2] - exp_eig[2]) < tol
