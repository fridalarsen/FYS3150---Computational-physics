import numpy as np

def RK4(y0, t0, t1, N, func):
    """
    Function for solving a differential equation using Runge Kutta 4.
    Arguments:
        y0 (float): Initial function-value
        t0 (float): Initial time
        t1 (float): Final time
        N (int): Total number of time steps
        func (function): Derivative of function to be approximated
    Returns:
        t (array): Next time-step
        y (array): Approximated function-value
    """

    h = (t1-t0)/float(N)

    t = np.linspace(t0, t1, N)
    y = np.zeros(N)
    y[0] = y0

    for i in range(1, N):
        k1 = h*func(t[i-1], y[i-1])
        k2 = h*func(t[i-1] + h/2., y[i-1] + k1/2.)
        k3 = h*func(t[i-1] + h/2., y[i-1] + k2/2.)
        k4 = h*func(t[i-1] + h, y[i-1] + k3)

        y[i] = y[i-1] + (1/6.)*(k1 + 2*k2 + 2*k3 + k4)

    return t, y

def RK4_step(y0, t0, h, func):
    """
    Function for computing the next step of a function using Runge Kutta 4.
    Arguments:
        y0 (float): Previous function-value
        t0 (float): Previous time-value
        h (float): Step-length
        func (function): Derivative of function to be approximated
    Returns:
        t (array): Next time-step
        y (array): Approximated function-value
    """

    k1 = h*func(t0, y0)
    k2 = h*func(t0 + h/2., y0 + k1/2.)
    k3 = h*func(t0 + h/2., y0 + k2/2.)
    k4 = h*func(t0 + h, y0 + k3)

    y = y0 + (1/6.)*(k1 + 2*k2 + 2*k3 + k4)
    t = t0 + h

    return t, y
