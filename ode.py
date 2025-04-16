from typing import Callable, Tuple

from numpy import arange, hstack, size, zeros, array
from scipy import integrate

import solving


def apply_eulers(f, h) -> Callable[[Tuple[float, float]], Tuple[float, float]]:
    def a(x):
        t, y = x
        return t + h, y + h * f(t, y)

    return a


"""
MATH2089 ode solvers: math2089euler, math2089heun, math2089rk4
(Frances Kuo 25 March 2023)
"""


def math2089euler(f, t0, tmax, N, y0):
    """
    Euler's method for a system of first order ODEs
      y' = f(t,y) for t in [t0, tmax] using N steps

    Parameters
    ----------
    f    : function f(t,y) returns an array of size m
    t0   : initial time
    tmax : final time
    N    : number of time steps
    y0   : initial value as an array of size m

    Returns
    -------
    t    : an array of size N+1
    y    : a matrix of size N+1 by m

    (Frances Kuo 25 March 2023)
    """
    m = size(y0)  # m is the dimension of the system
    h = (tmax - t0) / N  # h is the step size
    t = arange(t0, tmax + h / 2, h)  # t is an array of size N+1
    y = zeros((N + 1, m))  # y is a matrix of size N+1 by m

    y[0, :] = y0  # y0 is an array of size N+1
    for n in range(N):  # f returns an array of size m
        y[n + 1, :] = y[n, :] + h * f(t[n], y[n, :])  # Euler's method

    return t, y


###########################################################################


# HEUN IS order h²
def math2089heun(f, t0, tmax, N, y0):
    """
    Heun's method for a system of first order ODEs
      y' = f(t,y) for t in [t0, tmax] using N steps

    Parameters
    ----------
    f    : function f(t,y) returns an array of size m
    t0   : initial time
    tmax : final time
    N    : number of time steps
    y0   : initial value as an array of size m

    Returns
    -------
    t    : an array of size N+1
    y    : a matrix of size N+1 by m

    (Frances Kuo 25 March 2023)
    """
    m = size(y0)  # m is the dimension of the system
    h = (tmax - t0) / N  # h is the step size
    t = arange(t0, tmax + h / 2, h)  # t is an array of size N+1
    y = zeros((N + 1, m))  # y is a matrix of size N+1 by m

    y[0, :] = y0  # y0 is an array of size N+1
    for n in range(N):  # f returns an array of size m
        fn = f(t[n], y[n, :])
        ynp = y[n, :] + h * fn  # predictor
        y[n + 1, :] = y[n, :] + (h / 2) * (fn + f(t[n + 1], ynp))  # corrector

    return t, y


###########################################################################


def math2089rk4(f, t0, tmax, N, y0):
    """
    A 4-stage Runge-Kutta method for a system of first order ODEs
      y' = f(t,y) for t in [t0, tmax] using N steps

    Parameters
    ----------
    f    : function f(t,y) returns an array of size m
    t0   : initial time
    tmax : final time
    N    : number of time steps
    y0   : initial value as an array of size m

    Returns
    -------
    t    : an array of size N+1
    y    : a matrix of size N+1 by m

    (Frances Kuo 25 March 2023)
    """
    m = size(y0)  # m is the dimension of the system
    h = (tmax - t0) / N  # h is the step size
    t = arange(t0, tmax + h / 2, h)  # t is an array of size N+1
    y = zeros((N + 1, m))  # y is a matrix of size N+1 by m

    y[0, :] = y0  # y0 is an array of size N+1
    for n in range(N):  # f returns an array of size m
        k1 = f(t[n], y[n, :])
        k2 = f(t[n] + h / 2, y[n, :] + (h / 2) * k1)
        k3 = f(t[n] + h / 2, y[n, :] + (h / 2) * k2)
        k4 = f(t[n] + h, y[n, :] + h * k3)
        y[n + 1, :] = y[n, :] + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)  # RK4

    return t, y


def rk2_step(f, h, pair):
    t, y = pair
    k1 = f(t, y)
    k2 = f(t + 2 / 3 * h, y + 2 / 3 * h * k1)
    y_next = y + h / 4 * (k1 + 3 * k2)
    print(f"k1 = {k1}\nk2 = {k2}")
    return [t + h, y_next]


def solve_ivp(f, t0, tmax, y0):
    output = integrate.solve_ivp(f, [t0, tmax], y0, rtol=1e-7, atol=1e-10)
    t = output.t
    y = output.y
    return t, y.T
    # y_like_in_lecs = y.T


def solve_bvp(f, t_a, t_b, y_a, y_b, eta_low, eta_high):
    def distance_f(test_eta):
        print(f"η or 𝛈={test_eta}")
        x0 = array([y_a, test_eta])
        result = integrate.solve_ivp(f, [t_a, t_b], x0, rtol=1e-7, atol=1e-10)
        y = result.y.T[
            -1, 0
        ]  # Last value of y in the time series, and take the value (not y')
        distance = y - y_b
        return distance

    solving.brentq(distance_f, eta_low, eta_high)


def zip_res(r):
    t, y = r
    t = t[:, None]
    return hstack([t, y])
