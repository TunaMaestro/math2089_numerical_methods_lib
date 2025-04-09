from typing import Callable, Tuple
def apply_eulers(f, h) -> Callable[Tuple[float, float], Tuple[float, float]]:
    def a(x):
        t, y = x
        return t + h, y + h * f(t, y)
    return a

"""
MATH2089 ode solvers: math2089euler, math2089heun, math2089rk4
(Frances Kuo 25 March 2023)
"""

from numpy import size, arange, zeros

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
    m = size(y0)                   # m is the dimension of the system
    h = (tmax - t0) / N            # h is the step size
    t = arange(t0, tmax + h/2, h)  # t is an array of size N+1
    y = zeros((N+1, m))            # y is a matrix of size N+1 by m

    y[0,:] = y0                    # y0 is an array of size N+1
    for n in range(N):             # f returns an array of size m
        y[n+1,:] = y[n,:] + h * f(t[n], y[n,:])  # Euler's method

    return t, y

###########################################################################

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
    m = size(y0)                   # m is the dimension of the system
    h = (tmax - t0) / N            # h is the step size
    t = arange(t0, tmax + h/2, h)  # t is an array of size N+1
    y = zeros((N+1, m))            # y is a matrix of size N+1 by m

    y[0,:] = y0                    # y0 is an array of size N+1
    for n in range(N):             # f returns an array of size m
        fn = f(t[n], y[n,:])
        ynp = y[n,:] + h * fn                                # predictor
        y[n+1,:] = y[n,:] + (h/2) * ( fn + f(t[n+1], ynp) )  # corrector
        
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
    m = size(y0)                   # m is the dimension of the system
    h = (tmax - t0) / N            # h is the step size
    t = arange(t0, tmax + h/2, h)  # t is an array of size N+1
    y = zeros((N+1, m))            # y is a matrix of size N+1 by m

    y[0,:] = y0                    # y0 is an array of size N+1
    for n in range(N):             # f returns an array of size m
        k1 = f(t[n],       y[n,:]             )
        k2 = f(t[n] + h/2, y[n,:] + (h/2) * k1)
        k3 = f(t[n] + h/2, y[n,:] + (h/2) * k2)
        k4 = f(t[n] + h,   y[n,:] +     h * k3)
        y[n+1,:] = y[n,:] + (h/6) * ( k1 + 2*k2 + 2*k3 + k4 )  # RK4
        
    return t, y
