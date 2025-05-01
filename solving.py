from scipy.optimize import brentq


# brentq(f, a, b)


def newtons_method(f, df, x1, max_steps=10, tol=1e-10):
    """
    Perform Newton's method for a function f with derivative df.

    Parameters:
    - f: function, the function to find the root of
    - df: function, the derivative of f
    - x0: float, initial guess
    - max_steps: int, maximum number of iterations
    - tol: float, tolerance for stopping

    Returns:
    - x: float, the estimated root
    - steps: int, number of steps performed
    """
    x = x1
    for step in range(max_steps):
        print(f"x{step + 1} = {x}")
        fx = f(x)
        dfx = df(x)
        if abs(dfx) < 1e-14:
            raise ZeroDivisionError("Derivative near zero")
        x_new = x - fx / dfx
        if abs(x_new - x) < tol:
            return x_new, step + 1
        x = x_new
    return x, max_steps
