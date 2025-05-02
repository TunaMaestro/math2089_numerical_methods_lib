from numpy import linspace, hstack, ones, sum, concat
from scipy.special import roots_legendre
import scipy
import dataclasses

@dataclasses.dataclass
class NumIntResult:
    xs: list[float]
    weights: list[float]
    h: float
    Q: float

def trap(f, a, b, N):
    h = (b - a) / N

    xs = linspace(a, b, N+1)

    weights = h * hstack([1/2, ones(N-1), 1/2])

    Q = sum(weights * f(xs))

    return NumIntResult(
        xs, weights, h, Q
    )

def simp(f, a, b, N):
    h = (b - a) / N
    xs = linspace(a, b, N + 1)
    weights = concat([[1], [ 2 if (x % 2 == 1) else 4 for x in range(N-1) ], [1]])

    weights = h/3 * weights

    Q = sum(weights * f(xs))

    return NumIntResult(xs, weights, h, Q)

def legendre(f, a, b, N):
    xs, weights = roots_legendre(N)
    xs_ab = (a+b)/2 + (b-a)/2 * xs
    weights_ab = (b-a)/2 * weights

    Q = sum(weights_ab * f(xs_ab))

    return NumIntResult(xs_ab, weights_ab, None, Q)

def error_function(integration, f, a, b, I):
    return lambda N: abs(integration(f, a, b, N).Q - I)

def adaptive(f, a, b):
    # returns Q, abs error estimate
    Q, err = scipy.integrate.quad(f, a, b, epsabs=1e-10, epsrel=2e-8)
    return Q
