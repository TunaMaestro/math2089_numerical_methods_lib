from typing import Callable
from integration import NumIntResult


def numerical_integration_errors(
    method: Callable[[Callable[[float], float], float, float, int], NumIntResult],
    f,
    a,
    b,
    I,
):
    def E(N):
        v = method(f, a, b, N).Q
        err = abs(v - I)
        return err

    return E
