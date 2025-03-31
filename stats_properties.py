from dataclasses import dataclass
import numpy as np

@dataclass
class Properties:
    mean: float
    stdevp: float
    varp: float

def properties(xs):
    std = np.std(xs, ddof=1)
    return Properties(np.mean(xs), std, std ** 2)
