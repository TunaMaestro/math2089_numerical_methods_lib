import scipy
import math
import numpy as np


def interpret(ns: str) -> list[float]:
    seps = ",;\t\n"

    ns = "".join([" " if x in seps else x for x in ns])

    ns = ns.split(" ")

    return [float(x) for x in ns if x]


# numpy requires method="hazen"
def quantiles(ns: list[float], percentiles: list[float]) -> list[float]:
    return scipy.stats.mstats.mquantiles(ns, percentiles, alphap=0.5, betap=0.5)


def five_summary(ns: list[float]) -> list[float]:
    return quantiles(ns, [0, 0.25, 0.5, 0.75, 1.0])


def q1q3(ns) -> tuple[float, float]:
    summary = five_summary(ns)
    return summary[1], summary[3]


def iqr_outliers(ns):
    summary = five_summary(ns)
    q1, q3 = q1q3(ns)

    iqr = q3 - q1

    min_cutoff = q1 - iqr * 1.5
    max_cutoff = q3 + iqr * 1.5

    return min_cutoff, max_cutoff


def filter_outliers(ns):
    _min, _max = iqr_outliers(ns)
    return [x for x in ns if not _min <= x <= _max]


# histogram
