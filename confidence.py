import numpy as np
from scipy.stats import t, norm, ttest_1samp, ttest_ind, ttest_rel
from math import sqrt

from pprint import pprint
from dataclasses import dataclass

@dataclass
class Proportion:
    perc: float
    z: float
    a: float
    n: int
    np_1_minus_p: float
    low: float
    high: float

"""
  Percentile in [0, 1] i.e. 0.95 for a 95% confidence  
"""
def sample_confidence(xs: list[float], confidence: float):
    mean = np.mean(xs)
    std = np.std(xs, ddof=1)
    perc = conf_to_ppf(confidence)
    z = t.ppf(perc, df=len(xs) - 1)
    print(f"{mean=}\n{std=}\n{perc=}\n{z=}")
    a = z * std / (len(xs) ** 0.5)
    print(f"Interval: [ {mean - a} , {mean + a} ]")

"""
    Confidence as 95% to 0.975 for inverse norm.
"""
def conf_to_ppf(conf: float) -> float:
    return 1 - ((1 - conf) / 2 )
    
def required_sample(z, std, err):
    return (z * std / err) ** 2

def proportion(yes: int, n: int, confidence: float):
    perc = conf_to_ppf(confidence)
    z = t.ppf(perc, df=n-1)
    pp = yes / n
    a = z * sqrt(pp*(1-pp)/n)
    print(f"{perc=}\n"
          f"{z=}\n"
          f"{a=}\n"
          f"np(1-p)={n * pp * (1-pp)}")
    low = pp - a
    high = pp + a
    print(f"[ {low} , {high} ]")
    return Proportion(perc, z, a, n, n * pp * (1-pp), low, high)

def required_sample_prop(yes: int, n: int, confidence: float, error: float):
    perc = conf_to_ppf(confidence)
    z = t.ppf(perc, df=n-1)
    z = norm.ppf(perc)
    pp = yes / n
    var = n * pp * (1 - pp)
    sd_P = (var / (n)) ** 0.5
    return required_sample(z, sd_P, error)

# when taking the result of this, the P val is correct for H₀: == and Hₐ: !=
def ttest(xs, expected_mean):
    return ttest_1samp(xs, expected_mean)

def test_ineq_p(res):
    test = res.statistic
    left_tail = t.cdf(test, res.df)
    right_tail = 1 - left_tail
    print(f"{left_tail=}\n{right_tail=}")
    

def ttest_prop(yes, n, expected):
    pp = yes / n
    z = (pp - expected) / sqrt(expected * (1 - expected) / n)
    P = t.cdf(z, df=n-1)
    print(f"z = {z}")
    print(f"P = {P}")
    print(f"nπ(1-π) = {n * expected * (1 - expected)}")
    print(f"np(1-p) = {n * pp * (1 - pp)}")

# use ttest_ind for comparison experiments

def two_means_confidence(xs1: list[float], xs2: list[float], confidence: float):
    ttest_res = ttest_ind(xs1, xs2)
    pprint(ttest_res)
    ci = ttest_res.confidence_interval(confidence)
    pprint(ci)

def partition_pred(xs, f):
    return [x for x in xs if f(x)], [x for x in xs if not f(x)]

# use ttest_rel(x1, x2) for pairwise
