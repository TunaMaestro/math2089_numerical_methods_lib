import math
from scipy.stats import poisson, binom, norm, expon

# Returns a function that evaluats the poisson CDF at x
# poisson mean = lambda
def p_poisson(_lambda: float):
    return lambda x: poisson.cdf(x, mu=_lambda)


def p_norm(mean=0, stdev=1):
    return lambda x: norm.cdf(x, loc=mean, scale=stdev)

def p_invnorm(mean=0, stdev=1):
    return lambda x: norm.ppf(x, loc=mean, scale=stdev)

def p_expon(mu):
    return lambda x: expon.cdf(x, scale=mu)

def p_invexpon(mu):
    return lambda x: expon.ppf(x, scale=mu)

# binom.cdf(k=4, n=14, p=0.1818)
    
# Formula:
# def poisson(_lambda: float, x):
#     return math.exp(-_lambda) * _lambda ** x / (math.factorial(x))

# # scipy.stats.poisson.pmf(10, mu=20) mu is lambda
#
# Since it was derived as a limit case of the Binomial distribution when n is 'large' and π is 'small', one can expect the Poisson distribution to be a good approximation to Bin(n,π) in that case


# sum(f(x) * poisson.pmf(x, mu=3.5) for x in range(100))
