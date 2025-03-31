import dists
import stats_properties
import confidence
from pprint import pprint


# Laptops

def laptops(time_a: float, time_b: float, chance: float, fn_const: float, fn_surd: float):
    diff = time_b - time_a
    time_below = (1 - chance) * diff + time_a

    F = lambda x: (fn_const * x + fn_surd * 2/3 * (x ** (3/2))) / diff

    expected = F(time_b) - F(time_a)

    print(f"{time_below=}\n{expected=}")

# Given Z∼N(0,1), use Matlab or Python to calculate a value z∗
# such that P(−z∗<Z<z∗)=0.99

def symmetric_z_for_perc(perc: float):
    return dists.p_invnorm()(perc / 2 + 0.5)

def detonators(yes: int, n: int, conf: float):
    print(f"Sample prop: {yes / n}")
    prop = confidence.proportion(yes, n, conf)

# It is recommended that each month you don't eat any more than 600 grams
# of prawns caught in Sydney Harbour (due to dioxin contamination).
#
# Prawns do NOT need to be normally distributed


# Upper limit exceeded 5% of the time means upper_limit_perc=0.95
def prawns(mean, std, N, upper_limit_perc):
    chance_exceed = 1 - dists.p_norm(mean * N, std * (N ** 0.5))(600)

    max_change = upper_limit_perc

    upper_limit = dists.p_invnorm(mean * N, std * (N ** 0.5))(max_change)

    print(f"""{chance_exceed=}
    {upper_limit=}""")

# Ignition time
# Requires normal distribution

def ignition(xs: list[float], conf: float):
    print("\x1b[31mCHECK VAR OR STD\x1b[m")
    props = stats_properties.properties(xs)
    pprint(props)
    confidence.sample_confidence(xs, conf)
    print("\x1b[31mCHECK VAR OR STD\x1b[m")

# Bolts:
# 1) ppf = dists.p_norm()(diff / σ * sqrt(N))
#    within = (ppf - 1/2) * 2
#    outside = 1 - within
# 2) n = (σ / diff * Z) ** 2

if __name__ == "__main__":
    mean = 15.1
    std = 4.5
    N = 42
    prawns(mean, std, N)


