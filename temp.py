import numpy as np
from scipy import stats

# Input data
weight = np.array(
    [
        1.20,
        1.90,
        1.94,
        1.66,
        1.63,
        1.06,
        1.21,
        1.18,
        1.69,
        1.38,
        1.77,
        1.50,
        1.72,
        1.99,
        1.38,
        1.78,
        1.93,
        1.21,
        1.65,
        1.13,
    ]
)
fuel_efficiency = np.array(
    [
        12.38,
        5.17,
        11.62,
        10.89,
        11.12,
        16.32,
        15.21,
        14.93,
        12.65,
        14.15,
        10.50,
        7.92,
        11.87,
        8.97,
        12.28,
        7.35,
        8.47,
        14.40,
        13.75,
        13.82,
    ]
)

# Basic statistics
n = len(weight)
x_mean = np.mean(weight)
y_mean = np.mean(fuel_efficiency)

# Regression coefficients
Sxy = np.sum((weight - x_mean) * (fuel_efficiency - y_mean))
Sxx = np.sum((weight - x_mean) ** 2)
print(f"{Sxy=} {Sxx=}")
b1 = Sxy / Sxx
b0 = y_mean - b1 * x_mean

# Fitted values and residuals
y_pred = b0 + b1 * weight
residuals = fuel_efficiency - y_pred
RSS = np.sum(residuals**2)
MSE = RSS / (n - 2)
SE_b1 = np.sqrt(MSE / Sxx)
SE_b0 = np.sqrt(MSE * (1 / n + x_mean**2 / Sxx))

# R^2 and correlation
TSS = np.sum((fuel_efficiency - y_mean) ** 2)
R2 = 1 - RSS / TSS
r = np.corrcoef(weight, fuel_efficiency)[0, 1]

# t-test for slope
t_stat = b1 / SE_b1
df = n - 2
p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))

# Confidence interval for b1 (99%)
alpha = 0.01
t_crit = stats.t.ppf(1 - alpha / 2, df)
ci_b1 = (b1 - t_crit * SE_b1, b1 + t_crit * SE_b1)

# Prediction at weight = 1.35
x_new = 1.35
y_new_pred = b0 + b1 * x_new

(b0, b1, R2, r, t_stat, df, p_value, ci_b1, y_new_pred)
