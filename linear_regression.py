# S = 1/dof *  sum((y_i - y(x)) ** 2 for x, y_i in oxy)
# S_xx = sum((x - xbar) ** 2 for x in xs)
# b1/sqrt(S/S_xx)

from dataclasses import dataclass
import numpy as np
from scipy import stats


def linear(xs, b0, b1):
    return b0 + b1 * np.array(xs)


@dataclass
class ConfidenceIntervalReport:
    t_value: float
    p_value: float
    ci_b1_low: float
    ci_b1_high: float
    alpha: float
    df: int

    def report(self) -> str:
        return (
            f"t-value: {self.t_value:.4f}\n"
            f"p-value: {self.p_value:.4f}\n"
            f"{int((1 - self.alpha) * 100)}% Confidence interval for β₁: "
            f"[ {self.ci_b1_low:.2f} ,  {self.ci_b1_high:.2f} ]\n"
            f"Degrees of freedom: {self.df}\n"
        )


@dataclass
class RegressionResult:
    xs: list[float]
    ys: list[float]

    n: int
    degrees_of_freedom: int

    b0: float
    b1: float

    predicted: list[float]
    residues: list[float]

    R_squared: float
    reduced_R_squared: float

    Sxx: float
    Syy: float

    covariance: float
    correlation: float

    def eval(self, xs):
        return linear(xs, self.b0, self.b1)

    def confidence_interval(self, confidence) -> ConfidenceIntervalReport:
        alpha = 1 - confidence
        mse = np.sum(np.array(self.residues) ** 2) / self.degrees_of_freedom
        se_b1 = np.sqrt(mse / self.Sxx)
        t_stat = self.b1 / se_b1
        p_val = 2 * (1 - stats.t.cdf(abs(t_stat), self.degrees_of_freedom))
        t_crit = stats.t.ppf(1 - alpha / 2, self.degrees_of_freedom)
        ci_low = self.b1 - t_crit * se_b1
        ci_high = self.b1 + t_crit * se_b1

        return ConfidenceIntervalReport(
            t_value=t_stat,
            p_value=p_val,
            ci_b1_low=ci_low,
            ci_b1_high=ci_high,
            alpha=alpha,
            df=self.degrees_of_freedom,
        )

    def report(self) -> str:
        return (
            f"Linear Regression Summary\n"
            f"             {'-' * 30}\n"
            f"n:           {self.n}\n"
            f"DoF:         {self.degrees_of_freedom}\n"
            f"b0:          {self.b0:.4f}\n"
            f"b1:          {self.b1:.4f}\n"
            f"R²:          {self.R_squared:.3f}\n"
            f"Correlation: {self.correlation:.4f}\n"
        )

    def question(self, conf, eval_point):
        print(self.report())
        print(self.confidence_interval(conf).report())

        print(f"Model at x={eval_point}:  {self.eval(eval_point)}")


def compute_regression(xs_: list[float], ys_: list[float]) -> RegressionResult:
    xs = np.array(xs_)
    ys = np.array(ys_)

    n = len(xs)
    x_mean = np.mean(xs)
    y_mean = np.mean(ys)

    Sxx = np.sum((xs - x_mean) ** 2)
    Syy = np.sum((ys - y_mean) ** 2)
    Sxy = np.sum((xs - x_mean) * (ys - y_mean))

    b1 = Sxy / Sxx
    b0 = y_mean - b1 * x_mean

    predicted = linear(xs, b0, b1)
    residues = ys - predicted

    RSS = np.sum(residues**2)
    TSS = Syy

    R_squared = 1 - RSS / TSS
    df = n - 2
    reduced_R_squared = 1 - ((1 - R_squared) * (n - 1)) / df

    covariance = Sxy / (n - 1)
    correlation = Sxy / np.sqrt(Sxx * Syy)

    return RegressionResult(
        xs=xs.tolist(),
        ys=ys.tolist(),
        n=n,
        degrees_of_freedom=df,
        b0=b0,
        b1=b1,
        predicted=predicted.tolist(),
        residues=residues.tolist(),
        R_squared=R_squared,
        reduced_R_squared=reduced_R_squared,
        Sxx=Sxx,
        Syy=Syy,
        covariance=covariance,
        correlation=correlation,
    )
