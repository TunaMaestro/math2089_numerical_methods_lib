from dataclasses import field
from typing import List, Optional
from pandas import DataFrame
from statsmodels.regression.linear_model import RegressionResultsWrapper
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd
import probplot

from dataclasses import dataclass
from itertools import combinations
import numpy as np
import scipy.stats as stats
from collections import namedtuple

ALPHA = 0.05


PairwiseComparison = namedtuple(
    "PairwiseComparison",
    [
        "group_a",
        "group_b",
        "t_stat",
        "p_uncorrected",
        "p_bonferroni",
        "comparison_count",
    ],
)


def anova_from_dataframe(df: DataFrame) -> tuple[DataFrame, RegressionResultsWrapper]:
    df["x"] = df["x"].astype("category")
    model = smf.ols("y ~ C(x)", data=df).fit()
    anova = sm.stats.anova_lm(model, typ=2)
    print(anova)
    return anova, model


def anova_with_block_from_dataframe(
    df: DataFrame,
) -> tuple[DataFrame, RegressionResultsWrapper]:
    df["x"] = df["x"].astype("category")
    df["block"] = df["block"].astype("category")
    model = smf.ols("y ~ C(x) + C(block)", data=df).fit()
    anova = sm.stats.anova_lm(model, typ=2)
    print(anova)
    return anova, model


def plot_residuals_vs_fitted(model: RegressionResultsWrapper) -> None:
    residuals = model.resid
    fitted = model.fittedvalues
    plt.figure()
    plt.scatter(fitted, residuals)
    plt.axhline(0, linestyle="--", color="gray")
    plt.xlabel("Fitted Values")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Fitted")
    plt.tight_layout()
    plt.show()


def plot_residuals_qq(model: RegressionResultsWrapper) -> None:
    residuals = model.resid
    probplot.pplot(residuals)


def compute_pairwise_bonferroni(
    model: RegressionResultsWrapper,
    df: DataFrame,
) -> List[PairwiseComparison]:
    groups = df["x"].cat.categories
    group_vals = {g: df[df["x"] == g]["y"].values for g in groups}
    n_groups = len(groups)
    df_resid = model.df_resid
    mse = np.sum(model.resid**2) / df_resid
    s_pooled = np.sqrt(mse)
    K = n_groups * (n_groups - 1) // 2
    results: List[PairwiseComparison] = []

    for a, b in combinations(groups, 2):
        ya = group_vals[a]
        yb = group_vals[b]
        na = len(ya)
        nb = len(yb)
        mean_diff = np.mean(ya) - np.mean(yb)
        se = s_pooled * np.sqrt(1 / na + 1 / nb)
        t_stat = mean_diff / se
        p_uncorrected = 2 * stats.t.sf(np.abs(t_stat), df_resid)
        p_bonferroni = min(p_uncorrected * K, 1.0)
        results.append(
            PairwiseComparison(str(a), str(b), t_stat, p_uncorrected, p_bonferroni, K)
        )

    return results


def print_pairwise_results(results: List[PairwiseComparison], alpha: float) -> None:
    if not results:
        return
    K = results[0].comparison_count
    print(f"Bonferroni correction factor K = {K}\n")

    for r in results:
        output = (
            f"{r.group_a} vs {r.group_b} | "
            f"t = {r.t_stat:.4f}, "
            f"p_uncorrected = {r.p_uncorrected:.4f}, "
            f"p_bonferroni = {r.p_bonferroni:.4f}"
        )

        # Check if p_bonferroni is less than alpha and make it bold
        if r.p_bonferroni < alpha:
            output = f"\033[1;32m{output} !!\033[0m"  # ANSI escape code for bold

        print(output)


@dataclass
class AnovaDataset:
    xs: list
    ys: list
    blocks: Optional[list] = None
    model: RegressionResultsWrapper = field(init=False)

    def __post_init__(self) -> None:
        if self.blocks is None:
            _, m = anova_from_dataframe(self.to_dataframe())
        else:
            _, m = anova_with_block_from_dataframe(self.to_dataframe())
        self.model = m

    def to_dataframe(self) -> DataFrame:
        df = pd.DataFrame({"x": self.xs, "y": self.ys})
        df["x"] = df["x"].astype("category")
        if self.blocks is not None:
            df["block"] = self.blocks
            df["block"] = df["block"].astype("category")
        return df

    def run_anova(self) -> tuple[DataFrame, RegressionResultsWrapper]:
        return anova_from_dataframe(self.to_dataframe())

    def run_anova_with_block(self) -> tuple[DataFrame, RegressionResultsWrapper]:
        return anova_with_block_from_dataframe(self.to_dataframe())

    def plot_residuals_fitted(self) -> None:
        plot_residuals_vs_fitted(self.model)

    def plot_residuals_qq(self) -> None:
        plot_residuals_qq(self.model)

    def pairwise_comparisons(self) -> List[PairwiseComparison]:
        return compute_pairwise_bonferroni(self.model, self.to_dataframe())

    def print_pairwise(self) -> None:
        print_pairwise_results(self.pairwise_comparisons(), ALPHA)


"""
F distribution:

k-1, n-k

Correct assumptions:
Errors are independent between and within samples - depends on the design, cannot be checked from given information

Errors come from a normal distribution - looks fine from normal quantile plot

Errors have equal variance across groups - seems reasonable since sample standard deviations are within a factor of two of each other

    
"""
