"""The statistics every cell gets: seed-level tests and pooled descriptive numbers.

The seed is the unit of inference (decision D7). A seed trains one adapter and evaluates
it on its own eval ids, and every condition of a seed shares those ids, so a test pairs
two conditions within each seed and runs over the per-seed means: the mean difference,
a t interval with n_seeds - 1 degrees of freedom, and the two-sided paired t.

Pooled response-level numbers (Cohen's d, rates by label or length) are descriptive.
Their intervals come from one bootstrap that resamples prompt ids, because a prompt can
sit in several seeds' draws and responses to one prompt are not independent.

Library calls: scipy.stats (t.ppf, ttest_rel, fisher_exact, pearsonr) and
statsmodels.stats.multitest.multipletests(method="holm"). Cohen's kappa is written out
(kappa below), because the bootstrap scores thousands of tables at once; a test checks it
against sklearn.metrics.cohen_kappa_score.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


def t_interval(values, level: float = 0.95) -> tuple[float, float, float]:
    """Mean and two-sided t interval with n - 1 df. Returns (mean, lo, hi)."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(values.mean())
    if n < 2:
        return mean, float("nan"), float("nan")
    se = float(values.std(ddof=1)) / math.sqrt(n)
    half = float(stats.t.ppf(0.5 + level / 2, n - 1)) * se
    return mean, mean - half, mean + half


def paired_test(a, b) -> dict:
    """Seed-level paired comparison a - b over the seeds where both have a value.

    A seed with no value in either condition (for example a seed whose dimension grades
    all failed) is left out, and n_seeds says how many remain. With fewer than two
    seeds there is no interval and no test.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    diffs = a - b
    mean, lo, hi = t_interval(diffs, 0.95)
    _, lo90, hi90 = t_interval(diffs, 0.90)
    if len(diffs) < 2 or np.allclose(diffs, diffs[0]):
        # No spread across seeds (e.g. both conditions at 0%): t is undefined.
        t_stat, p = float("nan"), float("nan")
    else:
        res = stats.ttest_rel(a, b)
        t_stat, p = float(res.statistic), float(res.pvalue)
    return {"diff": mean, "ci95": [lo, hi], "ci90": [lo90, hi90], "t": t_stat,
            "df": len(diffs) - 1, "p": p, "n_seeds": int(len(diffs)),
            "per_seed_diff": diffs.tolist()}


def cohens_d(a, b) -> float:
    """Cohen's d for a - b with the pooled SD."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return float("nan")
    s1, s2 = np.var(a, ddof=1), np.var(b, ddof=1)
    sp = math.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    return float((np.mean(a) - np.mean(b)) / sp) if sp > 0 else float("nan")


def holm(pvals) -> list[float]:
    """Holm-adjusted p-values; NaN entries stay NaN and are left out of the family."""
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if ok.any():
        out[ok] = multipletests(p[ok], method="holm")[1]
    return out.tolist()


def kappa(conf, quadratic: bool = False) -> np.ndarray:
    """Cohen's kappa of K x K tables of counts (the last two axes; rows are the first
    rater), so one call scores a whole stack of bootstrap tables.

    quadratic=True weights a disagreement by the squared distance between the categories
    0..K-1, for an ordinal scale such as a 0 to 2 grade. The result is that of
    sklearn.metrics.cohen_kappa_score(y1, y2, labels=range(K), weights=None or
    "quadratic", sample_weight=w), which tests/test_analysis_cell.py checks. NaN where the
    two raters' marginals leave no expected disagreement.
    """
    conf = np.asarray(conf, dtype=float)
    k = conf.shape[-1]
    i = np.arange(k)
    w = ((i[:, None] - i[None, :]) ** 2).astype(float) if quadratic else (i[:, None] != i[None, :]).astype(float)
    n = conf.sum(axis=(-2, -1))
    expected = conf.sum(axis=-1)[..., :, None] * conf.sum(axis=-2)[..., None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        expected = expected / n[..., None, None]
        observed_d = (w * conf).sum(axis=(-2, -1))
        expected_d = (w * expected).sum(axis=(-2, -1))
        return np.where(expected_d > 0, 1.0 - observed_d / expected_d, np.nan)


def fisher_p(k1: int, n1: int, k2: int, n2: int) -> float:
    """Two-sided Fisher exact p for k1/n1 against k2/n2."""
    return float(stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]]).pvalue)


def mcnemar_exact_p(table) -> float:
    """Exact McNemar p of a paired 2x2 table [[both, first only], [second only, neither]]:
    the binomial test on the two discordant counts (statsmodels 0.14,
    statsmodels.stats.contingency_tables.mcnemar with exact=True)."""
    from statsmodels.stats.contingency_tables import mcnemar

    return float(mcnemar(table, exact=True).pvalue)


def welch_p(a, b) -> float:
    """Welch's t-test p, which treats every value as independent (descriptive only)."""
    with np.errstate(all="ignore"):
        return float(stats.ttest_ind(a, b, equal_var=False).pvalue)


def pearson_r(x, y) -> float:
    """Pearson r, or NaN when either side has no spread."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    # numpy 2.2 on macOS Accelerate raises spurious overflow warnings inside the BLAS
    # dot product here; the result was checked against a pure-Python Pearson.
    with np.errstate(all="ignore"):
        return float(stats.pearsonr(x, y).statistic)


def spearman_rho(x, y) -> float:
    with np.errstate(all="ignore"):
        return float(stats.spearmanr(x, y).statistic)


def odds_ratio(p1: float, p0: float) -> float:
    """Odds at rate p1 over odds at rate p0. NaN when a rate is 0 or 1."""
    if not (0 < p1 < 1 and 0 < p0 < 1):
        return float("nan")
    return (p1 / (1 - p1)) / (p0 / (1 - p0))


def percentile_ci(x) -> list[float]:
    """Percentile 95% interval of the finite values; NaN when there are none."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return [float("nan"), float("nan")]
    lo, hi = np.percentile(x, [2.5, 97.5])
    return [float(lo), float(hi)]


class ClusterBootstrap:
    """Bootstrap over prompt ids, shared by every pooled statistic of one cell.

    Each resample draws prompt ids with replacement and keeps every response to a
    drawn prompt, in every seed and condition. One weight matrix (resamples x prompts)
    is drawn once from a fixed seed and reused, so all intervals of a cell come from
    the same resamples and the output is deterministic.
    """

    def __init__(self, prompt_ids, resamples: int, seed: int):
        self.prompt_ids = sorted(prompt_ids)
        self.index = {p: i for i, p in enumerate(self.prompt_ids)}
        n = len(self.prompt_ids)
        rng = np.random.default_rng(seed)
        draws = rng.integers(0, n, size=(resamples, n))
        # How often each prompt is drawn in each resample.
        self.weights = np.zeros((resamples, n))
        np.add.at(self.weights, (np.repeat(np.arange(resamples), n), draws.ravel()), 1.0)

    def sums(self, frame: pd.DataFrame, column: str = "v") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per resample: the weighted count, sum and sum of squares of a column."""
        idx = frame["prompt_id"].map(self.index).to_numpy()
        v = frame[column].to_numpy(dtype=float)
        size = len(self.prompt_ids)
        cnt = np.bincount(idx, minlength=size).astype(float)
        s = np.bincount(idx, weights=v, minlength=size)
        q = np.bincount(idx, weights=v * v, minlength=size)
        # einsum rather than matmul: same result, but it does not go through BLAS,
        # which on macOS Accelerate raises spurious FP warnings and can change the
        # last bits between machines.
        out = np.einsum("bp,pk->bk", self.weights, np.stack([cnt, s, q], axis=1))
        return out[:, 0], out[:, 1], out[:, 2]

    def totals(self, prompt_ids, values) -> np.ndarray:
        """Per resample: the sum over rows of each column of values (rows x k), each row
        counted as often as the resample draws its prompt."""
        idx = pd.Series(list(prompt_ids)).map(self.index).to_numpy()
        values = np.asarray(values, dtype=float)
        per_prompt = np.zeros((len(self.prompt_ids), values.shape[1]))
        np.add.at(per_prompt, idx, values)
        return np.einsum("bp,pk->bk", self.weights, per_prompt)

    def means(self, frame: pd.DataFrame) -> np.ndarray:
        """Per resample: the pooled mean of column v."""
        n, s, _ = self.sums(frame)
        with np.errstate(invalid="ignore", divide="ignore"):
            return s / n

    def diff_and_d(self, fa: pd.DataFrame, fb: pd.DataFrame) -> dict:
        """Percentile 95% intervals of the pooled mean difference and Cohen's d (a - b)."""
        na, sa, qa = self.sums(fa)
        nb, sb, qb = self.sums(fb)
        with np.errstate(invalid="ignore", divide="ignore"):
            ma, mb = sa / na, sb / nb
            va = (qa - sa * sa / na) / (na - 1)
            vb = (qb - sb * sb / nb) / (nb - 1)
            sp = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
            diff = ma - mb
            d = diff / sp
        d = np.where(sp > 0, d, np.nan)
        return {"diff_ci95": percentile_ci(diff), "d_ci95": percentile_ci(d)}
