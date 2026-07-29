"""Statistical helpers for benchmaxxing.

Thin wrappers over the established libraries (scipy, scikit-learn) with pure-numpy
implementations where a library is either absent or overkill. The two functions that
genuinely need statsmodels (Cochran-Mantel-Haenszel and mixed-effects logistic
regression) guard their import and raise a clear, actionable ImportError when the
optional ``stats`` extra is not installed.

Reuse-the-originals: nothing here re-implements what scipy or scikit-learn already do.
Only Cochran's Q, the percentile bootstrap, and the multiple-comparison corrections are
hand-rolled, because they are small and have no direct single-call equivalent we depend on.
"""

from __future__ import annotations

from typing import Callable, NamedTuple, Sequence

import numpy as np

__all__ = [
    "McNemarResult",
    "CochranQResult",
    "FisherResult",
    "CMHResult",
    "MultipleComparisonResult",
    "PermutationResult",
    "mcnemar",
    "cochran_q",
    "fisher_exact",
    "cochran_mantel_haenszel",
    "mixed_effects_logit",
    "bootstrap_ci",
    "paired_permutation_test",
    "phi_coefficient",
    "jaccard",
    "cohen_kappa",
    "multiple_comparison",
]


class McNemarResult(NamedTuple):
    """Result of the exact McNemar test. ``statistic`` is the smaller discordant count."""

    statistic: float
    pvalue: float


class CochranQResult(NamedTuple):
    """Result of Cochran's Q test. ``statistic`` is Q, distributed chi-square with df=k-1."""

    statistic: float
    pvalue: float
    df: int


class FisherResult(NamedTuple):
    """Result of Fisher's exact test on a 2x2 table."""

    oddsratio: float
    pvalue: float


class CMHResult(NamedTuple):
    """Result of the Cochran-Mantel-Haenszel test across strata."""

    statistic: float
    pvalue: float


class MultipleComparisonResult(NamedTuple):
    """Adjusted p-values and the boolean reject vector for a family of tests."""

    pvalues_adjusted: np.ndarray
    reject: np.ndarray


class PermutationResult(NamedTuple):
    """Result of a paired sign-flip permutation test. ``statistic`` is the mean difference."""

    statistic: float
    pvalue: float


def mcnemar(b: int, c: int) -> McNemarResult:
    """Exact McNemar test on the two discordant cell counts of a paired 2x2 table.

    ``b`` and ``c`` are the off-diagonal counts (one classifier right / the other wrong,
    and vice versa). Uses the exact binomial test (scipy.stats.binomtest) with p=0.5 over
    the ``b + c`` discordant pairs, which is the exact form of McNemar's test.

    Returns a :class:`McNemarResult` whose ``statistic`` is ``min(b, c)`` (the count fed
    to the binomial test) and whose ``pvalue`` is the two-sided exact p-value.
    """
    from scipy.stats import binomtest

    b = int(b)
    c = int(c)
    n = b + c
    if n == 0:
        # No discordant pairs: nothing to distinguish the two classifiers.
        return McNemarResult(statistic=0.0, pvalue=1.0)
    k = min(b, c)
    res = binomtest(k, n, 0.5, alternative="two-sided")
    return McNemarResult(statistic=float(k), pvalue=float(res.pvalue))


def cochran_q(binary_matrix: np.ndarray) -> CochranQResult:
    """Cochran's Q test for k related binary treatments over n subjects.

    ``binary_matrix`` is an ``(n_subjects, k_conditions)`` array of 0/1 values (one row per
    subject, one column per condition). Returns Q and its p-value from the chi-square
    distribution with ``k - 1`` degrees of freedom. Pure numpy for Q; scipy for the tail.

    For ``k == 2`` this reduces to the (uncorrected) McNemar chi-square ``(b - c)**2 / (b + c)``.
    """
    from scipy.stats import chi2

    x = np.asarray(binary_matrix, dtype=float)
    if x.ndim != 2:
        raise ValueError("binary_matrix must be 2-D (n_subjects x k_conditions)")
    n, k = x.shape
    if k < 2:
        raise ValueError("Cochran's Q needs at least 2 conditions (columns)")
    if not np.all(np.isin(x, (0.0, 1.0))):
        raise ValueError("binary_matrix must contain only 0/1 values")

    col_sums = x.sum(axis=0)          # C_j: successes per condition
    row_sums = x.sum(axis=1)          # R_i: successes per subject
    total = col_sums.sum()            # total successes

    denom = k * total - np.sum(row_sums**2)
    if denom == 0:
        # Every subject is constant across conditions: Q is undefined; treat as no effect.
        return CochranQResult(statistic=0.0, pvalue=1.0, df=k - 1)

    numer = (k - 1) * (k * np.sum(col_sums**2) - total**2)
    q = float(numer / denom)
    df = k - 1
    pvalue = float(chi2.sf(q, df))
    return CochranQResult(statistic=q, pvalue=pvalue, df=df)


def fisher_exact(table_2x2: Sequence[Sequence[float]]) -> FisherResult:
    """Fisher's exact test on a 2x2 contingency table.

    Wraps :func:`scipy.stats.fisher_exact` and returns the sample (unconditional)
    odds ratio and the two-sided p-value as a :class:`FisherResult`.
    """
    from scipy.stats import fisher_exact as _fisher_exact

    result = _fisher_exact(table_2x2)
    oddsratio, pvalue = result  # SignificanceResult unpacks to (statistic, pvalue)
    return FisherResult(oddsratio=float(oddsratio), pvalue=float(pvalue))


def cochran_mantel_haenszel(list_of_2x2: Sequence[np.ndarray]) -> CMHResult:
    """Cochran-Mantel-Haenszel test of a common odds ratio across strata.

    ``list_of_2x2`` is a sequence of 2x2 tables (one per stratum). Requires statsmodels
    (the ``stats`` extra); guards the import and raises a clear ImportError if it is
    missing. Returns the CMH statistic and p-value under the null of no association.
    """
    try:
        from statsmodels.stats.contingency_tables import StratifiedTable
    except ImportError as exc:  # pragma: no cover - exercised only when extra is absent
        raise ImportError(
            "cochran_mantel_haenszel requires statsmodels. "
            "Install the optional extra with: pip install 'benchmaxxing[stats]' "
            "(or: pip install statsmodels)."
        ) from exc

    # statsmodels expects tables stacked on the last axis: shape (2, 2, n_strata).
    tables = [np.asarray(t, dtype=float) for t in list_of_2x2]
    stacked = np.dstack(tables)
    st = StratifiedTable(stacked)
    test = st.test_null_odds(correction=True)
    return CMHResult(statistic=float(test.statistic), pvalue=float(test.pvalue))


def mixed_effects_logit(y, fixed_effects_df, groups):
    """Mixed-effects (random-intercept) logistic regression.

    Parameters
    ----------
    y : array-like of shape (n,)
        Binary 0/1 outcome for each observation.
    fixed_effects_df : pandas.DataFrame or array-like of shape (n, p)
        Fixed-effects design matrix. Include an explicit intercept column if you want one;
        no intercept is added automatically.
    groups : array-like of shape (n,)
        Grouping label per observation. A random intercept is fitted for each unique group.

    Returns
    -------
    The fitted statsmodels results object from
    ``statsmodels.genmod.bayes_mixed_glm.BinomialBayesMixedGLM`` (variational-Bayes fit),
    which exposes ``fe_mean`` (fixed-effect posterior means), ``summary()`` and friends.

    Requires statsmodels (the ``stats`` extra); guards the import and raises a clear
    ImportError if it is missing.
    """
    try:
        from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    except ImportError as exc:  # pragma: no cover - exercised only when extra is absent
        raise ImportError(
            "mixed_effects_logit requires statsmodels. "
            "Install the optional extra with: pip install 'benchmaxxing[stats]' "
            "(or: pip install statsmodels)."
        ) from exc

    from scipy import sparse

    y_arr = np.asarray(y, dtype=float)
    exog = np.asarray(getattr(fixed_effects_df, "values", fixed_effects_df), dtype=float)
    if exog.ndim == 1:
        exog = exog[:, None]
    groups_arr = np.asarray(groups)

    # One variance component: a random intercept per group. exog_vc is the indicator design
    # (n x n_groups); ident marks every column as belonging to the same variance component.
    uniq = np.unique(groups_arr)
    exog_vc = np.zeros((y_arr.shape[0], uniq.shape[0]), dtype=float)
    for j, g in enumerate(uniq):
        exog_vc[groups_arr == g, j] = 1.0
    ident = np.zeros(uniq.shape[0], dtype=int)

    model = BinomialBayesMixedGLM(y_arr, exog, sparse.csr_matrix(exog_vc), ident)
    return model.fit_vb()


def bootstrap_ci(
    data: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap confidence interval for a scalar statistic.

    Pure numpy. Resamples ``data`` with replacement ``n_boot`` times, applies ``statistic``
    to each resample, and reads the percentile interval. Returns ``(point, low, high)`` where
    ``point`` is ``statistic(data)`` on the observed sample.
    """
    x = np.asarray(data)
    if x.ndim != 1:
        raise ValueError("data must be 1-D")
    n = x.shape[0]
    if n == 0:
        raise ValueError("data must be non-empty")
    if not 0.0 < ci < 1.0:
        raise ValueError("ci must be in the open interval (0, 1)")

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_stats = np.array([statistic(x[row]) for row in idx], dtype=float)

    alpha = 1.0 - ci
    lo_pct = 100.0 * (alpha / 2.0)
    hi_pct = 100.0 * (1.0 - alpha / 2.0)
    low, high = np.percentile(boot_stats, [lo_pct, hi_pct])
    point = float(statistic(x))
    return point, float(low), float(high)


def cluster_bootstrap_ci(
    values: np.ndarray,
    clusters,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI that resamples whole clusters instead of single observations.

    ``clusters`` labels each value with the unit it shares correlated structure with -- the case
    a twin came from, say. Each resample draws the distinct clusters with replacement and pools
    every observation in the drawn clusters, so the interval answers "does this hold on new
    clusters" rather than treating correlated observations inside one cluster as independent, which
    understates the width. ``point`` is ``statistic`` over the observed values, unaffected by the
    clustering. When every cluster is a singleton this reduces exactly to :func:`bootstrap_ci` with
    the same seed.
    """
    x = np.asarray(values)
    labels = list(clusters)
    if x.ndim != 1:
        raise ValueError("values must be 1-D")
    if len(labels) != x.shape[0]:
        raise ValueError("clusters must label every value")
    if x.shape[0] == 0:
        raise ValueError("values must be non-empty")
    if not 0.0 < ci < 1.0:
        raise ValueError("ci must be in the open interval (0, 1)")

    # group indices by cluster in first-appearance order, so a run of singleton clusters draws the
    # same index stream as bootstrap_ci and keeps its pinned intervals reproducible
    grouped: dict = {}
    for i, label in enumerate(labels):
        grouped.setdefault(label, []).append(i)
    groups = [np.asarray(idx) for idx in grouped.values()]
    n = len(groups)

    rng = np.random.default_rng(seed)
    draw = rng.integers(0, n, size=(n_boot, n))
    boot_stats = np.array(
        [statistic(x[np.concatenate([groups[g] for g in row])]) for row in draw], dtype=float
    )

    alpha = 1.0 - ci
    low, high = np.percentile(boot_stats, [100.0 * (alpha / 2.0), 100.0 * (1.0 - alpha / 2.0)])
    point = float(statistic(x))
    return point, float(low), float(high)


# Above this many nonzero pairs the 2**m sign patterns are too many to enumerate, so switch to
# sampling. 2**18 = 262144 rows is still cheap; beyond it Monte Carlo is indistinguishable.
_PERMUTATION_EXACT_MAX = 18


def paired_permutation_test(
    differences: Sequence[float],
    n_permutations: int = 10000,
    seed: int = 0,
) -> PermutationResult:
    """Two-sided paired permutation test that the mean paired difference is zero.

    The randomization is the sign flip: under the null that each paired difference is symmetric
    about zero, flipping a difference's sign is equally likely, so the reference distribution is
    the mean difference over the sign assignments. Unlike a sign test it keeps every pair's
    magnitude, and unlike McNemar it does not collapse continuous per-case rates to a win/lose
    count -- so it is the paired-difference analogue of the ``adoption_delta`` bootstrap CI.

    Zero differences carry no sign information; with ``m`` nonzero pairs and ``m`` small enough
    all ``2**m`` sign vectors are enumerated for an exact p-value, otherwise ``n_permutations``
    sign vectors are sampled (seeded) and the p-value uses the ``(hits + 1) / (draws + 1)`` bound
    so it is never zero. Note the exact p is floored at ``2 / 2**m``: a handful of pairs cannot
    reach small p no matter how large the gap, which is an honest limit of the sample size.
    """
    diffs = np.asarray([d for d in differences if d is not None], dtype=float)
    observed = float(np.mean(diffs)) if diffs.size else 0.0
    nonzero = diffs[diffs != 0.0]
    m = int(nonzero.size)
    if m == 0:
        return PermutationResult(statistic=observed, pvalue=1.0)

    # |mean| >= |observed| collapses to |sign . nonzero| >= |sum(nonzero)| (the pair count is a
    # shared denominator that cancels); the tolerance absorbs float rounding on fractional rates.
    target = abs(float(nonzero.sum())) - 1e-12
    if m <= _PERMUTATION_EXACT_MAX:
        bits = (np.arange(2**m)[:, None] >> np.arange(m)) & 1
        signs = 1.0 - 2.0 * bits
        pvalue = float(np.mean(np.abs(signs @ nonzero) >= target))
        return PermutationResult(statistic=observed, pvalue=pvalue)

    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_permutations, m))
    hits = int(np.count_nonzero(np.abs(signs @ nonzero) >= target))
    return PermutationResult(statistic=observed, pvalue=(hits + 1) / (n_permutations + 1))

def phi_coefficient(y_true, y_pred) -> float:
    """Phi coefficient for two binary vectors, i.e. the Matthews correlation coefficient.

    Wraps :func:`sklearn.metrics.matthews_corrcoef`, which equals the phi coefficient for
    the binary case.
    """
    from sklearn.metrics import matthews_corrcoef

    return float(matthews_corrcoef(y_true, y_pred))


def jaccard(y_a, y_b) -> float:
    """Jaccard similarity between two binary vectors (intersection over union of the 1s).

    Wraps :func:`sklearn.metrics.jaccard_score` with the default binary averaging.
    """
    from sklearn.metrics import jaccard_score

    return float(jaccard_score(y_a, y_b))


def cohen_kappa(r1, r2) -> float:
    """Cohen's kappa inter-rater agreement between two label sequences.

    Wraps :func:`sklearn.metrics.cohen_kappa_score`.
    """
    from sklearn.metrics import cohen_kappa_score

    return float(cohen_kappa_score(r1, r2))


def multiple_comparison(
    pvalues: Sequence[float],
    method: str = "bh",
    alpha: float = 0.05,
) -> MultipleComparisonResult:
    """Adjust a family of p-values for multiple comparisons (pure numpy).

    Parameters
    ----------
    pvalues : sequence of float
        The raw per-test p-values.
    method : {"bh", "holm"}
        ``"bh"`` for Benjamini-Hochberg (FDR step-up); ``"holm"`` for Holm-Bonferroni
        (FWER step-down). No statsmodels dependency.
    alpha : float
        Significance threshold applied to the adjusted p-values.

    Returns
    -------
    MultipleComparisonResult
        ``pvalues_adjusted`` (same order as the input) and a boolean ``reject`` array.
    """
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1:
        raise ValueError("pvalues must be 1-D")
    if not np.all(np.isfinite(p)):
        # A nan sorts last and would poison the reversed cumulative minimum in the BH
        # step-up, silently destroying every rejection in the family. Refuse loudly: the
        # caller must drop or explain non-finite p-values (e.g. an undefined overlap)
        # before adjustment, so the family size m is an honest count of real tests.
        raise ValueError(
            "pvalues must be finite; drop or handle nan/inf p-values before adjustment"
        )
    m = p.shape[0]
    if m == 0:
        return MultipleComparisonResult(
            pvalues_adjusted=np.array([], dtype=float),
            reject=np.array([], dtype=bool),
        )

    order = np.argsort(p)
    ranked = p[order]
    key = method.lower()

    if key == "bh":
        # Step-up: multiply by m/rank, then enforce monotonicity from the largest p down.
        adj_sorted = ranked * m / np.arange(1, m + 1)
        adj_sorted = np.minimum.accumulate(adj_sorted[::-1])[::-1]
    elif key == "holm":
        # Step-down: multiply by (m - rank + 1), then enforce monotonicity upward.
        adj_sorted = ranked * (m - np.arange(m))
        adj_sorted = np.maximum.accumulate(adj_sorted)
    else:
        raise ValueError("method must be 'bh' or 'holm'")

    adj_sorted = np.clip(adj_sorted, 0.0, 1.0)
    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adj_sorted
    reject = adjusted <= alpha
    return MultipleComparisonResult(pvalues_adjusted=adjusted, reject=reject)


def required_pairs(
    p_discordant: float,
    effect: float,
    power: float = 0.8,
    alpha: float = 0.05,
) -> int:
    """Twin pairs needed for a paired flip-rate (McNemar) test.

    Closed-form normal approximation for the number of twin pairs required to detect a
    given difference in flip rates between the CLEAN and CONTAMINATED conditions with a
    two-sided McNemar / paired-proportion test.

    Parameters
    ----------
    p_discordant : float
        Expected proportion of *discordant* pairs, ``psi = p10 + p01`` in ``(0, 1]``: the
        fraction of twin pairs whose two conditions disagree. Concordant pairs (both right
        or both wrong) carry no information for the paired test, so a smaller discordant
        proportion needs more pairs.
    effect : float
        Flip-rate difference to detect, ``delta = p10 - p01`` (the gap between the two
        discordant cells). Only the magnitude matters; ``|effect| <= p_discordant`` since
        the two discordant probabilities are non-negative and sum to ``p_discordant``.
    power : float, default 0.8
        Target power ``1 - beta`` in ``(0, 1)``.
    alpha : float, default 0.05
        Two-sided significance level in ``(0, 1)``.

    Returns
    -------
    int
        Number of twin pairs, rounded up to the next whole pair.

    Notes
    -----
    Let ``psi = p_discordant`` and ``delta = effect``. Over ``n`` pairs the signed
    discordant count ``D = b - c`` has mean ``n * delta``; under the null it has variance
    ``n * psi`` and under the alternative variance ``n * (psi - delta**2)``. Equating the
    two-sided rejection threshold ``z_{1-alpha/2} * sqrt(n * psi)`` with the alternative
    distribution at power ``1 - beta`` gives::

        n = (z_{1-alpha/2} * sqrt(psi) + z_{1-beta} * sqrt(psi - delta**2))**2 / delta**2

    where ``z_q`` is the standard-normal quantile (``scipy.stats.norm.ppf``). This is the
    standard McNemar / paired-proportion sample-size approximation (e.g. Lachin 2011). It
    assumes the normal approximation to the binomial holds (not a tiny expected discordant
    count ``n * psi``) and that ``psi`` and the flip-rate split are specified correctly a
    priori. See :func:`achieved_power` for the inverse.
    """
    from scipy.stats import norm

    psi = float(p_discordant)
    delta = abs(float(effect))
    if not 0.0 < psi <= 1.0:
        raise ValueError("p_discordant must be in the interval (0, 1]")
    if delta == 0.0:
        raise ValueError("effect must be non-zero (no finite sample detects a null difference)")
    if delta > psi:
        raise ValueError("|effect| cannot exceed p_discordant (discordant cells are non-negative)")
    if not 0.0 < power < 1.0:
        raise ValueError("power must be in the open interval (0, 1)")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in the open interval (0, 1)")

    z_alpha = float(norm.ppf(1.0 - alpha / 2.0))
    z_beta = float(norm.ppf(power))
    var_alt = max(psi - delta * delta, 0.0)
    numer = z_alpha * np.sqrt(psi) + z_beta * np.sqrt(var_alt)
    n = (numer * numer) / (delta * delta)
    return int(np.ceil(n))


def achieved_power(
    n: int,
    p_discordant: float,
    effect: float,
    alpha: float = 0.05,
) -> float:
    """Power of the paired flip-rate (McNemar) test for a fixed number of pairs.

    Inverse companion of :func:`required_pairs` using the same normal approximation. For
    ``n`` twin pairs, discordant proportion ``psi = p_discordant`` and flip-rate difference
    ``delta = effect``::

        power = Phi((sqrt(n) * |delta| - z_{1-alpha/2} * sqrt(psi)) / sqrt(psi - delta**2))

    where ``Phi`` is the standard-normal CDF (``scipy.stats.norm.cdf``). Feeding the ``n``
    returned by :func:`required_pairs` back in recovers at least the requested power (the
    ceiling rounding only ever adds pairs). Returns a value in ``(0, 1)``.
    """
    from scipy.stats import norm

    psi = float(p_discordant)
    delta = abs(float(effect))
    if n <= 0:
        raise ValueError("n must be a positive number of pairs")
    if not 0.0 < psi <= 1.0:
        raise ValueError("p_discordant must be in the interval (0, 1]")
    if delta == 0.0:
        raise ValueError("effect must be non-zero")
    if delta > psi:
        raise ValueError("|effect| cannot exceed p_discordant (discordant cells are non-negative)")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in the open interval (0, 1)")

    z_alpha = float(norm.ppf(1.0 - alpha / 2.0))
    var_alt = max(psi - delta * delta, 0.0)
    if var_alt == 0.0:
        # Degenerate: every discordant pair points the same way; the threshold is always cleared.
        return 1.0
    z = (np.sqrt(n) * delta - z_alpha * np.sqrt(psi)) / np.sqrt(var_alt)
    return float(norm.cdf(z))
