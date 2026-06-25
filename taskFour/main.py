"""
FICO score bucketing (quantization) for the PD model.

Charlie's requirement: given an arbitrary number of ratings (buckets), find the
FICO score boundaries that "best summarise" the data, then expose a map that
takes a raw FICO score and returns a rating, where a LOWER rating number means
a BETTER credit score (rating 1 = best bucket).

Two optimisation criteria are implemented, both solved exactly with dynamic
programming so the approach generalises to any number of buckets and any new
dataset:

1. Mean-squared-error (MSE) quantization
   Treat every record's FICO score as a point we are approximating by the
   mean of whichever bucket it falls into. Minimise the total squared error.
   This is exactly 1-D k-means / "Jenks natural breaks" and only looks at the
   distribution of scores themselves (it ignores default labels).

2. Log-likelihood (LL) quantization
   Treat each bucket as having its own probability of default p_i = k_i / n_i
   (k_i = defaults in bucket, n_i = records in bucket) and maximise the
   log-likelihood of observing the actual defaults under a Bernoulli model
   with one default-probability per bucket. This is the more sophisticated
   approach because it directly optimises how well the buckets separate risk,
   not just how well they summarise the score distribution.

Both are solved with the same style of DP:
  - Sort unique FICO scores.
  - Precompute prefix sums so the "cost" (or "gain") of merging any
    contiguous range of scores into one bucket is O(1) to evaluate.
  - DP over (number of buckets used so far, last boundary index), which is
    the standard "partition a sequence into K contiguous segments to
    optimise a separable cost" dynamic program.

Complexity: O(N^2 * K) where N = number of distinct FICO scores in the
training data and K = number of buckets. For FICO scores N is at most ~450
(300-850), so this is fast even for K up to a few dozen.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Sequence


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _aggregate_by_score(fico_scores: np.ndarray, defaults: np.ndarray):
    """Collapse raw records into one row per distinct FICO score.

    Returns sorted unique scores, the count of records at each score (n_i),
    and the count of defaults at each score (k_i). Aggregating first is what
    keeps the DP fast - we never need more states than there are distinct
    scores, regardless of how many millions of rows the dataset has.
    """
    df = pd.DataFrame({"score": fico_scores, "default": defaults})
    grouped = df.groupby("score")["default"].agg(n="count", k="sum")
    grouped = grouped.sort_index()
    scores = grouped.index.to_numpy()
    n = grouped["n"].to_numpy(dtype=float)
    k = grouped["k"].to_numpy(dtype=float)
    return scores, n, k


def _segment_boundaries_from_cuts(scores: np.ndarray, cut_indices: Sequence[int]):
    """Turn DP cut points (indices into `scores`) into human-readable
    (lower, upper) score boundaries for each bucket, lowest bucket first.
    cut_indices is the sorted list of internal cut points, e.g. for 3 buckets
    over 5 unique scores, cut_indices could be [2, 4] meaning bucket
    boundaries are [0:2], [2:4], [4:5] in index space.
    """
    bounds = []
    prev = 0
    edges = list(cut_indices) + [len(scores)]
    for edge in edges:
        lo = scores[prev]
        hi = scores[edge - 1]
        bounds.append((float(lo), float(hi)))
        prev = edge
    return bounds


# ---------------------------------------------------------------------------
# 1) MSE-optimal quantization
# ---------------------------------------------------------------------------

def mse_optimal_boundaries(fico_scores, defaults, n_buckets: int):
    """Find bucket boundaries minimising total squared error of approximating
    each score by its bucket mean (weighted by how many borrowers have that
    score). Defaults are accepted only so the function signature matches the
    log-likelihood version - they are not used in the MSE objective.
    """
    scores, n, _k = _aggregate_by_score(np.asarray(fico_scores), np.asarray(defaults))
    N = len(scores)
    K = n_buckets
    if K >= N:
        # Degenerate: more buckets than distinct scores - one score per bucket.
        return [(float(s), float(s)) for s in scores]

    # Prefix sums for O(1) cost-of-a-segment lookups.
    S0 = np.concatenate([[0.0], np.cumsum(n)])               # weight (count)
    S1 = np.concatenate([[0.0], np.cumsum(n * scores)])       # weighted sum
    S2 = np.concatenate([[0.0], np.cumsum(n * scores ** 2)])  # weighted sum of squares

    def seg_cost(l: int, r: int) -> float:
        """Weighted SSE of grouping scores[l:r] into a single bucket."""
        w = S0[r] - S0[l]
        if w == 0:
            return 0.0
        s1 = S1[r] - S1[l]
        s2 = S2[r] - S2[l]
        # sum w_i*(x_i - mean)^2 = sum w_i*x_i^2 - (sum w_i*x_i)^2 / sum w_i
        return s2 - (s1 ** 2) / w

    INF = float("inf")
    # dp[k][i] = min cost of splitting scores[0:i] into k buckets
    dp = [[INF] * (N + 1) for _ in range(K + 1)]
    back = [[-1] * (N + 1) for _ in range(K + 1)]
    dp[0][0] = 0.0

    for k in range(1, K + 1):
        for i in range(k, N + 1):
            best_cost, best_j = INF, -1
            # j is the previous cut point; need at least one point per bucket
            for j in range(k - 1, i):
                if dp[k - 1][j] == INF:
                    continue
                cost = dp[k - 1][j] + seg_cost(j, i)
                if cost < best_cost:
                    best_cost, best_j = cost, j
            dp[k][i] = best_cost
            back[k][i] = best_j

    # Reconstruct cut points
    cuts = []
    i, k = N, K
    while k > 0:
        j = back[k][i]
        cuts.append(j)
        i, k = j, k - 1
    cuts = sorted(cuts[:-1])  # drop the implicit cut at 0

    return _segment_boundaries_from_cuts(scores, cuts)


# ---------------------------------------------------------------------------
# 2) Log-likelihood-optimal quantization
# ---------------------------------------------------------------------------

def _bucket_log_likelihood(n: float, k: float) -> float:
    """log-likelihood of k defaults out of n records under a Bernoulli(p)
    model with p = k/n (the maximum-likelihood estimate for that bucket).
    """
    if n == 0:
        return 0.0
    if k == 0 or k == n:
        # p is exactly 0 or 1: every record is explained perfectly, LL = 0
        return 0.0
    p = k / n
    return k * np.log(p) + (n - k) * np.log(1 - p)


def loglikelihood_optimal_boundaries(fico_scores, defaults, n_buckets: int):
    """Find bucket boundaries maximising the total log-likelihood of the
    observed defaults, where each bucket gets its own default probability
    p_i = k_i/n_i. This is the dynamic-programming approach referenced in
    the task: split the FICO range into contiguous segments and pick the
    K-1 cut points that jointly maximise sum_i LL(n_i, k_i).
    """
    scores, n, k = _aggregate_by_score(np.asarray(fico_scores), np.asarray(defaults))
    N = len(scores)
    K = n_buckets
    if K >= N:
        return [(float(s), float(s)) for s in scores]

    cumN = np.concatenate([[0.0], np.cumsum(n)])
    cumK = np.concatenate([[0.0], np.cumsum(k)])

    def seg_ll(l: int, r: int) -> float:
        return _bucket_log_likelihood(cumN[r] - cumN[l], cumK[r] - cumK[l])

    NEG_INF = float("-inf")
    dp = [[NEG_INF] * (N + 1) for _ in range(K + 1)]
    back = [[-1] * (N + 1) for _ in range(K + 1)]
    dp[0][0] = 0.0

    for kk in range(1, K + 1):
        for i in range(kk, N + 1):
            best_ll, best_j = NEG_INF, -1
            for j in range(kk - 1, i):
                if dp[kk - 1][j] == NEG_INF:
                    continue
                ll = dp[kk - 1][j] + seg_ll(j, i)
                if ll > best_ll:
                    best_ll, best_j = ll, j
            dp[kk][i] = best_ll
            back[kk][i] = best_j

    cuts = []
    i, kk = N, K
    while kk > 0:
        j = back[kk][i]
        cuts.append(j)
        i, kk = j, kk - 1
    cuts = sorted(cuts[:-1])

    return _segment_boundaries_from_cuts(scores, cuts)


# ---------------------------------------------------------------------------
# Rating map: turn boundaries into a usable FICO -> rating function
# ---------------------------------------------------------------------------

@dataclass
class FicoRatingMap:
    """Maps a raw FICO score to a rating, where rating 1 is the BEST bucket
    (highest scores) and rating n_buckets is the WORST bucket (lowest
    scores) - i.e. lower rating number = better credit score, as requested.
    """
    boundaries: list  # list of (lo, hi) tuples, ascending score order

    def __post_init__(self):
        # sort buckets ascending by score, then rating = position counted
        # from the top (best) bucket
        self._sorted = sorted(self.boundaries, key=lambda b: b[0])
        self._n = len(self._sorted)

    def rate(self, fico_score: float) -> int:
        for idx, (lo, hi) in enumerate(self._sorted):
            # last bucket's upper edge is inclusive of the max score
            if (lo <= fico_score <= hi) or (idx == self._n - 1 and fico_score > hi):
                bucket_from_bottom = idx
                return self._n - bucket_from_bottom
        # score below the lowest boundary -> worst rating
        return self._n

    def __call__(self, fico_score: float) -> int:
        return self.rate(fico_score)

    def as_table(self) -> pd.DataFrame:
        rows = []
        for idx, (lo, hi) in enumerate(self._sorted):
            rating = self._n - idx
            rows.append({"rating": rating, "fico_low": lo, "fico_high": hi})
        return pd.DataFrame(rows).sort_values("rating").reset_index(drop=True)


def build_rating_map(fico_scores, defaults, n_buckets: int, method: str = "loglikelihood") -> FicoRatingMap:
    """Convenience entry point Charlie's pipeline can call directly:

        rating_map = build_rating_map(df['fico_score'], df['default'], n_buckets=10)
        df['rating'] = df['fico_score'].apply(rating_map)

    method: 'loglikelihood' (default, risk-aware) or 'mse' (score-distribution-only).
    """
    if method == "mse":
        boundaries = mse_optimal_boundaries(fico_scores, defaults, n_buckets)
    elif method == "loglikelihood":
        boundaries = loglikelihood_optimal_boundaries(fico_scores, defaults, n_buckets)
    else:
        raise ValueError("method must be 'mse' or 'loglikelihood'")
    return FicoRatingMap(boundaries)


# ---------------------------------------------------------------------------
# Demo / sanity check against the supplied dataset
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = pd.read_csv("C:/Users/thoma/OneDrive/Documents/GitHub/Quant-Researcher-simulation/taskThree/Task 3 and 4_Loan_Data.csv")
    fico = df["fico_score"].to_numpy()
    default = df["default"].to_numpy()

    for n_buckets in (5, 10):
        print(f"\n=== n_buckets = {n_buckets} ===")

        mse_map = build_rating_map(fico, default, n_buckets, method="mse")
        print("-- MSE-optimal buckets --")
        print(mse_map.as_table().to_string(index=False))

        ll_map = build_rating_map(fico, default, n_buckets, method="loglikelihood")
        print("-- Log-likelihood-optimal buckets --")
        table = ll_map.as_table()
        # attach empirical default rate per bucket for sanity-checking
        rates = []
        for lo, hi in zip(table.fico_low, table.fico_high):
            mask = (fico >= lo) & (fico <= hi)
            rates.append(default[mask].mean() if mask.sum() else float("nan"))
        table["empirical_default_rate"] = rates
        print(table.to_string(index=False))