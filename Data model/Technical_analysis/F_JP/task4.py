import pandas as pd
import numpy as np

# -----------------------------------------------
# STEP 1: Load the data
# -----------------------------------------------

df = pd.read_csv('Task 3 and 4_Loan_Data.csv')

# We only need FICO score and default flag for this task
fico    = df['fico_score'].values
default = df['default'].values

n = len(fico)

# -----------------------------------------------
# STEP 2: What is bucketing and why do we need it?
# -----------------------------------------------

# FICO scores range from 408 to 850 — too many unique values
# for a categorical model. We need to group them into buckets
# (e.g. 400-449, 450-499, ...) such that each bucket captures
# a meaningfully different probability of default.
#
# The best bucketing minimises information loss — we want
# borrowers within the same bucket to behave similarly,
# and borrowers in different buckets to behave differently.
#
# We measure this using Log-Likelihood:
# For each bucket, we calculate how well a single default rate
# explains all the defaults in that bucket. The bucketing with
# the highest total log-likelihood is the best one.

# -----------------------------------------------
# STEP 3: Log-likelihood function for one bucket
# -----------------------------------------------

def bucket_log_likelihood(defaults, total):
    """
    Calculates the log-likelihood of a bucket.

    If p = default rate in the bucket, each borrower
    contributes log(p) if they defaulted, or log(1-p) if not.

    Parameters
    ----------
    defaults : int  — number of defaults in the bucket
    total    : int  — total borrowers in the bucket

    Returns
    -------
    float : log-likelihood (higher is better)
    """
    if total == 0:
        return 0

    p = defaults / total

    # Edge cases: if p=0 or p=1, log is undefined
    # Use a small clip to avoid log(0)
    p = np.clip(p, 1e-10, 1 - 1e-10)

    non_defaults = total - defaults
    return defaults * np.log(p) + non_defaults * np.log(1 - p)


# -----------------------------------------------
# STEP 4: Find the best bucket boundaries
# -----------------------------------------------

# We use dynamic programming to find the optimal boundaries.
# This tries every possible way to split the FICO range into
# n_buckets groups and picks the one with the highest
# total log-likelihood.

def find_best_buckets(fico, default, n_buckets):
    """
    Finds the FICO score boundaries that maximise
    total log-likelihood across all buckets.

    Parameters
    ----------
    fico     : array of FICO scores
    default  : array of 0/1 default flags
    n_buckets: number of buckets to create

    Returns
    -------
    boundaries : list of FICO score cutoff values
    """
    min_fico = fico.min()
    max_fico = fico.max()

    # Candidate boundary points — every unique FICO value
    candidates = np.unique(fico)

    best_ll     = -np.inf
    best_bounds = None

    # Try all combinations of (n_buckets - 1) boundaries
    # from the candidate values
    from itertools import combinations

    for bounds in combinations(candidates[1:], n_buckets - 1):
        boundaries = [min_fico] + list(bounds) + [max_fico + 1]
        total_ll   = 0

        for i in range(len(boundaries) - 1):
            low  = boundaries[i]
            high = boundaries[i + 1]
            mask = (fico >= low) & (fico < high)

            bucket_defaults = default[mask].sum()
            bucket_total    = mask.sum()
            total_ll       += bucket_log_likelihood(bucket_defaults, bucket_total)

        if total_ll > best_ll:
            best_ll     = total_ll
            best_bounds = boundaries

    return best_bounds, best_ll


# -----------------------------------------------
# STEP 5: Efficient version using sorted data
# -----------------------------------------------

# Trying every combination is slow for many candidates.
# Instead we use a smarter approach: sort by FICO and
# use dynamic programming over the sorted array.

def find_best_buckets_dp(fico, default, n_buckets):
    """
    Finds optimal FICO bucket boundaries using dynamic programming.
    Much faster than brute-force combinations.

    Returns
    -------
    boundaries : list of FICO cutoff values (length = n_buckets + 1)
    """
    # Sort by FICO score
    order  = np.argsort(fico)
    f_sort = fico[order]
    d_sort = default[order]

    # Get unique FICO values as possible split points
    unique_scores = np.unique(f_sort)
    m             = len(unique_scores)

    # Precompute cumulative defaults and counts for fast range queries
    # cum_defaults[i] = total defaults for borrowers with FICO <= unique_scores[i]
    cum_defaults = np.zeros(m)
    cum_counts   = np.zeros(m)

    for i, score in enumerate(unique_scores):
        mask            = f_sort <= score
        cum_defaults[i] = d_sort[mask].sum()
        cum_counts[i]   = mask.sum()

    def range_ll(i, j):
        """Log-likelihood for bucket covering unique_scores[i] to unique_scores[j]."""
        d = cum_defaults[j] - (cum_defaults[i-1] if i > 0 else 0)
        t = cum_counts[j]   - (cum_counts[i-1]   if i > 0 else 0)
        return bucket_log_likelihood(int(d), int(t))

    # dp[k][j] = best total log-likelihood using k buckets ending at index j
    # split[k][j] = where the last bucket started
    INF = -np.inf
    dp    = [[INF] * m for _ in range(n_buckets + 1)]
    split = [[0]   * m for _ in range(n_buckets + 1)]

    # Base: 1 bucket from 0 to j
    for j in range(m):
        dp[1][j]    = range_ll(0, j)
        split[1][j] = 0

    # Fill for 2..n_buckets
    for k in range(2, n_buckets + 1):
        for j in range(k - 1, m):
            best_val = INF
            best_i   = k - 1
            for i in range(k - 1, j + 1):
                val = dp[k-1][i-1] + range_ll(i, j) if i > 0 else INF
                if val > best_val:
                    best_val = val
                    best_i   = i
            dp[k][j]    = best_val
            split[k][j] = best_i

    # Backtrack to find boundaries
    boundaries = []
    j = m - 1
    for k in range(n_buckets, 0, -1):
        i = split[k][j]
        boundaries.append(unique_scores[i])
        j = i - 1

    boundaries.reverse()
    boundaries = [unique_scores[0]] + boundaries[1:] + [unique_scores[-1] + 1]

    return boundaries, dp[n_buckets][m - 1]


# -----------------------------------------------
# STEP 6: Build the bucket table and assign PD
# -----------------------------------------------

def build_bucket_table(fico, default, boundaries):
    """
    Given bucket boundaries, returns a table showing
    the default rate (PD) for each bucket.
    """
    rows = []
    for i in range(len(boundaries) - 1):
        low  = boundaries[i]
        high = boundaries[i + 1]
        mask = (fico >= low) & (fico < high)

        total    = mask.sum()
        defaults = default[mask].sum()
        pd_rate  = defaults / total if total > 0 else 0

        rows.append({
            'bucket'    : f"{low} - {high - 1}",
            'count'     : total,
            'defaults'  : defaults,
            'pd'        : round(pd_rate, 4)
        })

    return pd.DataFrame(rows)


# -----------------------------------------------
# STEP 7: Run for different numbers of buckets
# -----------------------------------------------

print("Finding optimal FICO buckets...\n")

for n in [5, 7, 10]:
    boundaries, ll = find_best_buckets_dp(fico, default, n)
    table          = build_bucket_table(fico, default, boundaries)

    print(f"===== {n} BUCKETS  (log-likelihood: {ll:.2f}) =====")
    print(table.to_string(index=False))
    print()


# -----------------------------------------------
# STEP 8: PD lookup function
# -----------------------------------------------

# Use 5 buckets as the final model (simple, interpretable)
FINAL_N_BUCKETS            = 5
final_boundaries, final_ll = find_best_buckets_dp(fico, default, FINAL_N_BUCKETS)
final_table                = build_bucket_table(fico, default, final_boundaries)

print(f"Final model: {FINAL_N_BUCKETS} buckets")
print(final_table.to_string(index=False))

def get_pd_from_fico(fico_score):
    """
    Returns the probability of default for a given FICO score
    based on the optimal bucketing.

    Parameters
    ----------
    fico_score : int — FICO score (300-850)

    Returns
    -------
    float : estimated probability of default
    """
    for i in range(len(final_boundaries) - 1):
        low  = final_boundaries[i]
        high = final_boundaries[i + 1]
        if low <= fico_score < high:
            return float(final_table.iloc[i]['pd'])
    return None

# Test
print("\nSample PD lookups:")
for score in [450, 550, 620, 700, 780]:
    print(f"  FICO {score}  ->  PD = {get_pd_from_fico(score)*100:.2f}%")
