import math
import numpy as np
import pandas as pd

# Load the loan data
df = pd.read_csv('Task 3 and 4_Loan_Data.csv')
print(df.head())
print(df[['fico_score', 'default']].describe())

# Aggregate to unique-FICO-score level (required for DP)
g = df.groupby('fico_score')['default'].agg(['count', 'sum']).reset_index()
g.columns = ['score', 'n', 'k']
g = g.sort_values('score').reset_index(drop=True)
scores, n_arr, k_arr = g['score'].values, g['n'].values, g['k'].values

# ── Log-Likelihood Dynamic Programming ─────────────────────────
def loglik_dp(scores, n_arr, k_arr, n_buckets):
    """
    Maximise: L = Σ [ k_b*log(p_b) + (n_b - k_b)*log(1 - p_b) ]
    p_b = k_b / n_b  (MLE default probability in bucket b)
    Laplace smoothing avoids log(0) for pure buckets.
    """
    S = len(scores)

    # Prefix sums for O(1) segment queries
    cum_n = np.zeros(S + 1)
    cum_k = np.zeros(S + 1)
    for i in range(S):
        cum_n[i+1] = cum_n[i] + n_arr[i]
        cum_k[i+1] = cum_k[i] + k_arr[i]

    def seg_loglik(i, j):
        n = cum_n[j] - cum_n[i]
        k = cum_k[j] - cum_k[i]
        if n == 0: return 0.0
        p = (k + 1) / (n + 2)
        return k * math.log(p) + (n - k) * math.log(1 - p)

    # DP tables
    NEG_INF = -float('inf')
    dp    = np.full((S +1, n_buckets +1), NEG_INF)
    split = np.zeros((S +1, n_buckets +1), dtype=int)
    dp[0][0] = 0.0

    for b in range(1, n_buckets +1):
        for j in range(b, S +1):
            for i in range(b -1, j):
                val = dp[i][b-1] + seg_loglik(i, j)
                if val > dp[j][b]:
                    dp[j][b]    = val
                    split[j][b] = i

    # Back-trace to recover boundary indices
    boundaries_idx = []
    j, b = S, n_buckets
    while b > 0:
        i = split[j][b]
        boundaries_idx.append(i)
        j, b = i, b -1
    boundaries_idx.reverse()

    # Convert score indices → actual FICO values
    boundaries = [int(scores[0])]
    for idx in boundaries_idx[1:]:
        boundaries.append(int(scores[idx]))
    boundaries[-1] = int(scores[-1])
    return boundaries

# ── Build Rating Map ────────────────────────────────────────────
def build_rating_map(df, boundaries):
    n_buckets = len(boundaries) - 1
    labels    = list(range(n_buckets, 0, -1))
    df = df.copy()
    df['rating'] = pd.cut(
        df['fico_score'], bins=boundaries,
        labels=labels, include_lowest=True, right=True,
    ).astype(int)
    return df

# ── Run ─────────────────────────────────────────────────────────
N_BUCKETS = 5
boundaries = loglik_dp(scores, n_arr, k_arr, N_BUCKETS)
print('Optimal boundaries:', boundaries)

df_rated = build_rating_map(df, boundaries)
summary = (
    df_rated.groupby('rating')['default']
             .agg(n=('count'), k=('sum'))
             .assign(pd_est= lambda x: x['k'] / x['n']))
print(summary)


