import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import date

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA
#    Henry Hub monthly spot prices ($/MMBtu), Oct 2020 – Sep 2024  (source: EIA)
#    Each row: [year, month, price]
# ─────────────────────────────────────────────────────────────────────────────
RAW = np.array([
    [2020,10,2.39],[2020,11,2.61],[2020,12,2.58],
    [2021, 1,2.71],[2021, 2,5.35],[2021, 3,2.62],[2021, 4,2.66],
    [2021, 5,2.91],[2021, 6,3.26],[2021, 7,3.84],[2021, 8,4.07],
    [2021, 9,5.16],[2021,10,5.51],[2021,11,5.05],[2021,12,3.76],
    [2022, 1,4.38],[2022, 2,4.69],[2022, 3,4.90],[2022, 4,6.60],
    [2022, 5,8.14],[2022, 6,7.70],[2022, 7,7.28],[2022, 8,8.81],
    [2022, 9,7.88],[2022,10,5.66],[2022,11,5.45],[2022,12,5.53],
    [2023, 1,3.27],[2023, 2,2.38],[2023, 3,2.31],[2023, 4,2.16],
    [2023, 5,2.15],[2023, 6,2.18],[2023, 7,2.55],[2023, 8,2.58],
    [2023, 9,2.64],[2023,10,2.98],[2023,11,2.71],[2023,12,2.52],
    [2024, 1,3.18],[2024, 2,1.72],[2024, 3,1.49],[2024, 4,1.60],
    [2024, 5,2.12],[2024, 6,2.54],[2024, 7,2.07],[2024, 8,1.99],
    [2024, 9,2.28],
])

years, months, prices = RAW[:, 0].astype(int), RAW[:, 1].astype(int), RAW[:, 2]
N = len(prices)  # 48 data points

# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING
#
#    The model captures two things:
#      (a) Long-run TREND  — a quadratic polynomial in time
#      (b) SEASONALITY     — sinusoidal (Fourier) terms with a 12-month period
#
#    Feature vector for each observation:
#      x = [1,  t,  t²,  sin(2πm/12),  cos(2πm/12),  sin(4πm/12),  cos(4πm/12)]
#
#    where:
#      t = normalised month index  (0 at Oct-2020, 1 at Sep-2024)
#      m = calendar month, 0-indexed (Jan=0, Feb=1, ..., Dec=11)
#
#    Why Fourier terms?
#      A single sine+cosine pair (frequency 1/year) can represent any sinusoid
#      of period 12 months.  Adding a second harmonic (frequency 2/year) allows
#      the seasonal curve to be asymmetric — e.g. a sharp winter spike but a
#      broad summer dip.
# ─────────────────────────────────────────────────────────────────────────────
def month_index(year, month):
    """Months elapsed since Oct 2020 (the first data point)."""
    return (year - 2020) * 12 + month - 10

def build_features(year, month):
    """Return the 7-element feature vector for a given year/month."""
    t = month_index(year, month) / (N - 1)      # normalised time [0, 1]
    theta = (month - 1) * 2 * np.pi / 12        # angle for calendar month
    return np.array([
        1,
        t,
        t ** 2,
        np.sin(theta),
        np.cos(theta),
        np.sin(2 * theta),
        np.cos(2 * theta),
    ])

# Build design matrix X (shape 48×7) and target vector y (shape 48,)
X = np.vstack([build_features(yr, mo) for yr, mo in zip(years, months)])
y = prices

# ─────────────────────────────────────────────────────────────────────────────
# 3. ORDINARY LEAST SQUARES (OLS)
#
#    We want coefficients β that minimise the sum of squared residuals:
#      min_β  ||y - X β||²
#
#    The closed-form solution (normal equations) is:
#      β = (XᵀX)⁻¹ Xᵀy
#
#    numpy.linalg.lstsq is numerically preferred over explicit inversion.
# ─────────────────────────────────────────────────────────────────────────────
coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

# Goodness of fit
y_fit = X @ coeffs
ss_res = np.sum((y - y_fit) ** 2)
ss_tot = np.sum((y - y.mean()) ** 2)
r_squared = 1 - ss_res / ss_tot
print(f"R² = {r_squared:.4f}  (1.0 = perfect fit, 0 = no better than mean)")

# ─────────────────────────────────────────────────────────────────────────────
# 4. PREDICTION FUNCTION
#
#    Given any date, build its feature vector and take the dot product with β.
#    Dates before the training window or more than 12 months beyond it
#    trigger a warning; the model can still return a number but accuracy drops.
# ─────────────────────────────────────────────────────────────────────────────
def estimate_price(input_date):
    """
    Estimate the Henry Hub natural gas spot price for a given date.

    Parameters
    ----------
    input_date : datetime.date  (or any object with .year and .month)

    Returns
    -------
    float  — estimated price in $/MMBtu
    """
    year, month = input_date.year, input_date.month
    idx = month_index(year, month)

    if idx < 0:
        print(f"  ⚠  {input_date} is before the training window (Oct 2020).")
    elif idx > N - 1 + 12:
        print(f"  ⚠  {input_date} is more than 12 months beyond the training data — extrapolation is unreliable.")

    feat = build_features(year, month)
    price = float(feat @ coeffs)
    return max(0.30, round(price, 2))   # floor at $0.30 (physical minimum)

# ── Quick demo ────────────────────────────────────────────────────────────────
test_dates = [
    date(2021,  8,  1),   # within training: summer 2021
    date(2022,  8,  1),   # within training: peak crisis
    date(2024,  3,  1),   # within training: 2024 record low
    date(2025,  1,  1),   # 3 months into forecast window
    date(2026,  6,  1),   # 10 months into forecast window
]
print("\n── Price estimates ──────────────────────────────────────────")
for d in test_dates:
    print(f"  {d}  →  ${estimate_price(d):.2f} / MMBtu")

# ─────────────────────────────────────────────────────────────────────────────
# 5. VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(12, 9))
fig.patch.set_facecolor('#0f172a')
for ax in axes:
    ax.set_facecolor('#1e293b')
    for spine in ax.spines.values():
        spine.set_edgecolor('#334155')
    ax.tick_params(colors='#94a3b8')
    ax.xaxis.label.set_color('#94a3b8')
    ax.yaxis.label.set_color('#94a3b8')
    ax.title.set_color('#e2e8f0')
    ax.grid(color='#1e3a5f', linestyle='--', linewidth=0.7)

# ── Panel 1: time-series ──────────────────────────────────────────────────────
ax1 = axes[0]
train_dates = [date(int(y), int(m), 1) for y, m in zip(years, months)]

# Forecast range: Oct 2024 – Sep 2025
fc_months = [(2024 + (9 + i) // 12, (9 + i) % 12 + 1) for i in range(12)]
fc_dates   = [date(yr, mo, 1) for yr, mo in fc_months]
fc_prices  = [estimate_price(date(yr, mo, 1)) for yr, mo in fc_months]

ax1.plot(train_dates, prices, 'o-', color='#60a5fa', lw=2,  ms=4,  label='Actual')
ax1.plot(train_dates, y_fit,  '--',  color='#34d399', lw=1.5,       label='Model fit')
ax1.plot(fc_dates,    fc_prices, 's-', color='#f97316', lw=2, ms=5, label='Forecast')
ax1.axvline(date(2024, 10, 1), color='#f59e0b', linestyle=':', lw=1.5, label='Forecast start')

ax1.set_title('Henry Hub Natural Gas — Actual, Model Fit & 12-Month Forecast', fontsize=13)
ax1.set_ylabel('Price ($/MMBtu)')
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
plt.setp(ax1.get_xticklabels(), rotation=35, ha='right', fontsize=8)
ax1.legend(facecolor='#0f172a', edgecolor='#334155', labelcolor='#e2e8f0', fontsize=9)
ax1.text(0.01, 0.97, f'R² = {r_squared:.3f}', transform=ax1.transAxes,
         color='#34d399', fontsize=9, va='top')

# ── Panel 2: seasonal component ───────────────────────────────────────────────
ax2 = axes[1]
mo_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
seasonal = np.array([
    coeffs[3]*np.sin(i*2*np.pi/12) + coeffs[4]*np.cos(i*2*np.pi/12) +
    coeffs[5]*np.sin(i*4*np.pi/12) + coeffs[6]*np.cos(i*4*np.pi/12)
    for i in range(12)
])
colors = ['#f97316' if s >= 0 else '#38bdf8' for s in seasonal]
ax2.bar(mo_names, seasonal, color=colors, edgecolor='#0f172a', linewidth=0.5)
ax2.axhline(0, color='#334155', lw=1.5)
ax2.set_title('Seasonal Component — Monthly Deviation from Trend ($/MMBtu)', fontsize=13)
ax2.set_ylabel('Seasonal effect ($/MMBtu)')

plt.tight_layout(pad=2.0)
plt.savefig('natgas_estimator.png', dpi=150, bbox_inches='tight')
plt.show()