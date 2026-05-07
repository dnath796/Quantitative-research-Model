"""
WTI Crude Oil Return Drivers — Full Pipeline
MFIN8852 Financial Econometrics Final Project

Pipeline:  EDA  →  ADF stationarity  →  OLS  →  Diagnostics  →  HAC  →  Excel output
Sample: December 1990 – December 2025
Dependent variable: wti_chg_pct (monthly % change in WTI price)

Outputs:
  residual_diagnostics.png
  coefficient_plot.png
  wti_regression_results.xlsx   ← full workbook with OLS and HAC side-by-side
"""

import datetime
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from openpyxl import Workbook
from openpyxl.styles import (Font, PatternFill, Alignment, Border, Side,
                              numbers as xl_numbers)
from openpyxl.utils import get_column_letter
from openpyxl import load_workbook

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import (het_breuschpagan,
                                          acorr_breusch_godfrey,
                                          linear_reset)
from statsmodels.stats.stattools import durbin_watson
from scipy import stats
from scipy.stats import f as f_dist


# ═══════════════════════════════════════════════════════════════════════════════
# 0.  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

EXCEL_PATH   = 'master_datasheet.xlsx'
SHEET_NAME   = 'Combined Monthly'
SAMPLE_START = '2015-12'
SAMPLE_END   = '2025-12'
OUTPUT_XLSX  = 'wti_regression_results.xlsx'

TARGET_COLS = [
    'wti_chg_pct',
    'dxy_chg_pct',
    'sp500_chg_pct',
    'gld_chg_pct',
    'copper_chg_pct',
    'ng_chg_pct',
    'us_crudeProd_chg_pct',
    'indpro_chg_pct',
    'crude_stk_excl_spr',
    'spr_inv_chg',
    'tcmsy10_rate_1st_diff',
    'vix_close',
    'refin_util_rate',
    'igrea_level',
    'gpr_idx_level',
]

LABELS = {
    'wti_chg_pct':           'WTI Return (%)',
    'dxy_chg_pct':           'DXY Return (%)',
    'sp500_chg_pct':         'S&P 500 Return (%)',
    'gld_chg_pct':           'Gold Return (%)',
    'copper_chg_pct':        'Copper Return (%)',
    'ng_chg_pct':            'Natural Gas Return (%)',
    'us_crudeProd_chg_pct':  'US Crude Prod. Chg (%)',
    'indpro_chg_pct':        'INDPRO Chg (%)',
    'crude_stk_excl_spr':    'Commercial Crude Stocks (Level)',
    'spr_inv_chg':           'SPR Inventory Chg',
    'tcmsy10_rate_1st_diff': '10-Yr Treasury Delta (pp)',
    'vix_close':             'VIX (Level)',
    'refin_util_rate':       'Refinery Utilization Rate',
    'igrea_level':           'Kilian IGREA (Level)',
    'gpr_idx_level':         'GPR Index (Level)',
}

# Regressors EXCLUDING refinery utilization (VIF=13) and target
REGRESSORS = [c for c in TARGET_COLS
              if c not in ('wti_chg_pct', 'refin_util_rate')]

# ── Style helpers ─────────────────────────────────────────────────────────────

def _hdr(bold=True, color='FFFFFF', bg='2F5496', size=10, italic=False):
    return {'font': Font(bold=bold, color=color, size=size, italic=italic,
                         name='Arial'),
            'fill': PatternFill('solid', start_color=bg),
            'align': Alignment(horizontal='center', vertical='center',
                               wrap_text=True)}

def _cell(bold=False, color='000000', bg=None, align='center', size=10,
          italic=False, wrap=False):
    fill = PatternFill('solid', start_color=bg) if bg else PatternFill()
    return {'font': Font(bold=bold, color=color, size=size, italic=italic,
                         name='Arial'),
            'fill': fill,
            'align': Alignment(horizontal=align, vertical='center',
                               wrap_text=wrap)}

THIN = Border(
    left=Side(style='thin'),  right=Side(style='thin'),
    top=Side(style='thin'),   bottom=Side(style='thin'),
)
MEDIUM_BOTTOM = Border(bottom=Side(style='medium'))

def apply(ws, row, col, value, style: dict, num_fmt=None):
    c = ws.cell(row=row, column=col, value=value)
    if 'font'  in style: c.font      = style['font']
    if 'fill'  in style: c.fill      = style['fill']
    if 'align' in style: c.alignment = style['align']
    c.border = THIN
    if num_fmt: c.number_format = num_fmt
    return c

def merge_hdr(ws, r1, c1, r2, c2, value, style):
    ws.merge_cells(start_row=r1, start_column=c1,
                   end_row=r2,   end_column=c2)
    c = ws.cell(row=r1, column=c1, value=value)
    if 'font'  in style: c.font      = style['font']
    if 'fill'  in style: c.fill      = style['fill']
    if 'align' in style: c.alignment = style['align']
    c.border = THIN

SIG_STARS = lambda p: '***' if p < 0.01 else ('**' if p < 0.05
                         else ('*' if p < 0.1 else ''))


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════

wb_in = load_workbook(EXCEL_PATH, read_only=True, data_only=True)
ws_in = wb_in[SHEET_NAME]
rows  = list(ws_in.iter_rows(values_only=True))
wb_in.close()

field_row = rows[2]
col_idx   = {f: i for i, f in enumerate(field_row) if f is not None}

records = []
for r in rows[3:]:
    if not isinstance(r[0], datetime.datetime):
        continue
    rec = {'Date': r[0]}
    for col in TARGET_COLS:
        rec[col] = r[col_idx[col]]
    records.append(rec)

df = pd.DataFrame(records).set_index('Date')
df.index = pd.to_datetime(df.index)
df = df.apply(pd.to_numeric, errors='coerce').dropna()
df = df.loc[SAMPLE_START:SAMPLE_END]
T  = len(df)

print(f'Sample: {df.index[0]:%Y-%m} to {df.index[-1]:%Y-%m}  |  T = {T}')

y     = df['wti_chg_pct']
X_raw = df[REGRESSORS]
X     = sm.add_constant(X_raw)


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  EDA — DESCRIPTIVE STATISTICS  (printed; also goes to Excel)
# ═══════════════════════════════════════════════════════════════════════════════

desc = df[TARGET_COLS].describe().T
desc['skew']     = df[TARGET_COLS].skew()
desc['kurtosis'] = df[TARGET_COLS].kurtosis()
print('\n── Descriptive Statistics ──')
print(desc[['mean', 'std', 'min', 'max', 'skew', 'kurtosis']].to_string())


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  STATIONARITY — ADF + KPSS
# ═══════════════════════════════════════════════════════════════════════════════

def run_adf(series, name):
    r = adfuller(series.dropna(), autolag='AIC', regression='c')
    return {'variable': name, 'adf_stat': r[0], 'adf_p': r[1],
            'adf_lags': r[2], 'adf_sig': SIG_STARS(r[1]),
            'adf_conclusion': 'Stationary' if r[1] < 0.05 else 'Unit Root'}

def run_kpss(series, name):
    stat, p, lags, crit = kpss(series.dropna(), regression='c',
                                nlags='auto')
    # KPSS: H0 = stationary; reject H0 means non-stationary
    conclusion = 'Non-Stationary' if p < 0.05 else 'Stationary'
    return {'kpss_stat': stat, 'kpss_p': p, 'kpss_lags': lags,
            'kpss_sig': SIG_STARS(p), 'kpss_conclusion': conclusion}

stat_rows = []
all_stat_cols = TARGET_COLS  # include refin_util_rate in ADF table
for col in all_stat_cols:
    adf  = run_adf(df[col], LABELS[col])
    kpss_res = run_kpss(df[col], LABELS[col])
    both_agree = (adf['adf_conclusion'] == 'Stationary' and
                  kpss_res['kpss_conclusion'] == 'Stationary')
    stat_rows.append({**adf, **kpss_res,
                      'overall': 'Stationary ✓' if both_agree
                                 else 'Conflict / Review'})

stat_df = pd.DataFrame(stat_rows)
print('\n── Stationarity Summary ──')
print(stat_df[['variable','adf_stat','adf_p','adf_conclusion',
               'kpss_stat','kpss_p','kpss_conclusion','overall']].to_string())


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  OLS ESTIMATION  (plain OLS — BEFORE correction)
# ═══════════════════════════════════════════════════════════════════════════════

ols = sm.OLS(y, X).fit()
resid  = ols.resid
fitted = ols.fittedvalues

print('\n' + '='*72)
print('OLS RESULTS (before HAC correction)')
print('='*72)
print(f'  R²={ols.rsquared:.4f}  Adj-R²={ols.rsquared_adj:.4f}  '
      f'F={ols.fvalue:.3f}  p={ols.f_pvalue:.4f}')


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  DIAGNOSTIC TESTS
# ═══════════════════════════════════════════════════════════════════════════════

# 5a. VIF
vif_vals = {}
for i, col in enumerate(X_raw.columns):
    vif_vals[col] = variance_inflation_factor(X_raw.values, i)

# 5b. Breusch-Pagan
bp_lm, bp_pval, bp_f, bp_fpval = het_breuschpagan(resid, X)

# 5c. Breusch-Godfrey
bg_lm, bg_pval, bg_f, bg_fpval = acorr_breusch_godfrey(ols, nlags=12)

# 5d. Durbin-Watson
dw = durbin_watson(resid)

# 5e. Jarque-Bera
jb_stat, jb_pval = stats.jarque_bera(resid)
skew_val  = float(stats.skew(resid))
kurt_val  = float(stats.kurtosis(resid))

# 5f. RESET test
reset_result = linear_reset(ols, power=3, use_f=True)
reset_f = reset_result.fvalue
reset_p = reset_result.pvalue

# 5g. Chow test — structural break 2014-01 (shale revolution)
break_date = '2014-01'
df_pre  = df.loc[:break_date]
df_post = df.loc[break_date:]
ols_pre  = sm.OLS(df_pre['wti_chg_pct'],
                  sm.add_constant(df_pre[REGRESSORS])).fit()
ols_post = sm.OLS(df_post['wti_chg_pct'],
                  sm.add_constant(df_post[REGRESSORS])).fit()
k = len(REGRESSORS) + 1
chow_num = (ols.ssr - ols_pre.ssr - ols_post.ssr) / k
chow_den = (ols_pre.ssr + ols_post.ssr) / (T - 2 * k)
chow_f   = chow_num / chow_den
chow_p   = 1 - f_dist.cdf(chow_f, k, T - 2 * k)

print(f'\n  Breusch-Pagan   : LM={bp_lm:.4f}  p={bp_pval:.4f}')
print(f'  Breusch-Godfrey : LM={bg_lm:.4f}  p={bg_pval:.4f}')
print(f'  Durbin-Watson   : {dw:.4f}')
print(f'  Jarque-Bera     : JB={jb_stat:.4f}  p={jb_pval:.4f}')
print(f'  RESET           : F={reset_f:.4f}  p={reset_p:.4f}')
print(f'  Chow (2014-01)  : F={chow_f:.4f}  p={chow_p:.4f}')

